#!/usr/bin/env python
"""
Step 1: Preprocessing

Generates all preprocess files needed by the ATM model (.pt files for topography,
temperature climatology, surface pressure, winds, and prescribed heating).

Usage:
    python scripts/01_preprocess.py --config config/experiments/my_exp.yaml
    python scripts/01_preprocess.py --config config/experiments/my_exp.yaml --force
    python scripts/01_preprocess.py --config config/experiments/my_exp.yaml --heating-only

Flags:
    --force          Regenerate all files even if the preprocess directory exists.
    --heating-only   Only regenerate the heating file (useful when the base
                     climatology files are already present but you want to try a
                     different heating scenario for the same resolution/season).

Output directory:
    FixedSeason: {preprocess_root}/preprocess__zw_{zw}__kmax_{kmax}_{season}_{y0}-{y1}/
    Gamma_AC   : {preprocess_root}/preprocess__zw_{zw}__kmax_{kmax}_annual_{y0}-{y1}/

Note (Gamma_AC only):
    The gamma distribution parameters (shapeAC.pt, scaleAC.pt) used by
    RunModel.Gamma.py are NOT generated here — they require fitting a gamma
    distribution to observed precipitation data and are experiment-specific.
    Generate them separately and place them in the preprocess directory.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import xarray as xr
import xesmf as xe
import yaml

from _config import load_config, build_preprocess_path
from _preprocess_common import (
    build_grid_params,
    season_month_indices,
    setup_spectral_transforms,
    vertical_structure,
    upsample_monthly_to_daily,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared: topography (identical for both model types)
# ---------------------------------------------------------------------------

def preprocess_topography(dlatlon, dsht, fullpath):
    print("  Processing topography...")
    url = "http://research.jisao.washington.edu/data_sets/elevation/elev.0.75-deg.nc"
    ds  = xr.open_dataset(url + "#mode=bytes", decode_times=False)
    del ds["time"]
    data = ds.data.squeeze()

    regridder    = xe.Regridder(data, dlatlon, "bilinear")
    topog_gg     = regridder(data) * 9.8
    topog_dev    = torch.from_numpy(np.where(topog_gg < 0.0, 0.0, topog_gg))
    coeffs       = dsht(topog_dev)

    torch.save(coeffs, os.path.join(fullpath, "topog.spectral.pt"))
    print("    Saved topog.spectral.pt")
    return coeffs, topog_gg.values if hasattr(topog_gg, "values") else topog_gg


# ---------------------------------------------------------------------------
# FixedSeason preprocessing functions
# ---------------------------------------------------------------------------

def mt_preprocess_temperature(cfg, dlatlon, dsht, sl, kmax, imax, jmax, fullpath):
    print("  Processing temperature climatology (FixedSeason)...")
    m0, m1, m2 = season_month_indices(cfg["season"])
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])

    ftemp = os.path.join(cfg["input_data_path"], "air.sig995.mon.mean.nc")
    Dtemp = xr.open_dataset(ftemp, autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    tsurf_climo  = Dtemp.air.groupby("time.month").mean(dim="time")
    tsurf_seas   = (tsurf_climo[m0] + tsurf_climo[m1] + tsurf_climo[m2]) / 3.0
    regridder    = xe.Regridder(tsurf_seas, dlatlon, "bilinear")
    tsurf_seas_g = regridder(tsurf_seas)

    rlaps, h0, tstrat = 6.8e-3, 8.2e3, 205.0
    temp_gg = np.zeros((kmax, jmax, imax))
    temp_gg[kmax - 1] = tsurf_seas_g.values + 273.16
    for k in np.arange(1, kmax, dtype=int):
        temp_gg[k] = temp_gg[kmax - 1] + h0 * rlaps * np.log(sl[k])
    temp_gg[:, :, imax - 1] = temp_gg[:, :, 0]
    temp_gg = np.where(temp_gg < tstrat, tstrat, temp_gg)

    coeffs = dsht(torch.from_numpy(temp_gg))
    torch.save(coeffs, os.path.join(fullpath, "temp.spectral.pt"))
    print("    Saved temp.spectral.pt")
    return coeffs


def mt_preprocess_surface_pressure(cfg, dlatlon, dsht, imax, fullpath):
    print("  Processing surface pressure climatology (FixedSeason)...")
    m0, m1, m2 = season_month_indices(cfg["season"])
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])

    fps  = os.path.join(cfg["input_data_path"], "pres.sfc.mon.mean.nc")
    Dps  = xr.open_dataset(fps, autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    psmean = Dps.pres.groupby("time.month").mean(dim="time")
    lnps   = np.log(psmean / (1000 * 100))
    lnps_seas = (lnps[m0] + lnps[m1] + lnps[m2]) / 3.0

    regridder = xe.Regridder(lnps_seas, dlatlon, "bilinear")
    lnps_g    = regridder(lnps_seas)
    lnps_g[:, imax - 1] = lnps_g[:, imax - 2]

    coeffs = dsht(torch.from_numpy(lnps_g.values))
    torch.save(coeffs, os.path.join(fullpath, "lnps.spectral.pt"))
    print("    Saved lnps.spectral.pt")
    return coeffs


def mt_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig, kmax, jmax, imax, fullpath):
    """
    FixedSeason prescribed heating. Two independent paths:

      heating_source == "custom": copy an already fully-computed heat.ggrid_*.pt
        tensor in as-is. No climatology, anomaly, or (a)/(b)/(c) below apply.

      heating_source in {cesm2, cca, cmap, cmorph}: build heating from
        (a) CLIMATOLOGY  -- NCEP precip.mon.mean.nc, averaged over
            start_year-end_year and the chosen Season.
        (b) ANOMALY      -- either a precomputed precip ANOMALY file (its
            own climatology already removed upstream, before it reaches
            this repo -- cesm2/cca are always this; cfg["anomaly_type"]
            defaults to "precomputed") OR, for cmap/cmorph only, a real
            raw-precip anomaly computed here by subtracting that same
            dataset's own climatology (cfg["anomaly_type"] == "compute"),
            mirroring gamma_preprocess_heating's CMAP-based approach.
        (c) MODIFICATIONS (both optional; omitted = today's unchanged
            behavior): a lat/lon region box (cfg["anomaly_lat_min"/"_max"/
            "anomaly_lon_min"/"_max"]) and/or a year composite
            (cfg["anomaly_years"]) -- applied uniformly whether the anomaly
            was precomputed or just computed above.
    """
    print(f"  Processing heating (FixedSeason, source={cfg['heating_source']})...")
    m0, m1, m2 = season_month_indices(cfg["season"])
    y0, y1     = str(cfg["start_year"]), str(cfg["end_year"])
    heating_name = cfg["heating_name"]
    outfile      = os.path.join(fullpath, f"heat.ggrid_{heating_name}.pt")

    source = cfg["heating_source"].lower()

    if source == "custom":
        src = cfg.get("heating_file")
        if not src:
            raise ValueError("heating_source=custom but no heating_file specified.")
        import shutil; shutil.copy(src, outfile)
        print(f"    Copied custom heating → {outfile}")
        return

    # ------------------------------------------------------------------
    # (b) ANOMALY -- load the anomaly source file. For cesm2/cca this is
    # always a precomputed anomaly, used as-is. For cmap/cmorph it may
    # instead be raw precip, with the anomaly computed just below.
    # ------------------------------------------------------------------
    if source == "cesm2":
        fprec = cfg.get("cesm2_precip_file")
        if not fprec:
            raise ValueError("heating_source=cesm2 but no cesm2_precip_file in config.")
        Dprec = xr.open_dataset(fprec, autoclose=True)
        rain_anom = Dprec["prec"]
    elif source == "cca":
        fprec = cfg.get("cca_precip_file")
        if not fprec:
            raise ValueError("heating_source=cca but no cca_precip_file in config.")
        Dprec = xr.open_dataset(fprec, autoclose=True)
        rain_anom = Dprec["prec"]
    elif source == "cmap":
        fprec = cfg.get("cmap_precip_file")
        if not fprec:
            raise ValueError("heating_source=cmap but no cmap_precip_file in config.")
        Dprec = xr.open_dataset(fprec, autoclose=True)
        rain_anom = Dprec["precip"]
    elif source == "cmorph":
        fprec = cfg.get("cmorph_precip_glob")
        if not fprec:
            raise ValueError("heating_source=cmorph but no cmorph_precip_glob in config.")
        Dprec = xr.open_mfdataset(fprec, autoclose=True)
        rain_anom = Dprec["cmorph"]
    else:
        raise ValueError(f"Unknown heating_source '{source}'. Use: cesm2, cca, cmap, cmorph, custom.")

    anomaly_type = cfg.get("anomaly_type", "precomputed")
    if anomaly_type == "compute":
        if source not in ("cmap", "cmorph"):
            raise ValueError(
                f"anomaly_type=compute is only supported for cmap/cmorph sources, got '{source}'."
            )
        # Anomaly relative to this SAME dataset's own climatology -- mirrors
        # gamma_preprocess_heating's CMAP-based approach exactly, rather than
        # mixing in the NCEP-reanalysis climatology from step (a) below.
        group_key = "time.month" if source == "cmap" else "time.dayofyear"
        clim = rain_anom.groupby(group_key).mean(dim="time")
        rain_anom = rain_anom.groupby(group_key) - clim
        print(f"    Anomaly computed from {source}'s own climatology (grouped by {group_key})")

    # ------------------------------------------------------------------
    # (c) MODIFICATIONS -- both optional; omitted == today's unchanged
    # behavior (rain_anom used exactly as loaded/computed above).
    # ------------------------------------------------------------------
    anomaly_years = cfg.get("anomaly_years")
    if anomaly_years:
        if "time" not in rain_anom.dims:
            raise ValueError(
                f"anomaly_years={anomaly_years!r} given but the {source} anomaly file "
                "has no 'time' dimension to select years from."
            )
        # Composite: for each requested year, average this Season's months
        # (same m0/m1/m2 used for the climatology below) within that year,
        # then average across years -- mirrors the enso_warm_years pattern
        # in gamma_preprocess_heating, adapted to fixed_season's Season
        # concept instead of one fixed month.
        season_months = [m0 + 1, m1 + 1, m2 + 1]  # xarray's time.month is 1-based
        composite = None
        for yr in anomaly_years:
            yr = str(yr)
            da_yr = rain_anom.sel(time=slice(f"{yr}-01-01", f"{yr}-12-31"))
            da_yr = da_yr.sel(time=da_yr["time.month"].isin(season_months))
            if da_yr.sizes.get("time", 0) == 0:
                raise ValueError(
                    f"anomaly_years: no time steps found for year {yr} "
                    f"(season={cfg['season']}) in the {source} anomaly file."
                )
            yr_mean = da_yr.mean(dim="time")
            composite = yr_mean if composite is None else composite + yr_mean
        rain_anom = composite / len(anomaly_years)
        print(f"    Anomaly composited over years {anomaly_years} "
              f"(season={cfg['season']} months only)")

    lat_min = cfg.get("anomaly_lat_min")
    lat_max = cfg.get("anomaly_lat_max")
    lon_min = cfg.get("anomaly_lon_min")
    lon_max = cfg.get("anomaly_lon_max")
    region_bounds = (lat_min, lat_max, lon_min, lon_max)
    if any(b is not None for b in region_bounds):
        if not all(b is not None for b in region_bounds):
            raise ValueError(
                "Partial region box given -- anomaly_lat_min/anomaly_lat_max/"
                "anomaly_lon_min/anomaly_lon_max must all be set together, or all "
                "left unset."
            )
        lat_cond = (rain_anom["lat"] >= lat_min) & (rain_anom["lat"] <= lat_max)
        lon_cond = (rain_anom["lon"] >= lon_min) & (rain_anom["lon"] <= lon_max)
        rain_anom = xr.where(lat_cond & lon_cond, rain_anom, 0.0)
        print(f"    Anomaly masked to region lat[{lat_min},{lat_max}] "
              f"lon[{lon_min},{lon_max}] (0-360 convention)")

    if "time" in rain_anom.dims:
        # anomaly_years is the only thing above that collapses a time
        # dimension. Without it, a genuinely multi-timestep source (any real
        # cmap/cmorph download, unlike cesm2/cca's historically pre-reduced
        # files) can't be combined with the single 2D climatology below --
        # fail clearly here rather than deep inside an xarray/regrid error.
        raise ValueError(
            f"The {source} anomaly source still has a 'time' dimension "
            f"({rain_anom.sizes['time']} steps) after modifications -- set "
            "anomaly_years to composite it down to a single field, or point at "
            "a file that's already a single 2D anomaly."
        )

    # ------------------------------------------------------------------
    # (a) CLIMATOLOGY -- NCEP precip.mon.mean.nc, Season mean over
    # start_year-end_year. Unchanged from before.
    # ------------------------------------------------------------------
    fclim     = os.path.join(cfg["input_data_path"], "precip.mon.mean.nc")
    Dclim     = xr.open_dataset(fclim, autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    prec_clim = Dclim.precip.groupby("time.month").mean(dim="time")
    prec_seas = (prec_clim[m0] + prec_clim[m1] + prec_clim[m2]) / 3.0

    regrid_anom = xe.Regridder(rain_anom, dlatlon, "bilinear")
    regrid_clim = xe.Regridder(prec_seas,  dlatlon, "bilinear")
    tmp = regrid_clim(prec_seas) + regrid_anom(rain_anom)
    tmp = np.where(tmp < 0.0, 0.0, tmp)

    dheat = xr.Dataset({"heat": (["lat", "lon"], tmp)},
                       coords={"lat": dlatlon["lat"], "lon": dlatlon["lon"]})
    globm = dheat.heat.mean(dim="lon").mean(dim="lat")
    tmp   = (dheat.heat - globm).values
    tropics = np.exp((-Lat * Lat) / 1000.0)
    tmp   = tropics * tmp
    tmp   = disht(dsht(torch.from_numpy(tmp))).numpy()

    vert_s = vertical_structure(kmax, delsig)
    Lv, rhow, Cp, Ps, grav = 2.5e6, 1000.0, 1005.0, 101325.0, 9.8
    beta = (Lv * rhow / Cp) * (grav / Ps) / (1000.0 * 86400.0)

    heat = torch.zeros((kmax, jmax, imax), dtype=torch.float64)
    for k in range(kmax):
        heat[k] = torch.from_numpy(tmp * vert_s[k] * beta)

    torch.save(heat, outfile)
    print(f"    Saved {os.path.basename(outfile)}")


def mt_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                        lnps_coeffs, sl, kmax, mw, zw, jmax, imax, fullpath):
    from subs1_utils import vortdivspec, gradq, press_to_sig

    print("  Processing wind/temperature climatology (FixedSeason)...")
    m0, m1, m2 = season_month_indices(cfg["season"])
    y0, y1     = str(cfg["start_year"]), str(cfg["end_year"])
    inpath     = cfg["input_data_path"]

    Duwnd = xr.open_dataset(os.path.join(inpath, "uwnd.mon.mean.nc"), autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    Dvwnd = xr.open_dataset(os.path.join(inpath, "vwnd.mon.mean.nc"), autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    Dair  = xr.open_dataset(os.path.join(inpath, "air.mon.mean.nc"),  autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))

    uwnd_clim = Duwnd.uwnd.groupby("time.month").mean(dim="time")
    vwnd_clim = Dvwnd.vwnd.groupby("time.month").mean(dim="time")
    air_clim  = Dair.air.groupby("time.month").mean(dim="time")
    obs_levels = np.flipud(Dair["level"].values)
    kobs       = len(obs_levels)

    lnps_gg = disht(lnps_coeffs)
    ps_gg   = torch.exp(lnps_gg) * 1000.0

    regridder   = xe.Regridder(Duwnd.uwnd, dlatlon, "bilinear")
    uwnd_seas   = (uwnd_clim[m0] + uwnd_clim[m1] + uwnd_clim[m2]) / 3.0
    vwnd_seas   = (vwnd_clim[m0] + vwnd_clim[m1] + vwnd_clim[m2]) / 3.0
    air_seas    = (air_clim[m0]  + air_clim[m1]  + air_clim[m2])  / 3.0

    upress_gg   = torch.zeros((kobs, jmax, imax), dtype=torch.float64)
    vpress_gg   = torch.zeros((kobs, jmax, imax), dtype=torch.float64)
    airpress_gg = torch.zeros((kobs, jmax, imax), dtype=torch.float64)
    for k in range(kobs):
        ki = kobs - k - 1
        upress_gg[ki]   = torch.from_numpy(regridder(uwnd_seas[k]).values)
        vpress_gg[ki]   = torch.from_numpy(regridder(vwnd_seas[k]).values)
        airpress_gg[ki] = torch.from_numpy(regridder(air_seas[k]).values) + 273.16
        for arr in (upress_gg, vpress_gg, airpress_gg):
            arr[ki, :, imax - 2] = arr[ki, :, imax - 3]
            arr[ki, :, imax - 1] = arr[ki, :, imax - 2]
            arr[ki, :, 0]        = arr[ki, :, 1]

    usig_gg = press_to_sig(kobs, imax, jmax, upress_gg,   obs_levels, ps_gg, sl, kmax)
    vsig_gg = press_to_sig(kobs, imax, jmax, vpress_gg,   obs_levels, ps_gg, sl, kmax)
    tsig_gg = press_to_sig(kobs, imax, jmax, airpress_gg, obs_levels, ps_gg, sl, kmax)
    tsig_gg = torch.where(tsig_gg < 205.0, 205.0, tsig_gg)

    for arr in (usig_gg, vsig_gg, tsig_gg):
        arr[:] = disht(dsht(arr))

    zmn, dmn       = vortdivspec(vsht, usig_gg, vsig_gg, kmax, mw, zw)
    vortsig_gg     = disht(zmn)
    divsig_gg      = disht(dmn)
    dxq_gg, dyq_gg = gradq(divsht, lnps_coeffs, mw, zw, imax, jmax)

    for name, arr in [("usig",    usig_gg),    ("vsig",    vsig_gg),
                      ("tsig",    tsig_gg),    ("vortsig", vortsig_gg),
                      ("divsig",  divsig_gg),  ("dxq_gg",  dxq_gg),
                      ("dyq_gg",  dyq_gg)]:
        torch.save(arr, os.path.join(fullpath, f"{name}.ggrid.pt"))
    print("    Saved usig, vsig, tsig, vortsig, divsig, dxq_gg, dyq_gg .ggrid.pt")


# ---------------------------------------------------------------------------
# Gamma_AC preprocessing functions
# ---------------------------------------------------------------------------

def _press_to_sig_batch(kmax, imax, jmax, press_data, press_levels, ps, slmodel, kmax_model):
    """
    Batch version of press_to_sig for Gamma_AC (processes 12 months simultaneously).
    press_data : (12, kmax, jmax, imax)
    ps         : (12, jmax, imax) surface pressure in mb
    Returns    : (12, kmax_model, jmax, imax)
    """
    sig_levels = torch.zeros((12, kmax, jmax, imax), dtype=torch.float64)
    sig_data   = torch.zeros((12, kmax_model, jmax, imax), dtype=torch.float64)
    slmap      = torch.zeros((12, kmax_model, jmax, imax), dtype=torch.float64)

    for k in range(kmax):
        sig_levels[:, k] = press_levels[k] / ps
    for k in range(kmax_model):
        slmap[:, k] = float(slmodel[k])

    for isig in range(kmax_model):
        for ipress in np.arange(kmax - 1, -1, -1, dtype=int):
            foo_up = (slmap[:, isig] > sig_levels[:, ipress - 1]).long()
            foo_dn = (slmap[:, isig] < sig_levels[:, ipress]).long()
            found  = ((foo_up + foo_dn) == 2).long()
            denom  = (torch.log(sig_levels[:, ipress])
                      - torch.log(sig_levels[:, ipress - 1]))
            numer1 = (torch.log(sig_levels[:, ipress])
                      - torch.log(slmap[:, isig]))
            numer2 = (torch.log(slmap[:, isig])
                      - torch.log(sig_levels[:, ipress - 1]))
            foo = (numer1 * press_data[:, ipress - 1]
                   + numer2 * press_data[:, ipress]) / denom
            sig_data[:, isig] = found * foo + (1 - found) * sig_data[:, isig]

    for isig in range(kmax_model):
        below = (slmap[:, isig] > sig_levels[:, kmax - 1]).long()
        sig_data[:, isig] = (below * press_data[:, kmax - 1]
                             + (1 - below) * sig_data[:, isig])
        above = (slmap[:, isig] < sig_levels[:, 0]).long()
        sig_data[:, isig] = (above * press_data[:, 0]
                             + (1 - above) * sig_data[:, isig])

    return sig_data


def gamma_preprocess_temperature(cfg, dlatlon, dsht, lats, lons, sl, kmax, imax, jmax, fullpath):
    """12-month climatology of surface temperature → daily spectral temperature (365 days)."""
    print("  Processing temperature climatology (Gamma_AC)...")
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])
    url   = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
             "Monthlies/surface/air.sig995.mon.mean.nc")
    Dtemp = xr.open_dataset(url, autoclose=True).sel(
        time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    tsurf_climo = Dtemp.air.groupby("time.month").mean(dim="time")   # (12, lat, lon)

    regridder      = xe.Regridder(tsurf_climo[0], dlatlon, "bilinear")
    tsurf_allmonths = regridder(tsurf_climo)    # (12, jmax, imax)

    rlaps, h0, tstrat = 6.8e-3, 8.2e3, 205.0
    temp_gg = np.zeros((12, kmax, jmax, imax))
    temp_gg[:, kmax - 1] = tsurf_allmonths.values + 273.16
    for k in np.arange(1, kmax, dtype=int):
        temp_gg[:, k] = temp_gg[:, kmax - 1] + h0 * rlaps * np.log(sl[k])
    temp_gg[:, :, :, imax - 1] = temp_gg[:, :, :, 0]
    temp_gg = np.where(temp_gg < tstrat, tstrat, temp_gg)

    # Upsample 12 monthly values → 365 daily values
    temp_daily = upsample_monthly_to_daily(temp_gg, "temp", lev_coord=sl,
                                            lats=lats, lons=lons)   # (365, kmax, jmax, imax)
    temp_coeffs = dsht(temp_daily)                                  # (365, kmax, mw, zw)
    torch.save(temp_coeffs, os.path.join(fullpath, "temp.spectral.pt"))
    print("    Saved temp.spectral.pt  [shape:", tuple(temp_coeffs.shape), "]")
    return temp_coeffs


def gamma_preprocess_surface_pressure(cfg, dlatlon, dsht, lats, lons, imax, fullpath):
    """12-month ln(ps) climatology → daily spectral lnps (365 days)."""
    print("  Processing surface pressure climatology (Gamma_AC)...")
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])
    url = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
           "Monthlies/surface_gauss/pres.sfc.mon.mean.nc")
    Dps    = xr.open_dataset(url, autoclose=True)
    psmean = Dps.pres.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")).groupby(
        "time.month").mean(dim="time")
    lnps   = np.log(psmean / (1000 * 100))

    regridder       = xe.Regridder(lnps, dlatlon, "bilinear")
    lnps_allmonths  = regridder(lnps)                     # (12, jmax, imax)
    lnps_allmonths.values[:, :, imax - 1] = lnps_allmonths.values[:, :, imax - 2]

    lnps_daily   = upsample_monthly_to_daily(lnps_allmonths.values, "lnps",
                                              coords_3d=False, lats=lats, lons=lons)
    lnps_coeffs  = dsht(lnps_daily)
    torch.save(lnps_coeffs, os.path.join(fullpath, "lnps.spectral.pt"))
    print("    Saved lnps.spectral.pt  [shape:", tuple(lnps_coeffs.shape), "]")
    return lnps_coeffs, lnps_allmonths.values


def gamma_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig, kmax, jmax, imax, fullpath):
    """
    Prescribed static heating from a composite of ENSO warm years (or another source).
    Writes heat.ggrid_{heating_name}.pt as a diagnostic artifact only — it is
    NOT loaded by RunModel.Gamma.py or RunModel.Gamma-noheating.py at runtime.
    The model's actual heating is entirely the stochastic gamma-distribution
    draw computed in latent_heat_release() from shapeAC.pt/scaleAC.pt (or
    shape_noheating.pt/scale_noheating.pt), independent of this file.
    """
    print(f"  Processing background heating (Gamma_AC, source={cfg['heating_source']})...")
    heating_name = cfg["heating_name"]
    outfile      = os.path.join(fullpath, f"heat.ggrid_{heating_name}.pt")

    source = cfg["heating_source"].lower()

    if source == "custom":
        src = cfg.get("heating_file")
        if not src:
            raise ValueError("heating_source=custom but no heating_file specified.")
        import shutil; shutil.copy(src, outfile)
        print(f"    Copied custom heating → {outfile}")
        return

    # Default: CMAP precipitation ENSO composite
    url   = ("http://psl.noaa.gov/thredds/dodsC/Datasets/cmap/enh/precip.mon.mean.nc")
    Dprec = xr.open_dataset(url, autoclose=True)
    prec_clim = Dprec.precip.groupby("time.month").mean(dim="time")
    prec_anom = Dprec.precip.groupby("time.month") - prec_clim

    # Default: composite of 1998 (strong El Niño) — override via config if needed
    enso_years  = cfg.get("enso_warm_years", ["1998"] * 13)
    anom        = prec_anom[0] * 0.0
    for yr in enso_years:
        anom = anom + prec_anom.sel(time=f"{yr}-02-01")
    rain_anom = anom / len(enso_years)

    regridder = xe.Regridder(rain_anom, dlatlon, "bilinear")
    tmp       = regridder(rain_anom).values
    tmp       = np.where(tmp < 0.0, 0.0, tmp)

    dheat = xr.Dataset({"heat": (["lat", "lon"], tmp)},
                       coords={"lat": dlatlon["lat"], "lon": dlatlon["lon"]})
    globm = dheat.heat.mean(dim="lon").mean(dim="lat")
    tmp   = (dheat.heat - globm).values
    tropics = np.exp((-Lat * Lat) / 1000.0)
    tmp   = tropics * tmp
    tmp   = disht(dsht(torch.from_numpy(tmp))).numpy()

    vert_s = vertical_structure(kmax, delsig)
    Lv, rhow, Cp, Ps, grav = 2.5e6, 1000.0, 1005.0, 101325.0, 9.8
    beta = (Lv * rhow / Cp) * (grav / Ps) / (1000.0 * 86400.0)

    heat = torch.zeros((kmax, jmax, imax), dtype=torch.float64)
    for k in range(kmax):
        heat[k] = tmp * vert_s[k] * beta

    torch.save(heat, outfile)
    print(f"    Saved {os.path.basename(outfile)}")


def gamma_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                           lnps_allmonths, lnps_coeffs,
                           sl, kmax, mw, zw, jmax, imax, lats, lons, fullpath):
    """
    12-month climatological winds (u, v, T, q) on sigma levels → daily (365 days).
    Also saves vorticity, divergence, and pressure gradient daily fields.
    """
    from subs1_utils import vortdivspec, gradq

    print("  Processing wind climatology (Gamma_AC)...")
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])
    url_u = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
             "Monthlies/pressure/uwnd.mon.mean.nc")
    url_v = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
             "Monthlies/pressure/vwnd.mon.mean.nc")
    url_t = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
             "Monthlies/pressure/air.mon.mean.nc")
    url_q = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
             "Monthlies/pressure/shum.mon.mean.nc")

    Duwnd = xr.open_dataset(url_u, autoclose=True)
    Dvwnd = xr.open_dataset(url_v, autoclose=True)
    Dair  = xr.open_dataset(url_t, autoclose=True)
    Dshum = xr.open_dataset(url_q, autoclose=True)

    sel  = dict(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    uwnd_clim = Duwnd.uwnd.sel(**sel).groupby("time.month").mean(dim="time")
    vwnd_clim = Dvwnd.vwnd.sel(**sel).groupby("time.month").mean(dim="time")
    air_clim  = Dair.air.sel(**sel).groupby("time.month").mean(dim="time")
    shum_clim = Dshum.shum.sel(**sel).groupby("time.month").mean(dim="time")

    # Pad specific humidity to match pressure levels (shum has fewer levels)
    shum_pad = air_clim * 0.0
    shum_pad[:, :8] = shum_clim

    obs_levels = np.flipud(Dair["level"].values)
    kobs       = len(obs_levels)

    # lnps on Gaussian grid → surface pressure in mb
    lnps_gg_12 = disht(lnps_coeffs)           # (365, jmax, imax)
    # Use the 12 monthly values for sigma interpolation
    lnps_monthly = torch.from_numpy(lnps_allmonths)   # (12, jmax, imax)
    ps_monthly   = torch.exp(lnps_monthly) * 1000.0   # mb

    regridder   = xe.Regridder(Duwnd.uwnd, dlatlon, "bilinear")
    upress_gg   = torch.zeros((12, kobs, jmax, imax), dtype=torch.float64)
    vpress_gg   = torch.zeros((12, kobs, jmax, imax), dtype=torch.float64)
    airpress_gg = torch.zeros((12, kobs, jmax, imax), dtype=torch.float64)
    shumpress_gg = torch.zeros((12, kobs, jmax, imax), dtype=torch.float64)

    for k in range(kobs):
        ki = kobs - k - 1
        upress_gg[:, ki]   = torch.from_numpy(regridder(uwnd_clim[:, k]).values)
        vpress_gg[:, ki]   = torch.from_numpy(regridder(vwnd_clim[:, k]).values)
        airpress_gg[:, ki] = torch.from_numpy(regridder(air_clim[:, k]).values)  + 273.16
        shumpress_gg[:, ki] = torch.from_numpy(regridder(shum_pad[:, k]).values)
        for arr in (upress_gg, vpress_gg, airpress_gg, shumpress_gg):
            arr[:, ki, :, imax - 2] = arr[:, ki, :, imax - 3]
            arr[:, ki, :, imax - 1] = arr[:, ki, :, imax - 2]
            arr[:, ki, :, 0]        = arr[:, ki, :, 1]

    print("    Interpolating pressure→sigma levels (12 months)...")
    usig_gg   = _press_to_sig_batch(kobs, imax, jmax, upress_gg,   obs_levels, ps_monthly, sl, kmax)
    vsig_gg   = _press_to_sig_batch(kobs, imax, jmax, vpress_gg,   obs_levels, ps_monthly, sl, kmax)
    tsig_gg   = _press_to_sig_batch(kobs, imax, jmax, airpress_gg, obs_levels, ps_monthly, sl, kmax)
    qsig_gg   = _press_to_sig_batch(kobs, imax, jmax, shumpress_gg, obs_levels, ps_monthly, sl, kmax)
    tsig_gg   = torch.where(tsig_gg < 205.0, 205.0, tsig_gg)

    # Spectral filter + vorticity/divergence/pressure gradient for each month
    vortsig_gg = torch.zeros((12, kmax, jmax, imax), dtype=torch.float64)
    divsig_gg  = torch.zeros((12, kmax, jmax, imax), dtype=torch.float64)
    dxq_gg     = torch.zeros((12, jmax, imax), dtype=torch.float64)
    dyq_gg     = torch.zeros((12, jmax, imax), dtype=torch.float64)

    for it in range(12):
        usig_gg[it]  = disht(dsht(usig_gg[it]))
        vsig_gg[it]  = disht(dsht(vsig_gg[it]))
        tsig_gg[it]  = disht(dsht(tsig_gg[it]))
        qsig_gg[it]  = disht(dsht(qsig_gg[it]))
        zmn_it, dmn_it      = vortdivspec(vsht, usig_gg[it], vsig_gg[it], kmax, mw, zw)
        vortsig_gg[it]      = disht(zmn_it)
        divsig_gg[it]       = disht(dmn_it)
        qmn_it              = lnps_coeffs[it] if lnps_coeffs.dim() > 2 else lnps_coeffs
        dxq_gg[it], dyq_gg[it] = gradq(divsht, qmn_it, mw, zw, imax, jmax)

    print("    Upsampling monthly → daily (365 days)...")
    fields_4d = [("usig",    usig_gg,    True,  sl),
                 ("vsig",    vsig_gg,    True,  sl),
                 ("tsig",    tsig_gg,    True,  sl),
                 ("qsig",    qsig_gg,    True,  sl),
                 ("vortsig", vortsig_gg, True,  sl),
                 ("divsig",  divsig_gg,  True,  sl)]
    fields_3d = [("dxq_gg",  dxq_gg,    False, None),
                 ("dyq_gg",  dyq_gg,    False, None)]

    for name, arr, is4d, lev in fields_4d:
        daily = upsample_monthly_to_daily(arr.numpy(), "data",
                                           lev_coord=lev, lats=lats, lons=lons)
        torch.save(daily, os.path.join(fullpath, f"{name}.ggrid.pt"))
        print(f"    Saved {name}.ggrid.pt  [shape: {tuple(daily.shape)}]")

    for name, arr, _, __ in fields_3d:
        daily = upsample_monthly_to_daily(arr.numpy(), "data",
                                           coords_3d=False, lats=lats, lons=lons)
        torch.save(daily, os.path.join(fullpath, f"{name}.ggrid.pt"))
        print(f"    Saved {name}.ggrid.pt  [shape: {tuple(daily.shape)}]")


def gamma_preprocess_landsea(topog_gg, lats, lons, fullpath):
    """Derive land-sea mask from topography and save it."""
    import numpy as _np
    landsea = _np.where(topog_gg <= 0.0, 0.0, 1.0)
    torch.save(torch.from_numpy(landsea), os.path.join(fullpath, "landsea.ggrid.pt"))
    print("    Saved landsea.ggrid.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATM workflow step 1: preprocessing")
    parser.add_argument("--config",        required=True, help="Path to YAML config file")
    parser.add_argument("--force",         action="store_true",
                        help="Regenerate all files even if the preprocess directory exists")
    parser.add_argument("--heating-only",  action="store_true",
                        help="Only regenerate the heating file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = cfg["model_type"]
    if model_type not in ("fixed_season", "gamma_ac"):
        raise ValueError(f"Unknown model_type: {model_type}")

    mw, jmax, imax = build_grid_params(cfg)
    zw   = cfg["zw"]
    kmax = cfg["kmax"]

    fullpath = build_preprocess_path(cfg)
    # Mirrors 02_run_model.py's own heating_file resolution exactly -- without
    # this, an experiment using heating_file_override (an existing file whose
    # name doesn't match heat.ggrid_{heating_name}.pt) would look incomplete
    # here even though it's already fully built, and step 1 would try (and
    # fail) to regenerate it via heating_source=custom with no heating_file set.
    heating_filename = cfg.get("heating_file_override") or f"heat.ggrid_{cfg['heating_name']}.pt"
    heating_file = os.path.join(fullpath, heating_filename)

    # Decide what to regenerate
    base_exists = (
        os.path.isdir(fullpath)
        and os.path.exists(os.path.join(fullpath, "temp.spectral.pt"))
        and os.path.exists(os.path.join(fullpath, "topog.spectral.pt"))
    )
    # heat.ggrid_{heating_name}.pt is only required for completeness for
    # fixed_season, which loads it every timestep. For gamma_ac it's a
    # write-only diagnostic artifact — RunModel.Gamma.py never loads it
    # (the model's actual heating is drawn at runtime from shapeAC.pt/
    # scaleAC.pt via latent_heat_release()) — so its absence shouldn't
    # block declaring the preprocess directory complete.
    heating_required_for_completeness = (model_type == "fixed_season")
    if base_exists and not args.force and not args.heating_only:
        if (not heating_required_for_completeness) or os.path.exists(heating_file):
            print(f"Preprocess directory already complete: {fullpath}")
            print("Use --force to regenerate all, or --heating-only to redo just the heating.")
            return
        print("Base files exist. Generating missing heating file only.")
        args.heating_only = True

    os.makedirs(fullpath, exist_ok=True)
    print(f"Preprocess directory: {fullpath}")

    # Point sys.path at the right subs1_utils
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir    = "Gamma_AC_Model" if model_type == "gamma_ac" else "FixedSeason_Model"
    sys.path.insert(0, os.path.join(project_root, model_dir))

    from subs1_utils import bscst
    delsig, si, sl, sikap, slkap, cth1, cth2, r1b, r2b = bscst(kmax)

    print("Setting up spectral transforms...")
    cost_lg, lats, lons, dlatlon, vsht, dsht, disht, dvsht, divsht = \
        setup_spectral_transforms(jmax, imax, mw, zw)
    _, Lat = np.meshgrid(lons, lats)

    # ---- FixedSeason ----
    if model_type == "fixed_season":
        if args.heating_only:
            mt_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig,
                                  kmax, jmax, imax, fullpath)
        else:
            _, topog_gg = preprocess_topography(dlatlon, dsht, fullpath)
            mt_preprocess_temperature(cfg, dlatlon, dsht, sl, kmax, imax, jmax, fullpath)
            lnps_coeffs = mt_preprocess_surface_pressure(cfg, dlatlon, dsht, imax, fullpath)
            mt_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig,
                                  kmax, jmax, imax, fullpath)
            mt_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                                lnps_coeffs, sl, kmax, mw, zw, jmax, imax, fullpath)

    # ---- Gamma_AC ----
    else:
        if args.heating_only:
            gamma_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig,
                                     kmax, jmax, imax, fullpath)
        else:
            _, topog_gg = preprocess_topography(dlatlon, dsht, fullpath)
            gamma_preprocess_landsea(topog_gg, lats, lons, fullpath)
            gamma_preprocess_temperature(cfg, dlatlon, dsht, lats, lons, sl,
                                         kmax, imax, jmax, fullpath)
            lnps_coeffs, lnps_allmonths = gamma_preprocess_surface_pressure(
                cfg, dlatlon, dsht, lats, lons, imax, fullpath)
            gamma_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig,
                                     kmax, jmax, imax, fullpath)
            gamma_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                                   lnps_allmonths, lnps_coeffs,
                                   sl, kmax, mw, zw, jmax, imax, lats, lons, fullpath)
            print()
            print("NOTE: gamma distribution parameters (shapeAC.pt, scaleAC.pt) are NOT")
            print("      generated here. Fit them from observed precipitation and place them")
            print("      in the preprocess directory before running 02_run_model.py.")

    print(f"\nPreprocessing complete.")
    print(f"  Directory : {fullpath}")
    print(f"  Heating   : heat.ggrid_{cfg['heating_name']}.pt")


if __name__ == "__main__":
    main()
