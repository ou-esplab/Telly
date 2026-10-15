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
    With a non-observed basic_state (Gamma_AC only: observed_sealevel, zonal_mean,
    landsea_contrast) the directory gets a "_<basic_state>" suffix, plus "_realtopo"
    with real topography (also applied to preprocess_path_override), and
    shape/scale/heat files are symlinked in from the observed directory. With real
    topography the surface pressure and sigma-level fields are built on the terrain.

Note (Gamma_AC only):
    The gamma distribution parameters (shapeAC.pt, scaleAC.pt) used by
    RunModel.Gamma.py are NOT generated here — they require fitting a gamma
    distribution to observed precipitation data and are experiment-specific.
    Generate them separately and place them in the preprocess directory.
"""

import argparse
import datetime
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import xarray as xr
import xesmf as xe
import yaml

from _config import load_config, build_preprocess_path, resolve_surface
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

OCEAN_RIPPLE_PHIS = 2000.0   # m^2/s^2 (~200 m): ocean-cell PHIS below this is ripple, not terrain


def _load_land_fraction(cfg, dlatlon):
    """ERA5 land-sea mask (fraction 0 ocean .. 1 land) on the model grid."""
    path = cfg["land_sea_mask_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"land_sea_mask_file not found: {path}")
    lsm = xr.open_dataset(path)["lsm"].squeeze(drop=True)
    if "longitude" in lsm.dims:
        lsm = lsm.rename({"longitude": "lon", "latitude": "lat"})
    if float(lsm.lon.min()) < 0.0:
        lsm = lsm.assign_coords(lon=(lsm.lon.values % 360.0)).sortby("lon")
    return xe.Regridder(lsm, dlatlon, "bilinear")(lsm).values


def preprocess_topography(cfg, dlatlon, dsht, fullpath, basic_state="observed"):
    """Save topog.spectral.pt (real PHIS). For any non-observed basic state also
    save topog_flat.spectral.pt (all zeros), which the model reads when
    topography=flat.

    Source: cfg["topography_file"] (gridded surface geopotential, m^2/s^2; the JISAO
    elevation server this used to download from is gone). That file is itself spectrally
    smoothed, so it has small +/- ripples over open ocean. Negative values are clipped, and
    ocean cells (<50% land in the ERA5 mask) with PHIS below OCEAN_RIPPLE_PHIS are set to
    zero. Real coastal mountains (e.g. the Andes) exceed that threshold and are kept; a
    plain "zero everything the mask calls ocean" would lower the Andes by ~1 km.

    Returns (coeffs, topog_gg, land_mask): the real gridded PHIS (used for the sea-level
    T/ps reduction) and the binary ERA5 land mask (land-sea contrast + landsea.ggrid.pt)."""
    print("  Processing topography...")
    path = cfg["topography_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"topography_file not found: {path}")
    ds   = xr.open_dataset(path)
    data = ds["topog"]

    regridder    = xe.Regridder(data, dlatlon, "bilinear")
    topog_gg     = regridder(data).values
    land_frac    = _load_land_fraction(cfg, dlatlon)
    land_mask    = (land_frac > 0.5).astype(np.float64)
    topog_gg     = np.clip(topog_gg, 0.0, None)
    topog_gg     = np.where((land_mask == 0.0) & (topog_gg < OCEAN_RIPPLE_PHIS), 0.0, topog_gg)
    topog_dev    = torch.from_numpy(topog_gg)
    coeffs       = dsht(topog_dev)

    torch.save(coeffs, os.path.join(fullpath, "topog.spectral.pt"))
    print("    Saved topog.spectral.pt")
    if basic_state != "observed":
        torch.save(torch.zeros_like(coeffs), os.path.join(fullpath, "topog_flat.spectral.pt"))
        print("    Saved topog_flat.spectral.pt (PHIS = 0)")
    return coeffs, topog_gg.copy(), land_mask


def zonal_mean_grid(arr):
    """Replace a gridded field (numpy array or torch tensor, longitude last)
    with its zonal mean, broadcast back to the full grid."""
    if isinstance(arr, torch.Tensor):
        return arr.mean(dim=-1, keepdim=True).expand_as(arr).clone()
    arr = np.asarray(arr)
    return np.broadcast_to(arr.mean(axis=-1, keepdims=True), arr.shape).copy()


def smoothed_land_mask(land_mask, npass=3):
    """Binary land mask on the Gaussian grid softened with a periodic-in-longitude
    3x3 box filter so the land-sea temperature step does not ring after the
    spectral truncation."""
    mask = np.where(np.asarray(land_mask) > 0.5, 1.0, 0.0)
    for _ in range(npass):
        padded = np.pad(mask, ((1, 1), (0, 0)), mode="edge")
        rows = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        mask = (np.roll(rows, 1, axis=1) + rows + np.roll(rows, -1, axis=1)) / 3.0
    return np.clip(mask, 0.0, 1.0)


def landsea_blend(field, land_mask, level_weight=None):
    """Zonal mean + a land-masked departure from it, with the departure's own
    zonal mean removed so the zonal-mean state is exactly unchanged (a pure
    stationary-eddy forcing: land-sea contrast, no zonal-mean temperature change).
    Longitude is the last axis; land_mask is (jmax, imax) and level_weight (optional,
    e.g. (nlev, 1, 1)) broadcast against the departure. Accepts numpy or torch."""
    zm = zonal_mean_grid(field)
    if isinstance(field, torch.Tensor):
        land_mask = torch.as_tensor(land_mask, dtype=field.dtype)
        if level_weight is not None:
            level_weight = torch.as_tensor(level_weight, dtype=field.dtype)
    anom = land_mask * (field - zm)
    if level_weight is not None:
        anom = anom * level_weight
    return zm + anom - zonal_mean_grid(anom)


def link_shared_gamma_files(cfg, fullpath):
    """Non-observed basic states use their own preprocess directory, but the
    gamma shape/scale parameter files and heating files don't depend on the
    basic state, so symlink them from the observed directory (existing files
    in the basic-state directory are left alone)."""
    if resolve_surface(cfg)[1] == "observed" or not os.path.isdir(fullpath):
        return
    base = build_preprocess_path({**cfg, "surface_mode": "realistic",
                                  "topography": "real", "basic_state": "observed"})
    if not os.path.isdir(base):
        print(f"  NOTE: observed preprocess directory not found ({base}); "
              "shape/scale files were not linked.")
        return
    nlinked = 0
    for fname in sorted(os.listdir(base)):
        if fname.endswith(".pt") and fname.startswith(("shape", "scale", "heat")):
            dst = os.path.join(fullpath, fname)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(base, fname), dst)
                nlinked += 1
    if nlinked:
        print(f"  Linked {nlinked} shape/scale/heat files from {base}")


def ncep_source(cfg, local_name, url):
    """Path of an NCEP monthly file: the local copy in cfg["input_data_path"] if
    present (same files fixed_season already uses; fast and independent of the
    NOAA server), otherwise the THREDDS URL."""
    local = os.path.join(cfg.get("input_data_path") or "", local_name)
    if cfg.get("input_data_path") and os.path.exists(local):
        print(f"    Using local {local_name}")
        return local
    print(f"    {local_name} not found locally; downloading from NOAA")
    return url


def surface_pressure_from_slp(slp_pa, t0_k, phis):
    """Surface pressure on terrain of surface geopotential `phis` under a sea-level
    pressure `slp_pa` and sea-level temperature `t0_k`, for a constant lapse-rate
    (6.5 K/km) atmosphere -- the balanced inverse of the sea-level reduction, used so
    that real topography sits on a hydrostatically consistent surface pressure."""
    gam, rgas, grav = 0.0065, 287.04, 9.8
    z = phis / grav
    return slp_pa * (1.0 - gam * z / t0_k) ** (grav / (rgas * gam))


def sea_level_pressure(ps_pa, tsurf_k, phis):
    """Reduce surface pressure to sea level with the ECMWF formula, the same one
    scripts/03_postprocess.py:compute_slp uses (lapse rate -6.5 K/km)."""
    laps, rgas = -0.0065, 287.04
    y = laps * phis / (rgas * tsurf_k * tsurf_k)
    tstar = tsurf_k * (1.0 + y / 2.0 + y ** 2 / 3.0)
    return ps_pa * np.exp(phis / (rgas * tstar) * (1.0 - y / 2.0 + y ** 2 / 3.0))


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


def _recipe_path(pt_path):
    return os.path.splitext(pt_path)[0] + ".config.yaml"


def _write_heating_recipe(cfg, source, outfile, fullpath):
    # A minimal, self-sufficient record of what produced this heating file --
    # exactly the cfg keys that determine it, reusable as a --config input to
    # regenerate the same file. Written next to the .pt so it travels with
    # the file if it's later copied/renamed (see the "custom" branch below).
    recipe = {
        # mt_preprocess_heating is fixed_season-only, so this is a constant --
        # included so the recipe is a standalone, runnable --heating-only
        # --config on its own (main() requires model_type to be present).
        "model_type": "fixed_season",
        "heating_source": source,
        "heating_name": cfg["heating_name"],
        "season": cfg["season"],
        "start_year": cfg["start_year"],
        "end_year": cfg["end_year"],
        "input_data_path": cfg["input_data_path"],
        "preprocess_path_override": fullpath,
        "zw": cfg.get("zw"),
        "kmax": cfg.get("kmax"),
    }
    for key in ("cesm2_precip_file", "cca_precip_file", "cmap_precip_file",
                "cmorph_precip_glob", "anomaly_type", "anomaly_years",
                "anomaly_lat_min", "anomaly_lat_max", "anomaly_lon_min", "anomaly_lon_max"):
        if cfg.get(key) is not None:
            recipe[key] = cfg[key]
    recipe["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(_recipe_path(outfile), "w") as f:
        yaml.safe_dump(recipe, f, sort_keys=False)


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
        src_recipe = _recipe_path(src)
        if os.path.exists(src_recipe):
            shutil.copy(src_recipe, _recipe_path(outfile))
            print(f"    Copied source's heating recipe → {_recipe_path(outfile)}")
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
    _write_heating_recipe(cfg, source, outfile, fullpath)
    print(f"    Saved {os.path.basename(_recipe_path(outfile))}")


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


def load_sig995_months(cfg, dlatlon):
    """12-month climatology of NCEP sig995 air temperature (degC) on the model grid."""
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])
    url = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
           "Monthlies/surface/air.sig995.mon.mean.nc")
    src = ncep_source(cfg, "air.sig995.mon.mean.nc", url)
    Dtemp = xr.open_dataset(src, autoclose=True).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    tsurf_climo = Dtemp.air.groupby("time.month").mean(dim="time")   # (12, lat, lon)
    regridder = xe.Regridder(tsurf_climo[0], dlatlon, "bilinear")
    return regridder(tsurf_climo)


def gamma_preprocess_temperature(cfg, dlatlon, dsht, lats, lons, sl, kmax, imax, jmax, fullpath,
                                 topog_gg=None, land_mask=None):
    """12-month climatology of surface temperature → daily spectral temperature (365 days).

    basic_state != observed: the sig995 temperature is reduced to sea level with the
    same lapse rate used for the vertical extension (so it is consistent with PHIS = 0
    and the sea-level-pressure ps). zonal_mean zonally averages it; landsea_contrast
    keeps only its land-masked departure from the zonal mean."""
    print("  Processing temperature climatology (Gamma_AC)...")
    topography, basic_state = resolve_surface(cfg)
    tsurf_allmonths = load_sig995_months(cfg, dlatlon)    # (12, jmax, imax), degC

    rlaps, h0, tstrat = 6.8e-3, 8.2e3, 205.0
    tsurf_vals = tsurf_allmonths.values
    if basic_state != "observed":
        if topog_gg is None:
            raise ValueError("topog_gg is required to reduce surface T to sea level.")
        # topog_gg is PHIS (m^2/s^2); /9.8 -> elevation in m
        tsurf_vals = tsurf_vals + rlaps * (np.asarray(topog_gg) / 9.8)
        if basic_state == "zonal_mean":
            tsurf_vals = zonal_mean_grid(tsurf_vals)
        elif basic_state == "landsea_contrast":
            tsurf_vals = landsea_blend(tsurf_vals, land_mask)
        if topography == "real":
            # sea-level state placed on real terrain: actual surface T is colder aloft
            tsurf_vals = tsurf_vals - rlaps * (np.asarray(topog_gg) / 9.8)
    temp_gg = np.zeros((12, kmax, jmax, imax))
    temp_gg[:, kmax - 1] = tsurf_vals + 273.16
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


def gamma_preprocess_surface_pressure(cfg, dlatlon, dsht, lats, lons, imax, fullpath,
                                      topog_gg=None):
    """12-month ln(ps) climatology → daily spectral lnps (365 days).

    basic_state != observed: PHIS = 0, so the basic-state ps must carry no orographic
    signal. The observed surface pressure is reduced to sea level (same ECMWF formula
    as the SLP postprocessing, using sig995 T and the real PHIS) instead of downloading
    a separate sea-level-pressure product."""
    print("  Processing surface pressure climatology (Gamma_AC)...")
    topography, basic_state = resolve_surface(cfg)
    y0, y1 = str(cfg["start_year"]), str(cfg["end_year"])
    url = ("http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
           "Monthlies/surface_gauss/pres.sfc.mon.mean.nc")
    src    = ncep_source(cfg, "pres.sfc.mon.mean.nc", url)
    Dps    = xr.open_dataset(src, autoclose=True)
    psmean = Dps.pres.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")).groupby(
        "time.month").mean(dim="time")                                   # (12, lat, lon), Pa

    if basic_state == "observed":
        lnps = np.log(psmean / (1000 * 100))
        regridder       = xe.Regridder(lnps, dlatlon, "bilinear")
        lnps_allmonths  = regridder(lnps)                     # (12, jmax, imax)
        lnps_allmonths.values[:, :, imax - 1] = lnps_allmonths.values[:, :, imax - 2]
    else:
        if topog_gg is None:
            raise ValueError("topog_gg is required to reduce surface pressure to sea level.")
        print("    basic_state=%s: reducing surface pressure to sea level" % basic_state)
        regridder = xe.Regridder(psmean, dlatlon, "bilinear")
        ps_g      = regridder(psmean)                         # (12, jmax, imax), Pa
        # The model's polar rows (+-88.57) lie just outside the source Gaussian grid
        # (+-88.54), so the regridder returns 0 there; copy the adjacent row instead.
        for row, nbr in ((0, 1), (-1, -2)):
            if (ps_g.values[:, row] <= 0.0).all():
                ps_g.values[:, row] = ps_g.values[:, nbr]
        if (ps_g.values <= 0.0).any():
            raise ValueError("Regridded surface pressure still has non-positive values.")
        tsurf_k   = load_sig995_months(cfg, dlatlon).values + 273.16
        phis      = np.asarray(topog_gg)
        slp_g     = sea_level_pressure(ps_g.values, tsurf_k, phis)
        lnps_allmonths = np.log(ps_g * 0.0 + slp_g / (1000 * 100))   # keep xarray container
        lnps_allmonths.values[:, :, imax - 1] = lnps_allmonths.values[:, :, imax - 2]
        if basic_state in ("zonal_mean", "landsea_contrast"):
            lnps_allmonths.values[...] = zonal_mean_grid(lnps_allmonths.values)   # zonal slp
        if topography == "real":
            # Real terrain needs a hydrostatically consistent surface pressure: reduce the
            # (zonal) sea-level pressure to the actual surface height. Without this the
            # mountains sit on a ~1010 hPa surface and the model blows up in the first day.
            t0 = tsurf_k + 0.0065 * phis / 9.8                 # sea-level temperature
            if basic_state in ("zonal_mean", "landsea_contrast"):
                t0 = zonal_mean_grid(t0)
            ps_topo = surface_pressure_from_slp(np.exp(lnps_allmonths.values) * 1.0e5, t0, phis)
            print("    topography=real: surface pressure placed on the terrain "
                  f"(min {ps_topo.min()/100:.0f} hPa)")
            lnps_allmonths.values[...] = np.log(ps_topo / 1.0e5)
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

    Reads cfg["diagnostic_heating_source"]/cfg["diagnostic_heating_file"] --
    deliberately separate key names from fixed_season's heating_source/
    heating_file (which control the model's REAL, load-bearing heating) so a
    saved config can't be misread as describing the real heating mechanism.
    """
    print(f"  Processing background heating (Gamma_AC, source={cfg['diagnostic_heating_source']})...")
    heating_name = cfg["heating_name"]
    outfile      = os.path.join(fullpath, f"heat.ggrid_{heating_name}.pt")

    source = cfg["diagnostic_heating_source"].lower()

    if source == "custom":
        src = cfg.get("diagnostic_heating_file")
        if not src:
            raise ValueError("diagnostic_heating_source=custom but no diagnostic_heating_file specified.")
        import shutil; shutil.copy(src, outfile)
        print(f"    Copied custom heating → {outfile}")
        return

    # Default: CMAP precipitation ENSO composite
    url   = ("http://psl.noaa.gov/thredds/dodsC/Datasets/cmap/enh/precip.mon.mean.nc")
    Dprec = xr.open_dataset(ncep_source(cfg, "precip.mon.mean.nc", url), autoclose=True)
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
        heat[k] = torch.as_tensor(np.asarray(tmp * vert_s[k] * beta), dtype=torch.float64)

    torch.save(heat, outfile)
    print(f"    Saved {os.path.basename(outfile)}")


def gamma_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                           lnps_allmonths, lnps_coeffs,
                           sl, kmax, mw, zw, jmax, imax, lats, lons, fullpath,
                           land_mask=None):
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

    _, basic_state = resolve_surface(cfg)
    Duwnd = xr.open_dataset(ncep_source(cfg, "uwnd.mon.mean.nc", url_u), autoclose=True)
    Dvwnd = xr.open_dataset(ncep_source(cfg, "vwnd.mon.mean.nc", url_v), autoclose=True)
    Dair  = xr.open_dataset(ncep_source(cfg, "air.mon.mean.nc", url_t), autoclose=True)
    # Specific humidity is only written to qsig.ggrid.pt; the model never reads it.
    # There is no local copy, so skip the download for the idealized basic states.
    do_humidity = basic_state == "observed"
    Dshum = xr.open_dataset(url_q, autoclose=True) if do_humidity else None

    sel  = dict(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    uwnd_clim = Duwnd.uwnd.sel(**sel).groupby("time.month").mean(dim="time")
    vwnd_clim = Dvwnd.vwnd.sel(**sel).groupby("time.month").mean(dim="time")
    air_clim  = Dair.air.sel(**sel).groupby("time.month").mean(dim="time")
    # Pad specific humidity to match pressure levels (shum has fewer levels)
    shum_pad = air_clim * 0.0
    if do_humidity:
        shum_clim = Dshum.shum.sel(**sel).groupby("time.month").mean(dim="time")
        shum_pad[:, :8] = shum_clim
    else:
        print("    Skipping specific humidity (unused by the model; qsig.ggrid.pt not written)")

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

    _, basic_state = resolve_surface(cfg)
    if basic_state in ("zonal_mean", "landsea_contrast"):
        # Zonally symmetric basic state (ps is already zonal-mean, so the
        # sigma interpolation below stays zonally symmetric too).
        print(f"    {basic_state}: zonally averaging u, v, T, q")
        upress_gg   = zonal_mean_grid(upress_gg)
        vpress_gg   = zonal_mean_grid(vpress_gg)
        shumpress_gg = zonal_mean_grid(shumpress_gg)
        if basic_state == "landsea_contrast":
            # Land-sea thermal contrast lives only in the temperature relaxation
            # target: zonal mean + the observed departure over (smoothed) land.
            # Winds/vorticity/divergence/ps stay zonal, so any stationary-wave
            # response has to be generated by the forcing, not imposed.
            if land_mask is None:
                raise ValueError("land_mask is required for basic_state=landsea_contrast.")
            print("    landsea_contrast: land-masked T asymmetry in the temperature target "
                  "(full strength >= 300 hPa, tapered to zero at 100 hPa)")
            # Stratospheric asymmetries (e.g. polar-vortex displacement) are a response to
            # tropospheric wave forcing, not a land-sea effect, so they are not imposed.
            w = np.clip((np.asarray(obs_levels, dtype=float) - 100.0) / 200.0, 0.0, 1.0)
            airpress_gg = landsea_blend(airpress_gg, land_mask, w[:, None, None])
        else:
            airpress_gg = zonal_mean_grid(airpress_gg)

    print("    Interpolating pressure→sigma levels (12 months)...")
    usig_gg   = _press_to_sig_batch(kobs, imax, jmax, upress_gg,   obs_levels, ps_monthly, sl, kmax)
    vsig_gg   = _press_to_sig_batch(kobs, imax, jmax, vpress_gg,   obs_levels, ps_monthly, sl, kmax)
    tsig_gg   = _press_to_sig_batch(kobs, imax, jmax, airpress_gg, obs_levels, ps_monthly, sl, kmax)
    qsig_gg   = (_press_to_sig_batch(kobs, imax, jmax, shumpress_gg, obs_levels, ps_monthly, sl, kmax)
                 if do_humidity else None)
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
        if do_humidity:
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
                 ("vortsig", vortsig_gg, True,  sl),
                 ("divsig",  divsig_gg,  True,  sl)]
    if do_humidity:
        fields_4d.insert(3, ("qsig", qsig_gg, True, sl))
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


def gamma_preprocess_landsea(land_mask, lats, lons, fullpath, basic_state="observed"):
    """Save the ERA5-derived land-sea mask. A zonal_mean basic state has no land,
    so its mask is all ocean (0)."""
    import numpy as _np
    landsea = _np.asarray(land_mask, dtype=float)
    if basic_state == "zonal_mean":
        landsea = _np.zeros_like(landsea)
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

    topography, basic_state = resolve_surface(cfg)
    if basic_state != "observed" and model_type != "gamma_ac":
        raise NotImplementedError(
            f"basic_state={basic_state} is only implemented for model_type=gamma_ac.")

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
            link_shared_gamma_files(cfg, fullpath)   # pick up shape/scale files added since
            print(f"Preprocess directory already complete: {fullpath}")
            print("Use --force to regenerate all, or --heating-only to redo just the heating.")
            return
        print("Base files exist. Generating missing heating file only.")
        args.heating_only = True

    os.makedirs(fullpath, exist_ok=True)
    print(f"Preprocess directory: {fullpath}  (topography={topography}, basic_state={basic_state})")

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
            preprocess_topography(cfg, dlatlon, dsht, fullpath)
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
            _, topog_gg, land_mask_bin = preprocess_topography(cfg, dlatlon, dsht, fullpath, basic_state)
            gamma_preprocess_landsea(land_mask_bin, lats, lons, fullpath, basic_state)
            land_mask = smoothed_land_mask(land_mask_bin) if basic_state == "landsea_contrast" else None
            gamma_preprocess_temperature(cfg, dlatlon, dsht, lats, lons, sl,
                                         kmax, imax, jmax, fullpath, topog_gg=topog_gg,
                                         land_mask=land_mask)
            lnps_coeffs, lnps_allmonths = gamma_preprocess_surface_pressure(
                cfg, dlatlon, dsht, lats, lons, imax, fullpath, topog_gg=topog_gg)
            gamma_preprocess_heating(cfg, dlatlon, dsht, disht, Lat, delsig,
                                     kmax, jmax, imax, fullpath)
            gamma_preprocess_winds(cfg, dlatlon, vsht, dsht, disht, divsht,
                                   lnps_allmonths, lnps_coeffs,
                                   sl, kmax, mw, zw, jmax, imax, lats, lons, fullpath,
                                   land_mask=land_mask)
            link_shared_gamma_files(cfg, fullpath)
            print()
            print("NOTE: gamma distribution parameters (shapeAC.pt, scaleAC.pt) are NOT")
            print("      generated here. Fit them from observed precipitation and place them")
            print("      in the preprocess directory before running 02_run_model.py.")

    print(f"\nPreprocessing complete.")
    print(f"  Directory : {fullpath}")
    print(f"  Heating   : heat.ggrid_{cfg['heating_name']}.pt")


if __name__ == "__main__":
    main()
