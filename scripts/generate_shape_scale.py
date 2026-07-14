#!/usr/bin/env python
"""
Fit gamma-distribution shape/scale parameters for the Gamma_AC model from
daily precipitation data, including composite-years (e.g. El Nino-style)
fits — ported from the manually-run notebook cells in
Gamma_AC_Model/reference_notebooks/preprocess.Gamma_heating.ipynb into
reusable, general-purpose code.

Output is compatible with RunModel.Gamma.py's --shapefile/--scalefile
(via 02_run_model.py's shape_file_override/scale_file_override config keys):
tensors of shape (365, jmax, imax), dtype float64, saved with torch.save.

Usage (CLI):
    python scripts/generate_shape_scale.py \\
        --precip-glob '/data/esplab/shared/obs/gridded/atm/precip/daily/CMORPH/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc' \\
        --precip-varname cmorph \\
        --start-date 1998-01-01 --end-date 2020-12-31 \\
        --zw 63 --output-dir /path/to/preprocess/dir --name MyControl

    # Composite (e.g. El Nino) fit — repeat --composite-window per event:
    python scripts/generate_shape_scale.py \\
        --precip-glob '...' --precip-varname cmorph \\
        --start-date 1998-01-01 --end-date 2020-12-31 \\
        --composite-window 2002-07-01:2003-06-30 \\
        --composite-window 2009-07-01:2010-06-30 \\
        --composite-window 2015-07-01:2016-06-30 \\
        --scale-qc-max 300 \\
        --zw 63 --output-dir /path/to/preprocess/dir --name MyWarm

    # No-heating (all-zero) case:
    python scripts/generate_shape_scale.py --noheating --zw 63 \\
        --output-dir /path/to/preprocess/dir --name MyNoHeating

Also importable directly:
    from generate_shape_scale import fit_gamma_shape_scale, zero_shape_scale
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
import xesmf as xe

from _preprocess_common import build_grid_params, upsample_monthly_to_daily

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "Gamma_AC_Model"))


def _build_grid(jmax, imax):
    """lat/lon/dlatlon only — no spectral transform objects, not needed here."""
    from subs1_utils import precompute_latitudes
    _, _, lats = precompute_latitudes(jmax)
    lats = 90 - 180 * lats / np.pi
    lons = np.linspace(0.0, 360.0 - 360.0 / imax, imax)
    dlatlon = xr.Dataset({"lat": lats, "lon": lons})
    return lats, lons, dlatlon


def _method_of_moments(mean_da, var_da, scale_qc_max=None):
    """
    shape = (mean/sigma)^2, scale = variance/mean.
    Non-finite -> 0, not just NaN -> 0: a location with zero variance but
    non-zero mean (e.g. constant precip) gives sigma=0 -> shape = inf, not
    nan, which np.isnan alone would miss and which then corrupts the
    downstream cubic interpolation.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(var_da)
        shape = (mean_da / sigma) ** 2
        scale = var_da / mean_da
    shape = np.where(np.isfinite(shape), shape, 0.0)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    if scale_qc_max is not None:
        scale = np.where(scale > scale_qc_max, 0.0, scale)
    return shape, scale


def _derive_shift(composite_windows, dummy_leap_year):
    """
    Derive the day-of-year circular-shift needed to realign a composite
    built from windows that don't start on Jan 1 back to a Jan-1-start
    daily cycle. All windows must share the same (month, day) start —
    raise rather than silently guess if they don't.
    """
    starts = {(pd.Timestamp(w[0]).month, pd.Timestamp(w[0]).day) for w in composite_windows}
    if len(starts) != 1:
        raise ValueError(
            f"All composite_windows must start on the same month/day; got {sorted(starts)}."
        )
    month, day = starts.pop()
    return pd.Timestamp(year=dummy_leap_year, month=month, day=day).dayofyear - 1


def fit_gamma_shape_scale(precip_glob, precip_varname, date_range, zw, output_dir, name,
                           composite_windows=None, scale_qc_max=None, dummy_leap_year=1952):
    """
    Fit gamma-distribution shape/scale parameters to daily precipitation.

    precip_glob      : file glob/path for daily precip data (xr.open_mfdataset)
    precip_varname   : variable name within the dataset (e.g. "cmorph")
    date_range       : (start, end) date strings. Must cover both the baseline
                        climatology period AND any composite_windows below —
                        this is intentionally explicit rather than "whatever
                        files happen to be on disk" (the notebook this was
                        ported from had that fragility).
    zw               : spectral resolution (42/63/124) -> target grid.
    output_dir       : directory to write shape_{name}.pt / scale_{name}.pt into.
    name             : label used in the output filename.
    composite_windows: optional list of (start, end) date-string tuples, e.g.
                        real El Nino event windows. If given, fits to the
                        day-of-year climatology of ONLY these composited
                        windows (anomalies computed against the date_range
                        baseline), circularly shifted back to a Jan-1-start
                        daily cycle. If None/empty, fits directly to the full
                        date_range instead (the "control" case).
    scale_qc_max     : if given, any fitted scale value above this threshold
                        is set to 0 (despiking for small-sample composites).
                        None = no clipping.
    dummy_leap_year  : reference calendar year for the internal triple-year
                        cubic-upsample trick; must be a leap year.

    Returns (shape_tensor, scale_tensor), each shape (365, jmax, imax),
    dtype float64. Also written to output_dir as shape_{name}.pt / scale_{name}.pt.
    """
    mw, jmax, imax = build_grid_params({"zw": zw})
    lats, lons, dlatlon = _build_grid(jmax, imax)

    precip_ds = xr.open_mfdataset(precip_glob, autoclose=True)
    precip_da = precip_ds[precip_varname]
    precip_vals = np.where(np.isnan(precip_da.values), 0.0, precip_da.values)
    precip_da = precip_da.copy(data=precip_vals)

    regridder = xe.Regridder(precip_da, dlatlon, "bilinear")
    precip_gg = regridder(precip_da)

    drain = xr.Dataset(
        {"precip": (["time", "lat", "lon"], precip_gg.values)},
        coords={"time": precip_da["time"], "lat": lats, "lon": lons},
    )
    drain = drain.sel(time=slice(date_range[0], date_range[1]))

    mean = drain.precip.groupby("time.dayofyear").mean(dim="time")
    anom = drain.precip.groupby("time.dayofyear") - mean

    if not composite_windows:
        var = (anom * anom).groupby("time.dayofyear").mean(dim="time")
        mean_final, var_final = mean, var
    else:
        anom_pieces = [anom.sel(time=slice(w[0], w[1])) for w in composite_windows]
        total_pieces = [drain.precip.sel(time=slice(w[0], w[1])) for w in composite_windows]
        for (w0, w1), piece in zip(composite_windows, anom_pieces):
            ndays = piece.sizes["time"]
            if ndays not in (365, 366):
                raise ValueError(
                    f"composite_windows must each span a full annual cycle (365 or 366 "
                    f"days) so the day-of-year composite has complete coverage; window "
                    f"{w0}:{w1} spans {ndays} days. Use e.g. a Jul-1-to-Jun-30 window."
                )
        n = sum(p.sizes["time"] for p in anom_pieces)
        synth_times = pd.date_range(start=f"{dummy_leap_year - 2}-01-01", periods=n, freq="D")

        anom_cat = xr.concat(anom_pieces, dim="time").assign_coords(time=synth_times)
        total_cat = xr.concat(total_pieces, dim="time").assign_coords(time=synth_times)

        mean_warm = total_cat.groupby("time.dayofyear").mean(dim="time")
        var_warm = (anom_cat * anom_cat).groupby("time.dayofyear").mean(dim="time")

        it_shift = _derive_shift(composite_windows, dummy_leap_year)
        n_doy = mean_warm.sizes["dayofyear"]
        mean_final = mean_warm.copy()
        var_final = var_warm.copy()
        shift = it_shift
        for it in range(n_doy):
            mean_final[it] = mean_warm[shift]
            var_final[it] = var_warm[shift]
            shift = (shift + 1) % n_doy

    # Stamp the day-of-year mean/variance onto a dummy calendar so they can be
    # resampled to monthly, then upsampled back to a smooth daily annual
    # cycle via the triple-year cubic trick — matching the source notebook.
    n_doy = mean_final.sizes["dayofyear"]
    daily_dates = pd.date_range(start=f"{dummy_leap_year}-01-01", periods=n_doy, freq="D")
    dmean = xr.Dataset({"tmean": (["time", "lat", "lon"], mean_final.values)},
                        coords={"time": daily_dates, "lat": lats, "lon": lons})
    dvar = xr.Dataset({"variance": (["time", "lat", "lon"], var_final.values)},
                       coords={"time": daily_dates, "lat": lats, "lon": lons})
    mean_monthly = dmean.resample(time="ME").mean()
    var_monthly = dvar.resample(time="ME").mean()
    shape_monthly, scale_monthly = _method_of_moments(
        mean_monthly.tmean, var_monthly.variance, scale_qc_max)

    shape_daily = upsample_monthly_to_daily(shape_monthly, "shape", coords_3d=False,
                                             lats=lats, lons=lons)
    scale_daily = upsample_monthly_to_daily(scale_monthly, "scale", coords_3d=False,
                                             lats=lats, lons=lons)
    # Clip negatives (small cubic-overshoot near sharp transitions) and any
    # stray non-finite value (defensive — the monthly input is already
    # finite-clean, but interpolation numerics are worth double-checking).
    shape_daily = torch.nan_to_num(shape_daily, nan=0.0, posinf=0.0, neginf=0.0)
    scale_daily = torch.nan_to_num(scale_daily, nan=0.0, posinf=0.0, neginf=0.0)
    shape_daily = torch.where(shape_daily < 0.0, torch.zeros_like(shape_daily), shape_daily)
    scale_daily = torch.where(scale_daily < 0.0, torch.zeros_like(scale_daily), scale_daily)

    os.makedirs(output_dir, exist_ok=True)
    shape_path = os.path.join(output_dir, f"shape_{name}.pt")
    scale_path = os.path.join(output_dir, f"scale_{name}.pt")
    torch.save(shape_daily, shape_path)
    torch.save(scale_daily, scale_path)
    print(f"  Saved {shape_path}")
    print(f"  Saved {scale_path}")

    return shape_daily, scale_daily


def zero_shape_scale(output_dir, name, zw):
    """
    Explicit all-zero shape/scale pair (a shape of 0 disables the gamma
    heating draw entirely — the "no heating" case). Written explicitly
    rather than via the notebook's confusing np.where(x<0, 0.0, 0.0) idiom
    (which is always-zero regardless of the condition, and works only by
    accident).
    """
    mw, jmax, imax = build_grid_params({"zw": zw})
    z = torch.zeros((365, jmax, imax), dtype=torch.float64)
    os.makedirs(output_dir, exist_ok=True)
    shape_path = os.path.join(output_dir, f"shape_{name}.pt")
    scale_path = os.path.join(output_dir, f"scale_{name}.pt")
    torch.save(z, shape_path)
    torch.save(z.clone(), scale_path)
    print(f"  Saved {shape_path}")
    print(f"  Saved {scale_path}")
    return z, z


def _parse_window(s):
    start, end = s.split(":")
    return (start, end)


def main():
    parser = argparse.ArgumentParser(description="Fit gamma-distribution shape/scale files")
    parser.add_argument("--precip-glob", help="File glob/path for daily precip data")
    parser.add_argument("--precip-varname", default="cmorph", help="Precip variable name")
    parser.add_argument("--start-date", help="Baseline climatology period start (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Baseline climatology period end (YYYY-MM-DD)")
    parser.add_argument("--composite-window", action="append", default=[], type=_parse_window,
                         metavar="START:END",
                         help="Composite window as START:END (repeatable, e.g. an El Nino event)")
    parser.add_argument("--scale-qc-max", type=float, default=None,
                         help="Clip fitted scale values above this to 0 (despiking)")
    parser.add_argument("--zw", type=int, required=True, help="Spectral resolution (42/63/124)")
    parser.add_argument("--output-dir", required=True, help="Directory to write shape/scale .pt files")
    parser.add_argument("--name", required=True, help="Output filename label")
    parser.add_argument("--noheating", action="store_true",
                         help="Write an explicit all-zero shape/scale pair instead of fitting")
    args = parser.parse_args()

    if args.noheating:
        zero_shape_scale(args.output_dir, args.name, args.zw)
        return

    if not (args.precip_glob and args.start_date and args.end_date):
        parser.error("--precip-glob, --start-date, and --end-date are required unless --noheating")

    fit_gamma_shape_scale(
        precip_glob=args.precip_glob,
        precip_varname=args.precip_varname,
        date_range=(args.start_date, args.end_date),
        zw=args.zw,
        output_dir=args.output_dir,
        name=args.name,
        composite_windows=args.composite_window or None,
        scale_qc_max=args.scale_qc_max,
    )


if __name__ == "__main__":
    main()
