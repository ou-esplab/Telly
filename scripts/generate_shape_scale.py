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

    # Trend-only fit -- adds the CMORPH precip trend, evaluated at target-year
    # relative to the record's own mean year, onto an existing control's
    # shape/scale (see fit_gamma_shape_scale_trend's docstring for why the
    # delta is anchored this way rather than to the record start):
    python scripts/generate_shape_scale.py --trend \\
        --precip-glob '...' --precip-varname cmorph \\
        --start-date 1998-01-01 --end-date 2024-08-31 \\
        --control-shape-path /path/to/preprocess/dir/shapeAC.pt \\
        --control-scale-path /path/to/preprocess/dir/scaleAC.pt \\
        --zw 63 --output-dir /path/to/preprocess/dir --name Trend

Also importable directly:
    from generate_shape_scale import (
        fit_gamma_shape_scale, fit_gamma_shape_scale_mjo, fit_gamma_shape_scale_trend,
        zero_shape_scale,
    )
"""

import argparse
import hashlib
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display available in this notebook/CLI context
import matplotlib.pyplot as plt
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


def _regrid_cache_path(cache_dir, precip_glob, precip_varname, zw):
    glob_hash = hashlib.sha1(precip_glob.encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"regridded_{precip_varname}_zw{zw}_{glob_hash}.nc")


def _load_or_build_regridded_precip(precip_glob, precip_varname, zw, lats, lons, dlatlon, cache_dir):
    # The expensive step (opening + regridding the whole precip_glob match)
    # doesn't depend on date_range at all -- fit_gamma_shape_scale only slices
    # to date_range afterward -- so caching it here speeds up every future
    # date range or composite-year choice, not just exact repeats.
    cache_path = _regrid_cache_path(cache_dir, precip_glob, precip_varname, zw) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        print(f"  Using cached regridded precip: {cache_path}")
        return xr.open_dataset(cache_path)["precip"]

    print("  No cache hit -- loading + regridding the full precip archive "
          "(this can take several minutes; cached afterward for any date range)...")
    precip_ds = xr.open_mfdataset(precip_glob, autoclose=True)
    precip_da = precip_ds[precip_varname]
    precip_vals = np.where(np.isnan(precip_da.values), 0.0, precip_da.values)
    precip_da = precip_da.copy(data=precip_vals)

    regridder = xe.Regridder(precip_da, dlatlon, "bilinear")
    precip_gg = regridder(precip_da)

    drain_full = xr.Dataset(
        {"precip": (["time", "lat", "lon"], precip_gg.values)},
        coords={"time": precip_da["time"], "lat": lats, "lon": lons},
    )

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        drain_full.to_netcdf(cache_path)
        print(f"  Cached regridded precip → {cache_path}")

    return drain_full["precip"]


def _save_diagnostic_plot(output_dir, name, lats, lons, shape_daily, scale_daily,
                           mean_final=None, mean_baseline=None):
    """
    Reference-notebook-style sanity-check figure -- plain pcolormesh/line plots
    (no cartopy; the reference notebook's actual shape/scale diagnostics never
    use it, only its unrelated intro rainfall map does), matching
    Gamma_AC_Model/reference_notebooks/preprocess.Gamma_heating.ipynb's cells:
      - 16/23: single-day spatial maps of the final shape/scale fields.
      - 14/20/34/37: annual-cycle line plot at one grid point.
      - 27: composite-vs-control zonal-mean difference (only when mean_final/
        mean_baseline are both given, i.e. a composite fit, not a plain control).
    """
    shape_np = shape_daily.numpy() if hasattr(shape_daily, "numpy") else np.asarray(shape_daily)
    scale_np = scale_daily.numpy() if hasattr(scale_daily, "numpy") else np.asarray(scale_daily)
    jmax, imax = shape_np.shape[1], shape_np.shape[2]
    # Closest actual latitude to the equator -- not just the middle grid
    # index, which only happens to land near-equatorial for some grids/
    # resolutions and isn't guaranteed to for others.
    j0 = int(np.argmin(np.abs(np.asarray(lats))))
    i0 = imax // 2
    day_scale = min(180, scale_np.shape[0] - 1)
    has_composite = mean_final is not None and mean_baseline is not None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    im0 = axes[0].pcolormesh(lons, lats, shape_np[0], cmap="Reds")
    axes[0].set_title("shape, day 0")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(lons, lats, scale_np[day_scale], cmap="Greens")
    axes[1].set_title(f"scale, day {day_scale}")
    fig.colorbar(im1, ax=axes[1])

    axes[2].plot(shape_np[:, j0, i0], label="shape")
    axes[2].plot(scale_np[:, j0, i0], label="scale")
    axes[2].set_title(f"annual cycle at lat={lats[j0]:.1f}, lon={lons[i0]:.1f}")
    axes[2].set_xlabel("day of year")
    axes[2].legend()

    if has_composite:
        # zonal mean (longitude-averaged) still leaves (dayofyear, lat) --
        # a Hovmoller-style image, not a single line, matching how xarray's
        # own .plot() would auto-render this same 2-D quantity (the
        # reference notebook's cell 27 relied on that auto-dispatch; a plain
        # matplotlib .plot() on the raw 2-D array instead draws one line per
        # latitude, which is unreadable).
        diff = (mean_final - mean_baseline).mean(dim="lon")
        diff_np = diff.values if hasattr(diff, "values") else np.asarray(diff)
        doy = np.arange(diff_np.shape[0])
        # diff_np is already (dayofyear, lat) -- pcolormesh(x, y, data) wants
        # data shaped (len(y), len(x)), so no transpose needed to put day of
        # year on the y-axis and lat on the x-axis.
        im3 = axes[3].pcolormesh(lats, doy, diff_np, cmap="BrBG", shading="auto")
        axes[3].set_title("composite minus control (zonal-mean precip)")
        axes[3].set_xlabel("lat")
        axes[3].set_ylabel("day of year")
        fig.colorbar(im3, ax=axes[3])
    else:
        axes[3].axis("off")

    fig.suptitle(f"Shape/scale diagnostics: {name}")
    fig.tight_layout()
    diagnostic_path = os.path.join(output_dir, f"diagnostic_{name}.png")
    fig.savefig(diagnostic_path, dpi=100)
    plt.close(fig)
    print(f"  Saved {diagnostic_path}")
    return diagnostic_path


def fit_gamma_shape_scale(precip_glob, precip_varname, date_range, zw, output_dir, name,
                           composite_windows=None, scale_qc_max=None, dummy_leap_year=1952,
                           precip_cache_dir=None, make_diagnostic_plot=True, exclude_dates=None):
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
    precip_cache_dir : directory to cache the regridded full precip_glob match
                        in (keyed by precip_glob/precip_varname/zw), so a later
                        call with a different date_range/composite_windows can
                        skip the slow open_mfdataset+regrid step entirely.
                        None (default) disables caching -- always reloads from
                        scratch, matching this function's original behavior.
    make_diagnostic_plot: if True (default), also save a reference-notebook-style
                        sanity-check figure to output_dir/diagnostic_{name}.png
                        (shape/scale maps, an annual-cycle line plot, and for a
                        composite fit, a composite-vs-control comparison).
    exclude_dates    : optional set/collection of dates to drop from date_range
                        before fitting, e.g. MJO-active days -- so the
                        resulting climatology represents "conditions when the
                        excluded days are absent" rather than all of
                        date_range indiscriminately. Applied once, right
                        after loading, before the day-of-year groupby --
                        agnostic to *why* dates are excluded (see this
                        module's _mjo_exclude_dates() for the MJO-specific
                        caller). None (default) excludes nothing.

    Returns (shape_tensor, scale_tensor), each shape (365, jmax, imax),
    dtype float64. Also written to output_dir as shape_{name}.pt / scale_{name}.pt.
    """
    mw, jmax, imax = build_grid_params({"zw": zw})
    lats, lons, dlatlon = _build_grid(jmax, imax)

    precip = _load_or_build_regridded_precip(
        precip_glob, precip_varname, zw, lats, lons, dlatlon, precip_cache_dir)
    drain = xr.Dataset({"precip": precip}).sel(time=slice(date_range[0], date_range[1]))
    if exclude_dates:
        keep = ~drain.time.to_index().isin(exclude_dates)
        drain = drain.isel(time=keep)

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

    if make_diagnostic_plot:
        _save_diagnostic_plot(
            output_dir, name, lats, lons, shape_daily, scale_daily,
            mean_final=mean_final if composite_windows else None,
            mean_baseline=mean if composite_windows else None,
        )

    return shape_daily, scale_daily


def _fit_monthly_trend_slope(precip_da):
    """
    Per-calendar-month linear trend (mm/day per year) via OLS regression of
    monthly-mean precip against year, vectorized across the grid with
    xarray's .polyfit -- one fit per calendar month (~26 yearly samples
    each), not per exact day-of-year (~26 samples/day, dominated by
    CMORPH's daily noise), matching fit_gamma_shape_scale's own
    monthly-resample smoothing philosophy.

    Returns (slope_monthly, year_mean):
      slope_monthly : (12, lat, lon) numpy array, Jan..Dec order.
      year_mean     : mean of the distinct years spanned by precip_da's
                       time coordinate -- computed once and shared across
                       all 12 months (rather than re-derived per month) so
                       a partial final year (e.g. 2024 only through Aug in
                       the default record) doesn't give different months
                       slightly different reference years.
    """
    monthly = precip_da.resample(time="ME").mean()
    year_mean = float(np.unique(monthly.time.dt.year.values).mean())

    slopes = []
    for m in range(1, 13):
        sub = monthly.sel(time=monthly.time.dt.month == m)
        sub = sub.assign_coords(year=("time", sub.time.dt.year.values)).swap_dims({"time": "year"})
        fit = sub.polyfit(dim="year", deg=1)
        slopes.append(fit.polyfit_coefficients.sel(degree=1).values)
    slope_monthly = np.stack(slopes, axis=0)
    return slope_monthly, year_mean


def _save_trend_diagnostic_plot(output_dir, name, lats, lons, shape_daily, scale_daily,
                                 shape_control, scale_control, delta_mean_daily,
                                 slope_monthly, clip_frac):
    """
    Trend-fit sanity-check figure (4 panels):
      1. Annual-mean trend slope map (mm/day per decade) -- is the spatial
         pattern coherent, not salt-and-pepper noise?
      2. Annual cycle at a near-equatorial reference point: control mean vs.
         trend-perturbed mean, overlaid -- is the delta small or large
         relative to the seasonal cycle it's riding on?
      3. Signal-to-noise map: annual-mean delta_mean / control's own std
         (std = scale*sqrt(shape), from the gamma variance formula) -- is
         the trend signal large relative to the control's own day-to-day
         variability, or noise-dominated?
      4. Printed pass/fail: shape left untouched (the whole point of "delta
         on the mean only, preserving relative variability"), and how often
         the mean-floor guard (a negative delta pushing the projected mean
         below 0, unphysical for precip) actually triggered.
    """
    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    shape_daily_np = _np(shape_daily)
    scale_daily_np = _np(scale_daily)
    shape_control_np = _np(shape_control)
    scale_control_np = _np(scale_control)
    delta_np = _np(delta_mean_daily)

    imax = shape_daily_np.shape[2]
    j0 = int(np.argmin(np.abs(np.asarray(lats))))
    i0 = imax // 2

    mean_control = shape_control_np * scale_control_np
    mean_new = shape_daily_np * scale_daily_np
    with np.errstate(divide="ignore", invalid="ignore"):
        control_std = scale_control_np * np.sqrt(shape_control_np)
        snr = delta_np.mean(axis=0) / control_std.mean(axis=0)
    snr = np.where(np.isfinite(snr), snr, 0.0)
    snr_frac = float(np.mean(np.abs(snr) > 0.1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    slope_annual = slope_monthly.mean(axis=0) * 10  # mm/day per decade
    vmax0 = np.nanpercentile(np.abs(slope_annual), 98) or 1.0
    im0 = axes[0].pcolormesh(lons, lats, slope_annual, cmap="BrBG",
                              vmin=-vmax0, vmax=vmax0, shading="auto")
    axes[0].set_title("annual-mean trend (mm/day per decade)")
    fig.colorbar(im0, ax=axes[0])

    axes[1].plot(mean_control[:, j0, i0], label="control mean")
    axes[1].plot(mean_new[:, j0, i0], label="control + trend")
    axes[1].set_title(f"mean precip at lat={lats[j0]:.1f}, lon={lons[i0]:.1f}")
    axes[1].set_xlabel("day of year")
    axes[1].legend()

    vmax2 = np.nanpercentile(np.abs(snr), 98) or 1.0
    im2 = axes[2].pcolormesh(lons, lats, snr, cmap="RdBu_r",
                              vmin=-vmax2, vmax=vmax2, shading="auto")
    axes[2].set_title(f"delta / control_std (SNR); |SNR|>0.1 over {snr_frac:.1%} of grid")
    fig.colorbar(im2, ax=axes[2])

    shape_identical = bool(np.array_equal(shape_daily_np, shape_control_np))
    axes[3].axis("off")
    axes[3].text(0.02, 0.75, f"shape unchanged from control: {shape_identical}",
                 transform=axes[3].transAxes)
    axes[3].text(0.02, 0.55, f"mean-floor guard triggered:\n{clip_frac:.2%} of day-gridpoint cells",
                 transform=axes[3].transAxes)

    fig.suptitle(f"Trend-only shape/scale diagnostics: {name}")
    fig.tight_layout()
    diagnostic_path = os.path.join(output_dir, f"diagnostic_{name}.png")
    fig.savefig(diagnostic_path, dpi=100)
    plt.close(fig)
    print(f"  Saved {diagnostic_path}")
    print(f"  shape unchanged from control: {shape_identical}")
    print(f"  mean-floor guard triggered: {clip_frac:.2%} of day-gridpoint cells")
    print(f"  |SNR|>0.1 over {snr_frac:.1%} of grid")
    return diagnostic_path


def fit_gamma_shape_scale_trend(precip_glob, precip_varname, date_range, zw, output_dir, name,
                                 control_shape_path, control_scale_path,
                                 target_year=None, scale_qc_max=None, dummy_leap_year=1952,
                                 precip_cache_dir=None, make_diagnostic_plot=True):
    """
    Fit a per-calendar-month linear trend from daily precipitation and add
    the trend-only component onto an existing control climatology's
    shape/scale tensors -- a "what if only the multi-decade drift were
    added on top of the control, nothing else" heating perturbation.

    control_shape_path/control_scale_path: an existing control's shape_*.pt/
                        scale_*.pt files (e.g. AC_Test's shapeAC.pt/
                        scaleAC.pt) -- RunModel.Gamma.py draws
                        precip ~ Gamma(shape, scale), so mean = shape*scale.
    target_year       : the year the trend-only delta is evaluated at.
                        Defaults to the last year of date_range.

                        IMPORTANT -- why this is not simply
                        "slope * record_length": an OLS regression line
                        always passes through (mean(year), mean(y)), so the
                        control's own mean already reflects the trend
                        evaluated at the record's *midpoint* year, not its
                        start. Anchoring the delta to
                        slope * (target_year - year_mean) -- where
                        year_mean is the regression sample's own mean year
                        -- means delta=0 exactly reproduces the control
                        (correct: no double-counting), and a target_year at
                        the record's end gives "how far the trend has
                        already pulled the climatology from its own
                        long-term average, as of now". This is NOT a
                        future-projection/extrapolation tool -- target_year
                        should stay within or near date_range; pushing it
                        far beyond the observed record is an extrapolation
                        claim this fit provides no basis for.
    scale_qc_max      : as in fit_gamma_shape_scale -- clip fitted scale
                        values above this to 0.
    precip_cache_dir  : as in fit_gamma_shape_scale.
    make_diagnostic_plot: if True (default), save diagnostic_{name}.png
                        (see _save_trend_diagnostic_plot).

    The control's shape tensor is left completely unchanged -- only scale is
    recomputed (scale_new = mean_new / shape_control) so the coefficient of
    variation (1/sqrt(shape), scale-invariant) is preserved exactly: this
    is "delta on the mean only", nothing else. mean_new is floored at 0
    before dividing (precip can't be negative; a strongly negative delta in
    a drying region could otherwise push it below zero).

    Returns (shape_daily, scale_daily, delta_mean_daily). Also written to
    output_dir as shape_{name}.pt / scale_{name}.pt.
    """
    mw, jmax, imax = build_grid_params({"zw": zw})
    lats, lons, dlatlon = _build_grid(jmax, imax)

    precip = _load_or_build_regridded_precip(
        precip_glob, precip_varname, zw, lats, lons, dlatlon, precip_cache_dir)
    drain = xr.Dataset({"precip": precip}).sel(time=slice(date_range[0], date_range[1]))

    slope_monthly, year_mean = _fit_monthly_trend_slope(drain.precip)
    if target_year is None:
        target_year = pd.Timestamp(date_range[1]).year
    delta_mean_monthly = slope_monthly * (target_year - year_mean)
    print(f"  Trend record mean year: {year_mean:.1f}; target year: {target_year} "
          f"(delta = slope * {target_year - year_mean:.1f} years)")

    delta_mean_daily = upsample_monthly_to_daily(
        delta_mean_monthly, "delta_mean", coords_3d=False, lats=lats, lons=lons)

    shape_control = torch.as_tensor(torch.load(control_shape_path))
    scale_control = torch.as_tensor(torch.load(control_scale_path))
    mean_control = shape_control * scale_control

    before_floor = mean_control + delta_mean_daily
    floor_triggered = before_floor < 0.0
    clip_frac = float(floor_triggered.to(torch.float64).mean().item())
    mean_new = torch.clamp(before_floor, min=0.0)

    shape_daily = shape_control
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_new = mean_new / shape_control
    scale_new = torch.nan_to_num(scale_new, nan=0.0, posinf=0.0, neginf=0.0)
    scale_new = torch.clamp(scale_new, min=0.0)
    if scale_qc_max is not None:
        scale_new = torch.where(scale_new > scale_qc_max, torch.zeros_like(scale_new), scale_new)

    os.makedirs(output_dir, exist_ok=True)
    shape_path = os.path.join(output_dir, f"shape_{name}.pt")
    scale_path = os.path.join(output_dir, f"scale_{name}.pt")
    torch.save(shape_daily, shape_path)
    torch.save(scale_new, scale_path)
    print(f"  Saved {shape_path}")
    print(f"  Saved {scale_path}")

    if make_diagnostic_plot:
        _save_trend_diagnostic_plot(
            output_dir, name, lats, lons, shape_daily, scale_new,
            shape_control, scale_control, delta_mean_daily, slope_monthly, clip_frac)

    return shape_daily, scale_new, delta_mean_daily


def _mjo_phase(pc1, pc2):
    """
    MJO phase (1-8) from OMI's two principal components.

    Sign convention (x=PC2, y=-PC1) matches NOAA PSL's stated OMI/RMM
    relationship ("the sign of OMI PC1 and the PC ordering should be
    reversed, so that OMI(PC2) is analogous to RMM(PC1) and -OMI(PC1) is
    analogous to RMM(PC2)" -- https://psl.noaa.gov/mjo/mjoindex/). The 157.5
    degree sector-boundary offset (rather than the naive 180) was found by
    grid search over sign/offset combinations, maximizing agreement against
    BOM's independently published RMM phase numbers (rmm.74toRealtime.txt)
    on days both indices call amplitude>1: 92.5% agreement within +/-1 phase
    across 5055 overlapping days, with a clean, symmetric, zero-centered
    error distribution -- not just an exact-match optimum, so not overfit to
    noise. See EXPERIMENTS.md's MJO section for the full validation.
    """
    angle_deg = np.degrees(np.arctan2(-pc1, pc2))
    return ((angle_deg + 157.5) // 45 % 8 + 1).astype(int)


def _read_omi_index(omi_index_path):
    """Parses NOAA PSL's omi.1x.txt (whitespace-delimited, no header: year
    month day PC1 PC2 amplitude) into a DataFrame with date/phase columns."""
    df = pd.read_csv(omi_index_path, sep=r"\s+",
                      names=["year", "month", "day", "PC1", "PC2", "amplitude"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values("date").reset_index(drop=True)
    df["phase"] = _mjo_phase(df["PC1"].values, df["PC2"].values)
    return df


def _identify_mjo_onsets(omi_index_path, target_phases, amplitude_threshold=1.0,
                          season_months=None):
    """
    Returns the onset date (first day) of every contiguous run where
    phase is in target_phases and amplitude > amplitude_threshold.

    target_phases : a single phase (int) or a list/tuple of phases (e.g. an
                     adjacent pair like [8, 1]) -- a day sequence that stays
                     active while transitioning between paired phases counts
                     as one continuous episode, not two, since grouping only
                     looks at the active boolean, not which specific phase.
    season_months : optional iterable of month numbers (e.g. {11,12,1,2,3,4}
                     for boreal winter); if given, only days in these months
                     are ever eligible to be "active" -- restricting which
                     season onsets/composites are drawn from. None (default)
                     = no restriction, any month.
    """
    if isinstance(target_phases, int):
        target_phases = [target_phases]
    df = _read_omi_index(omi_index_path)

    active = df["phase"].isin(target_phases) & (df["amplitude"] > amplitude_threshold)
    if season_months is not None:
        active &= df["date"].dt.month.isin(season_months)
    group_id = (active != active.shift(fill_value=False)).cumsum()
    onsets = df.loc[active].groupby(group_id[active])["date"].first().tolist()
    return onsets


def _mjo_exclude_dates(omi_index_path, amplitude_threshold=1.0, season_months=None):
    """
    Returns the set of dates to drop to build an "MJO-inactive" climatology:
    every date where OMI amplitude > amplitude_threshold (active, any phase),
    plus -- if season_months is given -- every date outside those months too.
    Meant to be passed directly as fit_gamma_shape_scale's exclude_dates, so
    the remaining (kept) dates are exactly "in-season and MJO-inactive" in
    one step -- matching the same season restriction used for the
    phase-pair composites this baseline is compared against (comparing a
    season-restricted composite to an all-year-inactive baseline would
    reintroduce the seasonal-averaging-window mismatch documented in
    EXPERIMENTS.md).
    """
    df = _read_omi_index(omi_index_path)
    exclude = df["amplitude"] > amplitude_threshold
    if season_months is not None:
        exclude = exclude | ~df["date"].dt.month.isin(season_months)
    return set(df.loc[exclude, "date"])


def fit_gamma_shape_scale_mjo(precip_glob, precip_varname, date_range, zw, output_dir, name,
                               omi_index_path, target_phases,
                               lag_days_before=5, lag_days_after=15,
                               amplitude_threshold=1.0, scale_qc_max=None,
                               season_months=None,
                               precip_cache_dir=None, make_diagnostic_plot=True):
    """
    Fit gamma-distribution shape/scale parameters to a lag-day composite of
    real MJO events in one phase (or an adjacent pair), then tile the
    resulting short cycle to fill the full 365-day array RunModel.Gamma.py
    expects.

    RunModel.Gamma.py indexes shape[daynumber]/scale[daynumber] purely by the
    real calendar day-of-year (0-364), with no other periodicity logic (see
    EXPERIMENTS.md's MJO section) -- so a short pattern tiled to 365 days
    cycles through the model automatically, with no model-code changes.

    Unlike fit_gamma_shape_scale()'s composite_windows path (built for a
    slowly-varying ANNUAL cycle, hence its monthly-resample-then-cubic-upsample
    smoothing step), this works at daily lag resolution throughout -- monthly
    resampling would leave only ~1 bin per MJO cycle and destroy the
    intraseasonal structure being composited.

    omi_index_path : path to NOAA PSL's omi.1x.txt.
    target_phases   : MJO phase 1-8 (int) or an adjacent pair (e.g. [8, 1],
                      [2, 3]) to composite onsets for (see _mjo_phase).
    lag_days_before/lag_days_after: composite window is
                      [onset - lag_days_before, onset + lag_days_after), i.e.
                      lag_days_before + lag_days_after days total. Tunable --
                      inspect the diagnostic plot's tile-wrap seam before
                      treating a choice as final.
    amplitude_threshold: minimum OMI amplitude to count as an "active" MJO day.
    scale_qc_max     : as in fit_gamma_shape_scale -- likely at least as
                      relevant here, since the event count compositing a
                      single phase is probably smaller than the ENSO
                      composite's whole-year samples.
    season_months    : optional iterable of month numbers (e.g.
                      {11,12,1,2,3,4} for boreal winter) restricting which
                      onsets are eligible -- MJO amplitude/character varies
                      seasonally, so mixing all months into one composite
                      blends different regimes together. None = no
                      restriction (any month). A short window can still
                      bleed a few days past the season boundary at an
                      onset near the edge -- not additionally clipped.

    Returns (shape_tensor, scale_tensor, usable_onsets). Also written to
    output_dir as shape_{name}.pt / scale_{name}.pt.
    """
    window_len = lag_days_before + lag_days_after
    if not (0 < window_len <= 365):
        raise ValueError(f"lag_days_before+lag_days_after must be in (0, 365], got {window_len}")

    mw, jmax, imax = build_grid_params({"zw": zw})
    lats, lons, dlatlon = _build_grid(jmax, imax)

    precip = _load_or_build_regridded_precip(
        precip_glob, precip_varname, zw, lats, lons, dlatlon, precip_cache_dir)
    drain = xr.Dataset({"precip": precip}).sel(time=slice(date_range[0], date_range[1]))

    onsets = _identify_mjo_onsets(omi_index_path, target_phases, amplitude_threshold, season_months)
    tmin = pd.Timestamp(drain.time.min().item())
    tmax = pd.Timestamp(drain.time.max().item())
    usable_onsets = [
        d for d in onsets
        if (d - pd.Timedelta(days=lag_days_before)) >= tmin
        and (d + pd.Timedelta(days=lag_days_after - 1)) <= tmax
    ]
    if not usable_onsets:
        raise ValueError(
            f"No phase-{target_phases} onsets found with a full lag window inside "
            f"{date_range} -- widen date_range or shrink the lag window."
        )
    print(f"  {len(onsets)} phase-{target_phases} onsets found in the OMI record; "
          f"{len(usable_onsets)} have a full lag window inside {date_range}.")

    pieces = []
    for onset in usable_onsets:
        w0 = onset - pd.Timedelta(days=lag_days_before)
        w1 = onset + pd.Timedelta(days=lag_days_after - 1)
        piece = drain.precip.sel(time=slice(w0, w1))
        if piece.sizes["time"] != window_len:
            continue  # gap in the precip record for this event -- skip rather than misalign
        pieces.append(piece.assign_coords(time=np.arange(window_len)))
    if not pieces:
        raise ValueError("No usable onsets had a complete, gap-free precip window -- "
                          "check precip_glob coverage.")
    print(f"  {len(pieces)} events contributed to the composite "
          f"(after dropping any with data gaps).")

    stacked = xr.concat(pieces, dim="event")
    mean_lag = stacked.mean(dim="event").values   # (lag, lat, lon)
    var_lag = stacked.var(dim="event").values      # (lag, lat, lon)

    shape_lag, scale_lag = _method_of_moments(mean_lag, var_lag, scale_qc_max)

    n_tiles = int(np.ceil(365 / window_len))
    shape_daily = np.tile(shape_lag, (n_tiles, 1, 1))[:365]
    scale_daily = np.tile(scale_lag, (n_tiles, 1, 1))[:365]
    shape_daily = torch.nan_to_num(torch.as_tensor(shape_daily), nan=0.0, posinf=0.0, neginf=0.0)
    scale_daily = torch.nan_to_num(torch.as_tensor(scale_daily), nan=0.0, posinf=0.0, neginf=0.0)
    shape_daily = torch.where(shape_daily < 0.0, torch.zeros_like(shape_daily), shape_daily)
    scale_daily = torch.where(scale_daily < 0.0, torch.zeros_like(scale_daily), scale_daily)

    os.makedirs(output_dir, exist_ok=True)
    shape_path = os.path.join(output_dir, f"shape_{name}.pt")
    scale_path = os.path.join(output_dir, f"scale_{name}.pt")
    torch.save(shape_daily, shape_path)
    torch.save(scale_daily, scale_path)
    print(f"  Saved {shape_path}")
    print(f"  Saved {scale_path}")

    if make_diagnostic_plot:
        _save_diagnostic_plot(output_dir, name, lats, lons, shape_daily, scale_daily)

    return shape_daily, scale_daily, usable_onsets


def zero_shape_scale(output_dir, name, zw, make_diagnostic_plot=True):
    """
    Explicit all-zero shape/scale pair (a shape of 0 disables the gamma
    heating draw entirely — the "no heating" case). Written explicitly
    rather than via the notebook's confusing np.where(x<0, 0.0, 0.0) idiom
    (which is always-zero regardless of the condition, and works only by
    accident).
    """
    mw, jmax, imax = build_grid_params({"zw": zw})
    lats, lons, _ = _build_grid(jmax, imax)
    z = torch.zeros((365, jmax, imax), dtype=torch.float64)
    os.makedirs(output_dir, exist_ok=True)
    shape_path = os.path.join(output_dir, f"shape_{name}.pt")
    scale_path = os.path.join(output_dir, f"scale_{name}.pt")
    torch.save(z, shape_path)
    torch.save(z.clone(), scale_path)
    print(f"  Saved {shape_path}")
    print(f"  Saved {scale_path}")
    if make_diagnostic_plot:
        _save_diagnostic_plot(output_dir, name, lats, lons, z, z)
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
    parser.add_argument("--trend", action="store_true",
                         help="Fit a trend-only delta onto an existing control's shape/scale "
                              "(requires --control-shape-path/--control-scale-path)")
    parser.add_argument("--control-shape-path", default=None,
                         help="Existing control shape_*.pt to add the trend delta onto (--trend)")
    parser.add_argument("--control-scale-path", default=None,
                         help="Existing control scale_*.pt to add the trend delta onto (--trend)")
    parser.add_argument("--target-year", type=float, default=None,
                         help="Year the trend delta is evaluated at (--trend). Defaults to the "
                              "last year of --end-date. See fit_gamma_shape_scale_trend's "
                              "docstring for why this isn't simply the record length.")
    parser.add_argument("--precip-cache-dir", default=None,
                         help="Cache the regridded precip_glob match here, keyed by "
                              "precip-glob/precip-varname/zw, so later runs with a different "
                              "date range or composite skip the slow reload+regrid")
    parser.add_argument("--no-diagnostic-plot", action="store_true",
                         help="Skip saving the reference-notebook-style sanity-check PNG "
                              "(diagnostic_{name}.png) alongside the shape/scale .pt files")
    args = parser.parse_args()

    if args.noheating:
        zero_shape_scale(args.output_dir, args.name, args.zw,
                          make_diagnostic_plot=not args.no_diagnostic_plot)
        return

    if not (args.precip_glob and args.start_date and args.end_date):
        parser.error("--precip-glob, --start-date, and --end-date are required unless --noheating")

    if args.trend:
        if not (args.control_shape_path and args.control_scale_path):
            parser.error("--trend requires --control-shape-path and --control-scale-path")
        fit_gamma_shape_scale_trend(
            precip_glob=args.precip_glob,
            precip_varname=args.precip_varname,
            date_range=(args.start_date, args.end_date),
            zw=args.zw,
            output_dir=args.output_dir,
            name=args.name,
            control_shape_path=args.control_shape_path,
            control_scale_path=args.control_scale_path,
            target_year=args.target_year,
            scale_qc_max=args.scale_qc_max,
            precip_cache_dir=args.precip_cache_dir,
            make_diagnostic_plot=not args.no_diagnostic_plot,
        )
        return

    fit_gamma_shape_scale(
        precip_glob=args.precip_glob,
        precip_varname=args.precip_varname,
        date_range=(args.start_date, args.end_date),
        zw=args.zw,
        output_dir=args.output_dir,
        name=args.name,
        composite_windows=args.composite_window or None,
        scale_qc_max=args.scale_qc_max,
        precip_cache_dir=args.precip_cache_dir,
        make_diagnostic_plot=not args.no_diagnostic_plot,
    )


if __name__ == "__main__":
    main()
