#!/usr/bin/env python
"""
Stationary-wave comparison for the AC_SW_* teaching experiments.

For each experiment, reads the step-3 geo_Pressure_*.nc output, discards the
spin-up, forms the seasonal-mean geopotential at one pressure level and removes
its zonal mean -- the stationary eddy z* -- then plots z* for every run and each
run's response relative to the baseline (run minus baseline).

Usage:
    python scripts/stationary_waves.py \\
        --experiment-root /data/esplab/kpegion/projects/AGCM_Experiments \\
        --runs AC_SW_base AC_SW_mtn AC_SW_landsea AC_SW_heat AC_SW_all \\
        --obs AC_Test --season DJF --level 200 --out-dir figures/stationary_waves

--obs is optional: an observed-basic-state run shown only for comparison with
the idealized all-on run. The first of --runs is the baseline.

Also printed per run: the RMS of z* (m) and a sampling-noise estimate from the
first and second half of the season's days (rainfall varies from day to day, so
short records are noisy; a response well above the noise estimate is a real signal).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
           "ANN": tuple(range(1, 13))}
GRAV = 9.8   # geo output is geopotential (m^2/s^2)


def seasonal_eddy_means(exp_dir, level, months, skip_days):
    """Return (z*_full, z*_half1, z*_half2, lats, lons): seasonal-mean geopotential
    height minus its zonal mean, for the whole post-spin-up record and each half."""
    files = sorted(glob.glob(os.path.join(exp_dir, "geo_Pressure_*.nc")))
    if not files:
        raise FileNotFoundError(f"No geo_Pressure_*.nc in {exp_dir} (run step 3 first).")

    # Pass 1: which days are used (after spin-up, in the season). The two "halves" are the
    # first and second half of those season days, so they are balanced even for one season.
    keeps, day = [], 0
    for f in files:
        with xr.open_dataset(f) as d:
            mon = d.time.dt.month.values
            n = d.sizes["time"]
        idx = np.arange(day, day + n)
        day += n
        keeps.append((idx >= skip_days) & np.isin(mon, months))
    nkeep = int(sum(k.sum() for k in keeps))
    if nkeep < 2:
        raise ValueError(f"{exp_dir}: only {nkeep} {months} days after skipping {skip_days} "
                         f"of {day} days of output")

    # Pass 2: accumulate the first and second half of the kept days. Points below the
    # ground (the level is under the surface, e.g. 850 hPa over high terrain) are NaN in
    # the pressure-level files; they are skipped, so a point is averaged over the days it
    # is above ground and stays NaN only if it never is.
    sums = [0.0, 0.0]
    cnts = [0, 0]
    rank = 0
    lats = lons = None
    for f, keep in zip(files, keeps):
        if not keep.any():
            continue
        with xr.open_dataset(f) as d:
            g = d.geo.sel(lev=level, method="nearest") / GRAV
            lats, lons = d.lat.values, d.lon.values
            vals = g.values
        sel_all = vals[keep]
        half = ((rank + np.arange(len(sel_all))) >= nkeep / 2).astype(int)
        rank += len(sel_all)
        for h in (0, 1):
            sel = sel_all[half == h]
            if len(sel):
                sums[h] = sums[h] + np.nansum(sel, axis=0)
                cnts[h] = cnts[h] + (~np.isnan(sel)).sum(axis=0)

    def _mean(s, c):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    means = [_mean(sums[h], cnts[h]) for h in (0, 1)]
    full = _mean(sums[0] + sums[1], cnts[0] + cnts[1])
    eddy = lambda m: m - np.nanmean(m, axis=-1, keepdims=True)
    return eddy(full), eddy(means[0]), eddy(means[1]), lats, lons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="Experiment names; the first is the baseline")
    ap.add_argument("--obs", default=None, help="Optional observed-basic-state run shown for comparison")
    ap.add_argument("--season", default="DJF", choices=sorted(SEASONS))
    ap.add_argument("--level", type=float, default=200.0, help="Pressure level in hPa (850/500/300/200)")
    ap.add_argument("--skip-days", type=int, default=365, help="Spin-up days to discard (default 365)")
    ap.add_argument("--out-dir", default="figures/stationary_waves")
    ap.add_argument("--vmax", type=float, default=None, help="Contour limit (m); default from the data")
    args = ap.parse_args()

    names = list(args.runs) + ([args.obs] if args.obs else [])
    data = {}
    for n in names:
        zf, z1, z2, lats, lons = seasonal_eddy_means(
            os.path.join(args.experiment_root, n), args.level, SEASONS[args.season], args.skip_days)
        data[n] = zf
        noise = np.sqrt(np.mean((z1 - z2) ** 2)) / 2.0   # sampling noise of the full-record mean
        print(f"{n:16s} RMS z* = {np.sqrt(np.mean(zf**2)):7.2f} m   sampling-noise estimate = {noise:6.2f} m")

    base = args.runs[0]
    resp_runs = args.runs[1:]
    ncol = 2
    nrow = len(resp_runs) + (1 if args.obs else 0)   # rows after the baseline row
    vmax = args.vmax or float(np.ceil(max(np.abs(data[n]).max() for n in names) / 20.0) * 20.0) or 20.0
    levels = np.linspace(-vmax, vmax, 21)
    fig, axes = plt.subplots(nrow + 1, ncol, figsize=(11, 3.4 * (nrow + 1)), squeeze=False,
                             constrained_layout=True)

    def panel(ax, field, title):
        cs = ax.contourf(lons, lats, field, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(0, 361, 60))
        ax.set_yticks(range(-60, 61, 30))
        return cs

    cs = panel(axes[0][0], data[base], f"{base}: z* (baseline)")
    axes[0][1].axis("off")
    for i, n in enumerate(resp_runs, start=1):
        panel(axes[i][0], data[n], f"{n}: z*")
        panel(axes[i][1], data[n] - data[base], f"{n} minus {base}: response")
    if args.obs:
        panel(axes[nrow][0], data[args.obs], f"{args.obs}: z* (observed basic state, comparison)")
        axes[nrow][1].axis("off")
    fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.6, label=f"z* ({args.level:.0f} hPa height, m)")
    fig.suptitle(f"{args.season} stationary eddies at {args.level:.0f} hPa", y=0.995)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"stationary_waves_{args.season}_{args.level:.0f}hPa.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
