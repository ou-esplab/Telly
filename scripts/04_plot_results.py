#!/usr/bin/env python
"""
Step 4: Standard Figures

Generates a standard set of diagnostic figures comparing the experiment
to an optional control run.

Usage:
    python scripts/04_plot_results.py --config config/experiments/my_exp.yaml

Figures are saved to:
    {experiment_dir}/figures/{var}{level}_mean.png      — time-mean field
    {experiment_dir}/figures/{var}{level}_diff.png      — difference vs control
                                                           (only if control_experiment set)

Config keys used here:
    control_experiment  : name of the control experiment directory
    spinup_days         : number of days to skip at the start (default 60)
    plot_levels_hpa     : pressure levels to plot (default [200, 850])
    plot_vars           : variables to plot (default [vvel, uvel, geo])
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import xarray as xr
import yaml

from _config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_experiment_path(cfg):
    return os.path.join(cfg["experiment_root"], cfg["experiment_name"])


def build_control_path(cfg):
    ctrl = cfg.get("control_experiment")
    if not ctrl:
        return None
    return os.path.join(cfg["experiment_root"], ctrl)


# Map file stub → in-file variable name and long name for plot title/colorbar
VAR_META = {
    "vvel": ("v",   "Meridional Wind",   "m s⁻¹",  (-10, 10, 1),  "RdGy_r"),
    "uvel": ("u",   "Zonal Wind",        "m s⁻¹",  (-15, 15, 1),  "RdGy_r"),
    "geo":  ("geo", "Geopotential",      "m² s⁻²", (-300, 300, 20), "RdBu_r"),
    "slp":  ("slp", "Sea Level Pressure","hPa",     (-5, 5, 0.5),  "RdBu_r"),
    "temp": ("temp","Temperature",       "K",       (-5, 5, 0.5),  "RdBu_r"),
}


def load_timemean(expdir, varfile, fieldname, level_hpa, spinup_days):
    """Load pressure-level data and return the time-mean field after spinup."""
    fpattern = os.path.join(expdir, f"{varfile}_Pressure_days_*.nc")
    ds = xr.open_mfdataset(fpattern, decode_times=True)
    da = ds[fieldname].sel(lev=float(level_hpa)).isel(time=slice(spinup_days, None))
    return da.mean(dim="time").load()


def make_map_plot(data, title, label, clevs, cmap, outfile):
    """Single-panel map plot using ProPlot with coastlines."""
    nlevs = clevs
    f, ax = pplt.subplots(
        nrows=1, ncols=1, proj="pcarree",
        proj_kw={"central_longitude": 180},
        figsize=(11, 5.5),
    )
    ax.format(
        coast=True, borders=True,
        coastcolor="black", borderscolor="black",
        latlines=20, lonlines=30,
        latlabels=True, lonlabels=True,
        fontsize=12,
    )
    cs = ax.contourf(
        data["lon"].values, data["lat"].values, data.values,
        norm="div", levels=nlevs, cmap=cmap, extend="both",
    )
    ax.format(title=title, titleweight="bold")
    f.colorbar(cs, loc="b", length=0.6, label=label)
    f.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(f)
    print(f"  Saved: {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATM workflow step 4: standard figures")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    expdir   = build_experiment_path(cfg)
    ctrldir  = build_control_path(cfg)
    figdir   = os.path.join(expdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    if not os.path.isdir(expdir):
        raise FileNotFoundError(
            f"Experiment directory not found: {expdir}\n"
            "Run 02_run_model.py and 03_postprocess.py first."
        )

    spinup    = cfg.get("spinup_days", 60)
    plot_levs = cfg.get("plot_levels_hpa", [200, 850])
    plot_vars = cfg.get("plot_vars", ["vvel", "uvel", "geo"])
    expname   = os.path.basename(expdir)

    print(f"Generating figures for: {expname}")
    print(f"  Variables : {plot_vars}")
    print(f"  Levels    : {plot_levs} hPa")
    print(f"  Spin-up   : {spinup} days skipped")
    print(f"  Output    : {figdir}")

    for varfile in plot_vars:
        if varfile not in VAR_META:
            print(f"  WARNING: unknown plot_var '{varfile}', skipping.")
            continue
        fieldname, longname, units, clev_spec, cmap = VAR_META[varfile]
        cmin, cmax, cint = clev_spec

        for lev in plot_levs:
            tag = f"{varfile}{lev}"

            # Absolute time-mean map
            try:
                da_exp = load_timemean(expdir, varfile, fieldname, lev, spinup)
            except Exception as exc:
                print(f"  WARNING: could not load {varfile} at {lev} hPa — {exc}")
                continue

            clevs_abs = np.arange(da_exp.values.min(), da_exp.values.max(),
                                  (da_exp.values.max() - da_exp.values.min()) / 20)
            make_map_plot(
                da_exp,
                title=f"{expname} — {longname} {lev} hPa (mean, skip {spinup} days)",
                label=units,
                clevs=clevs_abs,
                cmap="viridis",
                outfile=os.path.join(figdir, f"{tag}_mean.png"),
            )

            # Difference map (experiment − control)
            if ctrldir:
                if not os.path.isdir(ctrldir):
                    print(f"  WARNING: control directory not found: {ctrldir}")
                    continue
                try:
                    da_ctrl = load_timemean(ctrldir, varfile, fieldname, lev, spinup)
                except Exception as exc:
                    print(f"  WARNING: could not load control {varfile} at {lev} hPa — {exc}")
                    continue

                diff = da_exp - da_ctrl
                clevs_diff = np.arange(cmin, cmax + cint, cint)
                ctrl_name = os.path.basename(ctrldir)
                make_map_plot(
                    diff,
                    title=f"{expname} − {ctrl_name} — {longname} {lev} hPa",
                    label=units,
                    clevs=clevs_diff,
                    cmap=cmap,
                    outfile=os.path.join(figdir, f"{tag}_diff.png"),
                )

    print(f"\nFigure generation complete. Figures in: {figdir}")


if __name__ == "__main__":
    main()
