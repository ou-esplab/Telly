#!/usr/bin/env python
"""
Step 3: Post-processing

Interpolates model output from sigma levels to standard pressure levels,
and optionally computes sea-level pressure (SLP).

Usage:
    python scripts/03_postprocess.py --config config/experiments/my_exp.yaml

Output files are written to the experiment directory:
    {var}_Pressure_days_1-{N}.nc   for each var in postprocess_vars
    sealevelpressure_days_1-{N}.nc (if compute_slp=true)

For gamma_ac model_type, this script delegates to Gamma_AC_Model/PressureInterpMetPy.py.
"""

import argparse
import os
import subprocess
import sys

import metpy.interpolate
import numpy as np
import xarray as xr
import yaml

from _config import load_config, build_preprocess_path, build_experiment_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_grid_params(zw, kmax_override=None):
    match zw:
        case 42:  jmax, imax = 64,  128
        case 63:  jmax, imax = 96,  192
        case 124: jmax, imax = 188, 376
        case _:   raise ValueError(f"Unsupported zw={zw}.")
    kmax = kmax_override if kmax_override else {42: 11, 63: 26, 124: 26}[zw]
    return jmax, imax, kmax


# ---------------------------------------------------------------------------
# Pressure interpolation
# ---------------------------------------------------------------------------

def pressure_interp(datapath, zw, kmax, jmax, imax, dayst,
                    varnames, fieldnames, plev_hpa):
    """
    Interpolates model output for each variable in varnames from sigma to
    the specified pressure levels.

    varnames   : list of file name stubs  (e.g. ['vvel', 'uvel', 'geo'])
    fieldnames : list of in-file variable names (e.g. ['v', 'u', 'geo'])
    plev_hpa   : list of target pressure levels in hPa
    """
    stamp  = f"days_1-{dayst}"
    plev_r = np.array(plev_hpa, dtype=float) * 100.0   # hPa → Pa

    # Load surface pressure (always needed)
    fps = os.path.join(datapath, "lnps_????-??-??_????-??-??.nc")
    dps = xr.open_mfdataset(fps, decode_times=True, parallel=True)

    for DataSetname, Dataname in zip(varnames, fieldnames):
        outfile = os.path.join(datapath, f"{DataSetname}_Pressure_{stamp}.nc")
        if os.path.exists(outfile):
            print(f"  Skipping {DataSetname} — output already exists: {outfile}")
            continue

        fdata = os.path.join(datapath, f"{DataSetname}_????-??-??_????-??-??.nc")
        ddata = xr.open_mfdataset(fdata, decode_times=True, parallel=True)

        lats     = ddata["lat"].values
        lons     = ddata["lon"].values
        siglevs  = ddata["lev"].values

        dout     = np.zeros((dayst, len(plev_hpa), jmax, imax))
        pressure = np.zeros((kmax, jmax, imax))

        print(f"  Interpolating {DataSetname} ({dayst} days) → {plev_hpa} hPa ...")
        for k_day in range(dayst):
            vv   = ddata[Dataname][k_day, :, :, :].compute().values
            ps   = dps.lnps[k_day, :, :].compute().values
            surfp = np.exp(ps) * 1000.0 * 100.0   # Pa
            for kk in range(kmax):
                pressure[kk, :, :] = surfp * siglevs[kk]
            dout[k_day] = metpy.interpolate.log_interpolate_1d(
                plev_r, pressure, vv, axis=0
            )

        times  = ddata["time"].values
        dData  = xr.Dataset(
            {Dataname: (["time", "lev", "lat", "lon"], dout)},
            coords={"time": times, "lev": plev_hpa, "lat": lats, "lon": lons}
        )
        dData.to_netcdf(outfile)
        print(f"    Saved → {outfile}")


# ---------------------------------------------------------------------------
# Sea-level pressure
# ---------------------------------------------------------------------------

def compute_slp(datapath, preprocess_path, zw, kmax, jmax, imax, dayst, mw):
    """
    Derives sea-level pressure from model surface pressure, near-surface
    temperature, and topography using the ECMWF hydrostatic formula.
    """
    import torch
    import torch_harmonics.distributed as dist

    outfile = os.path.join(datapath, f"sealevelpressure_days_1-{dayst}.nc")
    if os.path.exists(outfile):
        print(f"  Skipping SLP — output already exists: {outfile}")
        return

    print("  Computing sea-level pressure...")

    # Load topography (spectral → grid)
    disht = dist.DistributedInverseRealSHT(
        jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False
    )
    topog_spec = torch.load(
        os.path.join(preprocess_path, "topog.spectral.pt"), weights_only=False
    )
    phi = disht(topog_spec).numpy()      # geopotential of terrain (m²/s²)

    # Load surface pressure and temperature
    fps   = os.path.join(datapath, "lnps_????-??-??_????-??-??.nc")
    ftemp = os.path.join(datapath, "temp_????-??-??_????-??-??.nc")
    dps   = xr.open_mfdataset(fps,   decode_times=True, parallel=True)
    dtemp = xr.open_mfdataset(ftemp, decode_times=True, parallel=True)

    lats = dps["lat"].values
    lons = dps["lon"].values

    # Physical constants
    laps  = -0.0065   # K/m lapse rate
    grav  = 9.8       # m/s²
    Rgas  = 287.04    # J/(kg·K)

    slp_out = np.zeros((dayst, jmax, imax))
    for k_day in range(dayst):
        lnps_d = dps.lnps[k_day, :, :].compute().values
        ps_d   = np.exp(lnps_d) * 101325.0               # Pa
        # near-surface temperature (level index kmax-2 is the lowest model layer)
        tsurf  = dtemp.temp[k_day, kmax - 2, :, :].compute().values

        # ECMWF formula (simplified)
        y      = laps * phi / (Rgas * tsurf * tsurf)
        Tstar  = tsurf * (1.0 + y / 2.0 + y ** 2 / 3.0)
        slp_d  = ps_d * np.exp(phi / (Rgas * Tstar) * (1.0 - y / 2.0 + y ** 2 / 3.0))
        slp_out[k_day] = slp_d / 100.0   # Pa → hPa

    times = dps["time"].values
    ds_slp = xr.Dataset(
        {"slp": (["time", "lat", "lon"], slp_out)},
        coords={"time": times, "lat": lats, "lon": lons}
    )
    ds_slp.to_netcdf(outfile)
    print(f"    Saved → {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATM workflow step 3: post-processing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    datapath       = build_experiment_path(cfg)
    preprocess_path = build_preprocess_path(cfg)

    if not os.path.isdir(datapath):
        raise FileNotFoundError(
            f"Experiment directory not found: {datapath}\n"
            "Run 02_run_model.py first."
        )

    zw   = cfg["zw"]
    kmax = cfg["kmax"]
    jmax, imax, _ = build_grid_params(zw, kmax)
    mw   = zw     # mw == zw for all supported resolutions

    dayst     = cfg["run_length_days"]
    plev_hpa  = cfg.get("pressure_levels_hpa", [850, 500, 300, 200])
    pvars     = cfg.get("postprocess_vars", ["uvel", "vvel", "geo"])

    # Map from file name stub to in-file variable name
    varname_map = {"uvel": "u", "vvel": "v", "geo": "geo", "temp": "temp"}
    fieldnames  = [varname_map.get(v, v) for v in pvars]

    model_type = cfg["model_type"]

    if model_type == "gamma_ac":
        # Delegate to the existing Gamma_AC script for each variable
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(project_root, "Gamma_AC_Model", "PressureInterpMetPy.py")
        expname = os.path.basename(datapath)
        print(f"Delegating to Gamma_AC_Model/PressureInterpMetPy.py (expname={expname})...")
        cmd = [sys.executable, script,
               "--expname", expname,
               "--dayst", str(dayst),
               "--datapath", datapath,
               "--zw", str(zw),
               "--kmax", str(kmax)]
        subprocess.run(cmd, check=True, cwd=os.path.join(project_root, "Gamma_AC_Model"))

    else:  # fixed_season
        print(f"Post-processing experiment: {datapath}")
        print(f"  Variables : {pvars}")
        print(f"  Levels    : {plev_hpa} hPa")

        pressure_interp(datapath, zw, kmax, jmax, imax, dayst,
                        pvars, fieldnames, plev_hpa)

    if cfg.get("compute_slp", False):
        compute_slp(datapath, preprocess_path, zw, kmax, jmax, imax, dayst, mw)

    print(f"\nPost-processing complete. Output in: {datapath}")


if __name__ == "__main__":
    main()
