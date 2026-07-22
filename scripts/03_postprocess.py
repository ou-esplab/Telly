#!/usr/bin/env python
"""
Step 3: Post-processing

Interpolates model output from sigma levels to standard pressure levels,
and optionally computes sea-level pressure (SLP).

Usage:
    python scripts/03_postprocess.py --config config/experiments/my_exp.yaml

Postprocessing is incremental: it looks at what raw {var}_{start}_{end}.nc
chunk files already exist on disk and skips any chunk that already has a
matching postprocessed output, so extending a run and re-running this step
only processes the new chunks. Output files, one per raw chunk:
    {var}_Pressure_{start}_{end}.nc      for each var in postprocess_vars
    sealevelpressure_{start}_{end}.nc    (if compute_slp=true)

For gamma_ac model_type, this script delegates to Gamma_AC_Model/PressureInterpMetPy.py.
"""

import argparse
import glob
import os
import re
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


_CHUNK_RE = re.compile(r"^.+_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$")


def find_chunks(datapath, stub):
    """
    Returns a (start, end, filepath) tuple for every raw {stub}_{start}_{end}.nc
    chunk file found in datapath, sorted chronologically by start date.
    """
    pattern = os.path.join(datapath, f"{stub}_????-??-??_????-??-??.nc")
    chunks = []
    for fpath in glob.glob(pattern):
        m = _CHUNK_RE.match(os.path.basename(fpath))
        if m:
            chunks.append((m.group(1), m.group(2), fpath))
    return sorted(chunks)


# ---------------------------------------------------------------------------
# Pressure interpolation
# ---------------------------------------------------------------------------

def pressure_interp(datapath, zw, kmax, jmax, imax,
                    varnames, fieldnames, plev_hpa):
    """
    Interpolates model output for each variable in varnames from sigma to
    the specified pressure levels, one raw output chunk at a time -- skips
    any chunk that already has a matching {var}_Pressure_{start}_{end}.nc.

    varnames   : list of file name stubs  (e.g. ['vvel', 'uvel', 'geo'])
    fieldnames : list of in-file variable names (e.g. ['v', 'u', 'geo'])
    plev_hpa   : list of target pressure levels in hPa
    """
    plev_r = np.array(plev_hpa, dtype=float) * 100.0   # hPa → Pa

    for DataSetname, Dataname in zip(varnames, fieldnames):
        chunks = find_chunks(datapath, DataSetname)
        if not chunks:
            print(f"  WARNING: no raw {DataSetname}_*.nc chunks found, skipping.")
            continue

        for start, end, fdata in chunks:
            outfile = os.path.join(datapath, f"{DataSetname}_Pressure_{start}_{end}.nc")
            if os.path.exists(outfile):
                print(f"  Skipping {DataSetname} {start}_{end} — already postprocessed.")
                continue

            fps   = os.path.join(datapath, f"lnps_{start}_{end}.nc")
            ddata = xr.open_dataset(fdata, decode_times=True)
            dps   = xr.open_dataset(fps, decode_times=True)

            lats     = ddata["lat"].values
            lons     = ddata["lon"].values
            siglevs  = ddata["lev"].values
            n_days   = ddata.sizes["time"]

            dout     = np.zeros((n_days, len(plev_hpa), jmax, imax))
            pressure = np.zeros((kmax, jmax, imax))

            print(f"  Interpolating {DataSetname} {start}_{end} ({n_days} days) → {plev_hpa} hPa ...")
            for k_day in range(n_days):
                vv    = ddata[Dataname][k_day, :, :, :].values
                ps    = dps.lnps[k_day, :, :].values
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
            dData.attrs["source_chunk"] = os.path.basename(fdata)
            dData.to_netcdf(outfile)
            print(f"    Saved → {outfile}")
            ddata.close()
            dps.close()


# ---------------------------------------------------------------------------
# Sea-level pressure
# ---------------------------------------------------------------------------

def compute_slp(datapath, preprocess_path, zw, kmax, jmax, imax, mw, model_type):
    """
    Derives sea-level pressure from model surface pressure, near-surface
    temperature, and topography using the ECMWF hydrostatic formula, one raw
    lnps/temp chunk at a time -- skips any chunk that already has a matching
    sealevelpressure_{start}_{end}.nc.
    """
    import torch
    import torch_harmonics.distributed as dist

    # Gamma_AC_Model's raw temp field is named "t" (matching uvel/vvel's
    # single-letter convention), not "temp" like fixed_season's -- same
    # distinction as scripts/04_plot_results.py's VAR_META override.
    temp_field = "t" if model_type == "gamma_ac" else "temp"

    chunks = find_chunks(datapath, "lnps")
    if not chunks:
        print("  WARNING: no raw lnps_*.nc chunks found, skipping SLP.")
        return

    # Load topography (spectral → grid) once, reused for every chunk.
    disht = dist.DistributedInverseRealSHT(
        jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False
    )
    topog_spec = torch.load(
        os.path.join(preprocess_path, "topog.spectral.pt"), weights_only=False
    )
    phi = disht(topog_spec).numpy()      # geopotential of terrain (m²/s²)

    # Physical constants
    laps  = -0.0065   # K/m lapse rate
    grav  = 9.8       # m/s²
    Rgas  = 287.04    # J/(kg·K)

    for start, end, fps in chunks:
        outfile = os.path.join(datapath, f"sealevelpressure_{start}_{end}.nc")
        if os.path.exists(outfile):
            print(f"  Skipping SLP {start}_{end} — already computed.")
            continue

        ftemp = os.path.join(datapath, f"temp_{start}_{end}.nc")
        dps   = xr.open_dataset(fps, decode_times=True)
        dtemp = xr.open_dataset(ftemp, decode_times=True)

        lats   = dps["lat"].values
        lons   = dps["lon"].values
        n_days = dps.sizes["time"]

        print(f"  Computing SLP {start}_{end} ({n_days} days)...")
        slp_out = np.zeros((n_days, jmax, imax))
        for k_day in range(n_days):
            lnps_d = dps.lnps[k_day, :, :].values
            ps_d   = np.exp(lnps_d) * 101325.0               # Pa
            # near-surface temperature (level index kmax-2 is the lowest model layer)
            tsurf  = dtemp[temp_field][k_day, kmax - 2, :, :].values

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
        ds_slp.attrs["source_chunk"] = os.path.basename(fps)
        ds_slp.to_netcdf(outfile)
        print(f"    Saved → {outfile}")
        dps.close()
        dtemp.close()


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

    # Kept only for Gamma_AC_Model/PressureInterpMetPy.py's --dayst CLI arg
    # (informational/back-compat -- see that script's docstring). Neither
    # pressure_interp() nor compute_slp() below use it: both are driven
    # entirely by what raw chunk files already exist on disk.
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

        pressure_interp(datapath, zw, kmax, jmax, imax,
                        pvars, fieldnames, plev_hpa)

    if cfg.get("compute_slp", False):
        compute_slp(datapath, preprocess_path, zw, kmax, jmax, imax, mw, model_type)

    print(f"\nPost-processing complete. Output in: {datapath}")


if __name__ == "__main__":
    main()
