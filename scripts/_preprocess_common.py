"""
Shared preprocessing helpers used by scripts/01_preprocess.py and by the
standalone generator tools (scripts/generate_shape_scale.py,
scripts/generate_heating.py). Extracted out of 01_preprocess.py so those
other modules can import this logic without needing to import
01_preprocess.py itself (not possible — a leading digit is not a valid
Python module-name character).
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr


def build_grid_params(cfg):
    zw = cfg["zw"]
    match zw:
        case 42:  return 42, 64, 128
        case 63:  return 63, 96, 192
        case 124: return 124, 188, 376
        case _:   raise ValueError(f"Unsupported zw={zw}. Supported: 42, 63, 124.")


def season_month_indices(season):
    """0-based month indices into a 12-element monthly climatology array."""
    s = season.upper()
    if s == "DJF": return 11, 0, 1
    if s == "JJA": return 5, 6, 7
    if s == "MAM": return 2, 3, 4
    if s == "SON": return 8, 9, 10
    raise ValueError(f"Unsupported season '{season}'. Use DJF, JJA, MAM, or SON.")


def setup_spectral_transforms(jmax, imax, mw, zw):
    import torch_harmonics as th
    import torch_harmonics.distributed as dist
    from subs1_utils import precompute_latitudes

    cost_lg, wlg, lats = precompute_latitudes(jmax)
    lats = 90 - 180 * lats / np.pi
    lons = np.linspace(0.0, 360.0 - 360.0 / imax, imax)
    dlatlon = xr.Dataset({"lat": lats, "lon": lons})

    vsht   = th.RealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    dsht   = dist.DistributedRealSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    disht  = dist.DistributedInverseRealSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    dvsht  = dist.DistributedRealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    divsht = dist.DistributedInverseRealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)

    return cost_lg, lats, lons, dlatlon, vsht, dsht, disht, dvsht, divsht


def vertical_structure(kmax, delsig):
    """Return normalized vertical heating profile weights."""
    vert_struc = np.zeros(kmax)
    if kmax == 11:
        vals = [0.0, 0.1, 0.2, 1.5, 1.9, 1.5, 0.9, 0.5, 0.2, 0.1, 0.0]
    elif kmax == 26:
        vals = [0.0,0.0,0.0,0.0,0.0, 0.25,0.5,0.75,1.0,1.5,
                1.75,1.75,1.75,2.0,2.0,2.0,2.0,1.75,1.75,1.5,
                1.25,0.75,0.5,0.25,0.0,0.0]
    else:
        raise ValueError(f"No default vert_struc for kmax={kmax}.")
    for k, v in enumerate(vals):
        vert_struc[k] = v
    rnorm = (vert_struc * delsig).sum()
    return vert_struc / rnorm


def upsample_monthly_to_daily(data_monthly, varname, coords_3d=True, lev_coord=None,
                               lats=None, lons=None, sl=None):
    """
    Upsample 12-month climatology to 365 daily values via cubic interpolation.
    Uses the triple-year trick: concatenate 3 identical years, resample to daily,
    then extract the middle year (indices 335:700 = Jan 1 to Dec 31).

    data_monthly : numpy or torch array, shape (12, ...) or (12, lev, lat, lon)
    Returns      : torch tensor, shape (365, ...) with daily values
    """
    if torch.is_tensor(data_monthly):
        data_monthly = data_monthly.numpy()

    times1 = pd.date_range(start="1950-01-01", end="1951-01-01", freq="ME")
    times2 = pd.date_range(start="1951-01-01", end="1952-01-01", freq="ME")
    times3 = pd.date_range(start="1952-01-01", end="1953-01-01", freq="ME")

    coords = {"lat": lats, "lon": lons}
    if coords_3d and lev_coord is not None:
        dims  = ["time", "lev", "lat", "lon"]
        extra = {"lev": lev_coord}
    else:
        dims  = ["time", "lat", "lon"]
        extra = {}

    def _make_ds(times):
        c = {"time": times, **extra, **coords}
        return xr.Dataset({varname: (dims, data_monthly)}, coords=c)

    dT = xr.concat([_make_ds(times1), _make_ds(times2), _make_ds(times3)], dim="time")
    Daily = dT.resample(time="D").interpolate("cubic")
    out = torch.from_numpy(Daily[varname][335:700].values)
    return out
