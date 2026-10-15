import numpy as np, xarray as xr, json
Z="/data/esplab/shared/reanalysis/era5/daily/z500/z500.{y}.nc"; U="/data/esplab/shared/reanalysis/era5/daily/u500/u500.{y}.nc"
yrs=range(1991,2021)
z=xr.open_mfdataset([Z.format(y=y) for y in yrs],combine="by_coords",chunks={"time":60}).z.mean("time").compute()/9.80665
u=xr.open_mfdataset([U.format(y=y) for y in yrs],combine="by_coords",chunks={"time":60})
u=u[list(u.data_vars)[0]].mean("time").compute()
z.to_netcdf("era5_z500_tmean_1991_2020.nc"); u.to_netcdf("era5_u500_tmean_1991_2020.nc")
print(z.dims,u.dims,float(z.latitude[0]),float(z.latitude[-1]))
