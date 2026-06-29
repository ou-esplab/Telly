#!/usr/bin/env python
# coding: utf-8

# In[1]:


import platform
import subprocess

import cartopy.crs as ccrs
import dask
import matplotlib.pyplot as plt
import metpy
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import scipy as sp
import xarray


# In[2]:


expname='AC_Test'
DataSetname = 'geo'
Dataname = 'geo'
datapath = '/data/esplab/kpegion/projects/AGCM_Experiments/'+expname+'/'

yrs=np.arange(2027,2070)

zw = 63
jmax = 96 
imax = 192
kmax = 26


# In[5]:


for yr in yrs:
    
    fps = datapath+'lnps_'+str(yr)+'-??-??_????-??-??.nc' 
    dps = xarray.open_mfdataset(fps,decode_times=True,parallel = True)
    
    fdata = datapath+DataSetname+'_'+str(yr)+'-??-??_????-??-??.nc'
    ddata = xarray.open_mfdataset(fdata,decode_times=True,parallel = True)
    
    lats = ddata['lat'].values
    lons = ddata['lon'].values
    times = ddata['time']
    siglevs = ddata['lev']
    
    plev = [850.0,500.0,300.0,200.0]
    plev_r = np.array(plev)*100.0 # mb to Pa
    
    # Create numpy array for output data
    tmp = (len(times),len(plev),jmax,imax) 
    dout = np.zeros(tmp)
    pressure = np.zeros((kmax,jmax,imax))
    
    # Interpolate
    for k in range (len(times)):
        vv = ddata[Dataname][k,:,:,:]
        ps = dps.lnps[k,:,:]
        surfp = (np.exp(ps))*1000.0*100.0 # in Pa
        for kk in range(kmax):
            pressure[kk,:,:] = surfp[:,:]*siglevs[kk]
        #vv = vv.compute()
        #ps = ps.compute()
        dout[k] = metpy.interpolate.log_interpolate_1d(plev_r,pressure,vv.compute(), axis=0)
    
    # Make xarray.Dataset out of output data
    outData = xarray.Dataset({Dataname: (['time','lev','lat','lon'],dout)},
                            coords={'time': times,'lev':plev, 'lat': lats, 'lon': lons})
    
    # Write out data
    outFile=datapath+DataSetname+'_Pressure_'+str(yr)+'.nc'
    print("Writing: ", outFile)
    outData.to_netcdf(outFile)


# In[ ]:




