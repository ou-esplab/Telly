#!/usr/bin/env python
# coding: utf-8

# In[1]:


import platform
import subprocess
import argparse

import cartopy.crs as ccrs
import dask
import matplotlib.pyplot as plt
import metpy
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import scipy as sp
import xarray


# # Model Variables
# In the following cell you can set the values of the variables relevant to the model. The details of each variable are included in the README. In most cases it is only necessary to set values for the standard variables. Note that any variable included in the model should be given the same value in the postprocess. For example, if the model used zw = 42 and kmax = 11, you should use zw = 42 and kmax = 11 below.

# In[2]:

# Parse commend line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--expname",nargs='?',default=None,help="experiment name")
parser.add_argument("--dayst",nargs='?',default=None,help="number of days to postprocess")
parser.add_argument("--datapath",nargs='?',default=None,help="experiment output directory (default: derived from expname under the platform-specific AGCM_Experiments path)")
parser.add_argument("--zw",nargs='?',type=int,default=63,help="zonal wavenumber")
parser.add_argument("--kmax",nargs='?',type=int,default=26,help="number of vertical levels")
args = parser.parse_args()

expstub=args.expname
dayst=int(args.dayst)

# Set postprocess parameters.

# Standard Variables
zw = args.zw
kmax = args.kmax
expname=expstub

DataSetnames=['vvel','uvel','geo']
Datanames=['v','u','geo']

# Advanced Variables
imax = None
jmax = None
#imax = 192
#jmax = 96
custom_path = args.datapath
custom_kmax = None


# In[3]:


# Set Dependent Variables

# Set value of kmax if custom_kmax is used.
if not(custom_kmax is None):
    kmax = custom_kmax
    print("Using custom value for kmax:", kmax)
# Otherwise check value for kmax.
elif kmax!=11 and kmax!=26:
    raise Exception("Unexpected value for kmax. Use custom_kmax and note that other values are implementable, but the user must modify subs1_utils.py routine bscst. If unclear email bkirtman@miami.edu for clarification.")

# Check value for zw.
# Afterwards, set jmax and imax values based on the value given to zw.
# If a value is already given for one of the listed variables, use that instead
match zw:
    case 42:
        jmax = 64 if (jmax is None) else jmax
        imax = 128 if (imax is None) else imax
    case 63:
        jmax = 96 if (jmax is None) else jmax
        imax = 192 if (imax is None) else imax
    case 124:
        jmax = 188 if (jmax is None) else jmax
        imax = 376 if (imax is None) else imax
    case _:
        if (jmax is None) or (imax is None):
            raise Exception("Unexpected value for zw. Other values are implementable, but the user must specify values for jmax and imax in the advanced variables section.")

print("zw =", zw,
      "\nkmax =", kmax,
      "\njmax =", jmax,
      "\nimax =", imax,
      "\ndayst =", dayst)


# In[4]:


# Set datapath.

# If custom_path was set, use that as the datapath.
# Otherwise create an appropriate datapath for the user's operating system.
user_platform = platform.system() if (custom_path is None) else "Custom Path"
print("Setting output datapath for", user_platform)
datapath = ''
match user_platform:
    case 'Custom Path':
        datapath = custom_path
    case 'Windows':
        foo = str(subprocess.check_output(['whoami']))
        end = len(foo) - 5
        uname = foo[2:end].split("\\\\")[1]
        datapath = "C:\\Users\\" + uname + "\\Documents\\AGCM_Experiments\\" + expname + "\\"
    case 'Darwin':
        foo = str(subprocess.check_output(['whoami']))
        end = len(foo) - 3
        uname = foo[2:end]
        datapath = '/Users/' + uname + '/Documents/AGCM_Experiments/' + expname + '/'
    case 'Linux':
        foo = str(subprocess.check_output(['whoami']))
        end = len(foo) - 3
        uname = foo[2:end]
        datapath = '/data/esplab/kpegion/projects/AGCM_Experiments/'+expname+'/'

    case _:
        raise Exception("Use case for this system/OS is not implemented. Consider using custom_path in the advanced variables.")

if not datapath.endswith('/'):
    datapath = datapath + '/'

# Set stamp for file names
stamp = 'days_1-' + str(dayst)

print("datapath =", datapath,
      "\nstamp =", stamp)


# In[5]:


fps = datapath+'lnps_????-??-??_????-??-??.nc' # always need surface pressure
print(fps)
dps = xarray.open_mfdataset(fps,decode_times=True,parallel = True)
#
#


for DataSetname,Dataname in zip(DataSetnames,Datanames):

    fdata = datapath+DataSetname+'_????-??-??_????-??-??.nc'
    ddata = xarray.open_mfdataset(fdata,decode_times=True,parallel = True)


    print(fps)
    print(dps)
    print(ddata)

    
    #
    # Create Data Array for Control Pressure level Data geopotenial, temp, u & v
    # 
    #
    lats = ddata['lat'].values
    lons = ddata['lon'].values
    plev = [850.0,500.0,300.0,200.0]
    plev_r = np.zeros(len(plev))
    for k in range(len(plev)):
        plev_r[k] = (plev[k])*100.0 # mb to Pa

    tmp = (dayst,len(plev),jmax,imax)

    dout = np.zeros(tmp)
    pressure = np.zeros((kmax,jmax,imax))
    siglevs = ddata['lev']

    for k in range (dayst):
        vv = ddata[Dataname][k,:,:,:]
        ps = dps.lnps[k,:,:]
        surfp = (np.exp(ps))*1000.0*100.0 # in Pa
        for kk in range(kmax):
            pressure[kk,:,:] = surfp[:,:]*siglevs[kk]
        vv = vv.compute()
        ps = ps.compute()
        dout[k] = metpy.interpolate.log_interpolate_1d(plev_r,pressure,vv, axis=0)

    times = ddata['time']
    dData = xarray.Dataset({Dataname: (['time','lev','lat','lon'],dout)},
                            coords={'time': times,'lev':plev, 'lat': lats, 'lon': lons})

    print(dout.shape)
    print(dData)
    dData.to_netcdf(datapath+DataSetname+'_Pressure_'+stamp+'.nc')

