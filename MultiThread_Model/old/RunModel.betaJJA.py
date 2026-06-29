#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import path
import pathlib
import platform
import subprocess
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_harmonics as th
import torch_harmonics.distributed as dist

from subs1_utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# # Model Variables
# In the following cell you can set the values of the variables relevant to the model. The details of each variable are included in the README. In most cases it is only necessary to set values for the standard variables.

# In[2]:

# Parse commend line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--expname",nargs='?',default=None,help="experiment name to be appended to T63L26_")
parser.add_argument("--toffset",nargs='?',default=None,help="number of days that have already run; 0 for cold start")
parser.add_argument("--ichunk",nargs='?',default=None,help="number of 30-day chunks to run")
args = parser.parse_args()

expstub=args.expname
toffset=int(args.toffset)
ichunk=int(args.ichunk)

# Set model parameters.

# Standard Variables
zw = 63
kmax = 26
expname = 'T63L26_'+expstub
#toffset = 150
#ichunk = 19

datapath_init = '/data/esplab/kpegion/projects/AGCM/MultiThread_Model/preprocess__zw_63__kmax_26_JJA_1979-2023/'

# Advanced Variables
mw = None
jmax = None
imax = None
steps_per_day = None
custom_path = None
custom_kmax = None


# In[3]:


# Initialize the model.

# Set value of kmax if custom_kmax is used.
if not(custom_kmax is None):
    kmax = custom_kmax
    print("Using custom value for kmax:", kmax)
# Otherwise check value for kmax.
elif kmax!=11 and kmax!=26:
    raise Exception("Unexpected value for kmax. Use custom_kmax and note that other values are implementable, but the user must modify subs1_utils.py routine bscst. If unclear email bkirtman@miami.edu for clarification.")

# Check value for zw.
# Afterwards, set mw, jmax, imax, and steps_per_day values based on the value given to zw.
# If a value is already given for one of the listed variables, use that instead.
match zw:
    case 42:
        mw = 42 if (mw is None) else mw
        jmax = 64 if (jmax is None) else jmax
        imax = 128 if (imax is None) else imax
        steps_per_day = 216 if (steps_per_day is None) else steps_per_day
    case 63:
        mw = 63 if (mw is None) else mw
        jmax = 96 if (jmax is None) else jmax
        imax = 192 if (imax is None) else imax
        steps_per_day = 324 if (steps_per_day is None) else steps_per_day
    case 124:
        mw = 124 if (mw is None) else mw
        jmax = 188 if (jmax is None) else jmax
        imax = 376 if (imax is None) else imax
        steps_per_day = 648 if (steps_per_day is None) else steps_per_day
    case _:
        if (mw is None) or (jmax is None) or (imax is None) or (steps_per_day is None):
            raise Exception("Unexpected value for zw. Other values are implementable, but the user must specify values for mw, jmax, imax, and steps_per_day in the advanced variables section.")
print("zw =", zw,
      "\nmw =", mw,
      "\nkmax =", kmax,
      "\njmax =", jmax,
      "\nimax =", imax,
      "\nsteps_per_day =", steps_per_day)

spec = (mw,zw,kmax)
grid = (imax,jmax,kmax)


# In[4]:


# Set preprocess path.

# Use this path whenever loading model data.
# The preprocess file shares a directory with the model and saves its data to
# a folder under that directory. This folder is named after the variable values in the data.
#preprocess_path = (
#    'preprocess'
#    + '__zw_' + str(zw)
#    + '__kmax_' + str(kmax)
#    + '_JJA/'
##    + '\\'
#)
preprocess_path=datapath_init

# Check that the path exists, throwing an exception if it doesn't.
folder = path.join(pathlib.Path().resolve(), preprocess_path)
print(folder)
if path.isdir(folder):
    print("Directory containing preprocess data was found.",
          "\npreprocess_path =", preprocess_path)
else:
    raise Exception("Directory containing preprocess data was not found. "
                    + "\npreprocess_path = " + str(preprocess_path)
                    + "\nfull path = " + str(folder)
                    + "\nRun preprocess.ipynb prior to running the model."
                    + "\nPreprocess must use the same variable values as the model."
    )


# In[5]:


print(folder)


# In[6]:


# Set output datapath.

# If custom_path was set, use that as the output datapath.
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
        if ( toffset == 0): # Cold Start
            subprocess.run(['rmdir', '/s', '/q', datapath], shell=True)
            subprocess.run(['mkdir', datapath], shell=True)
    case 'Darwin':
        foo = str(subprocess.check_output(['whoami']))
        end = len(foo) - 3
        uname = foo[2:end]
        datapath = '/Users/'+uname+'/Documents/AGCM_Experiments/'+expname+'/'
        if ( toffset == 0): # Cold Start
            subprocess.call(['rm','-r', datapath])
            subprocess.check_output(['mkdir', datapath])
    case 'Linux':
        foo = str(subprocess.check_output(['whoami']))
        end = len(foo) - 3
        uname = foo[2:end]
        datapath = '/data/esplab/kpegion/projects/AGCM/MultiThread_Model/experiments/'+expname+'/'
        if ( toffset == 0): # Cold Start
            subprocess.call(['rm','-r', datapath])
            subprocess.check_output(['mkdir',datapath])
    case _:
        raise Exception("Use case for this system/OS is not implemented. Consider using custom_path in the advanced variables.")
datapath_init = datapath if (datapath_init is None) else datapath
print("datapath =", datapath,
      "\ndatapath_init =", datapath_init)

cost_lg, wlg, lats, lons, vsht, dsht, disht, dvsht, divsht = set_spectral_transforms(jmax, imax, mw, zw)


# In[7]:


# For MacOS is higher than 12.3+
if torch.backends.mps.is_available():
    print("Running on GPU")
    device = torch.device("mps")
    print("MPS is activated:",torch.backends.mps.is_built()) # Was the current version of PyTorch built with MPS activated?
else:
    print("Running  on CPU")
    device = torch.device("cpu")

device


# In[8]:


#
# Initialization: Could read spectral restarts, or could
#              start at rest. If wanting to use grid point see
#              jupyter notebook preprocess
#              
#
# Implement at rest initial condition, but need coriolis since
# model predicts total vorticity
#
coriolis = np.broadcast_to([(4.0*np.pi/86400)*np.sin(lats[j,np.newaxis]*np.pi/180.0) for j in range(jmax)], (jmax, imax))
#
#
# Initialize spectral fields (at rest or to be read in)
# Need to read in background temperature data for Newtonian
# Relaxation and possible initialization, see preprocess
# for how to change source data or formulation.
#
temp_newton = torch.load(preprocess_path+'temp.spectral.pt') # newtonian 
#                                    relaxation temperature, spectral
lnpsclim = torch.load(preprocess_path+'lnps.spectral.pt') # lnps climatology for damping
#
# Read Climatology on Gausian Grid
#
tclim_gg = torch.load(preprocess_path+'tsig.ggrid.pt')
vortclim_gg = torch.load(preprocess_path+'vortsig.ggrid.pt')
divclim_gg = torch.load(preprocess_path+'divsig.ggrid.pt')
#
divclim = dsht(divclim_gg)
vortclim = dsht(vortclim_gg)
tclim = dsht(tclim_gg)
#
zmn1 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
zmn2 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
if ( toffset > 0 ): # Restart
    zmn1 = torch.load(datapath_init+'zmn1.spectral.pt')
    zmn2 = torch.load(datapath_init+'zmn2.spectral.pt')
zmn3 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
#
dmn1 = torch.zeros((kmax,mw,zw),dtype=torch.complex128) 
dmn2 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
if ( toffset > 0 ): # Restart
    dmn1 = torch.load(datapath_init+'dmn1.spectral.pt')
    dmn2 = torch.load(datapath_init+'dmn2.spectral.pt')
dmn3 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
#
tmn1 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
tmn2 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
if ( toffset > 0 ): # Restart
    tmn1 = torch.load(datapath_init+'tmn1.spectral.pt') 
    tmn2 = torch.load(datapath_init+'tmn2.spectral.pt')
tmn3 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
#
wmn1 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
wmn2 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
wmn3 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
#
qmn1 = torch.zeros((mw,zw),dtype=torch.complex128)
qmn2 = torch.zeros((mw,zw),dtype=torch.complex128)
if ( toffset > 0 ): # Restart
    qmn1 = torch.load(datapath_init+'qmn1.spectral.pt')
    qmn2 = torch.load(datapath_init+'qmn2.spectral.pt')
qmn3 = torch.zeros((mw,zw),dtype=torch.complex128)
#
#
if ( toffset == 0 ): # Only do this if cold start
    tmn1,tmn2,tmn3,qmn1,qmn2,qmn3 =\
    initialize(tclim,lnpsclim,kmax,mw,zw,tmn1,tmn2,tmn3)
#
# Topography data - this should be spectral data or can be
#                        initialized to zero. If grid point data
#                        is desired see preprocess for how to
#                        convert to spectral. 
#
# Setting topography to zero here
#
###phismn = torch.zeros((mw,zw),dtype=torch.complex128)
#
# If non-zero topog read here
#
phismn = torch.load(preprocess_path+'topog.spectral.pt')
#
#
# Adding heating here see preprocess.ipynb
#
heat = torch.load(preprocess_path+'heat.ggrid_'+expstub+'.pt')
#
# or set to zero
#
#####heat = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
#


# In[9]:


###
### Constants, parameters, vertical differencing parameters,
### matricies for geopotential height, etc ...
###
delsig, si, sl, sikap, slkap, cth1, cth2, r1b, r2b = bscst(kmax)
### The above code is in subs1_utils.py - vertical structure related
### This code would need to be changed if the vertical resolution
### is changed - could be done by simply specifying delsig in bscst
###
amtrx, cmtrx, dmtrx = mcoeff(kmax,si,sl,slkap,r1b,r2b,delsig)
### The above code is for geopotential height and implicit scheme
### in subs1_utils.py but unlikely any changes would be needed
emtrx = inv_em(dmtrx,steps_per_day,kmax,mw,zw)
### The above code
### emtrix is used in the implicit time scheme, computed once here to save cpu time
### changes unlikely


# In[10]:


#### Preprocessing is complete - now time to run model
#
#
# The Model Runs in 30-day chuncks - need to specify how many 30-day chunks to run
tl = 30 ##### tl is the chunk size - typically 30 days, but for testing 3 is reasonable
#
# Suggested ichunk for time dependent models: 120
#
ae = 6.371E+06 # Earth radius
tmnt = torch.zeros((tl,kmax,mw,zw),dtype=torch.complex128)
zmnt = tmnt.detach().clone()
dmnt = tmnt.detach().clone()
qmnt = torch.zeros((tl,mw,zw),dtype=torch.complex128)
wmnt = tmnt.detach().clone()
#
ddtdiv = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
ddtvort = ddtdiv.detach().clone()
ttend = ddtdiv.detach().clone()
#
vort = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
div = ddtdiv .detach().clone()
temp = ddtdiv.detach().clone()
qdot = ddtdiv.detach().clone()
#
times = pd.date_range(start = '1950-01-01', end='2100-01-01', freq='D')
# 
#ichunk = 20
#ichunk = 24      # For shortening the runtime while testing.
#
idays = tl * ichunk
#
#
#
#
# Begin Time Loop
#
ii = 0
savedat = 0
daycount = 0
total_days = 0
nstep = idays*steps_per_day
while ii < nstep:
    ii = ii + 1
    savedat = savedat + 1
    zmnt[daycount] = zmnt[daycount] + zmn1/steps_per_day
    dmnt[daycount] = dmnt[daycount] + dmn1/steps_per_day
    tmnt[daycount] = tmnt[daycount] + tmn1/steps_per_day
    qmnt[daycount] = qmnt[daycount] + qmn1/steps_per_day
    wmnt[daycount] = wmnt[daycount] + wmn1/steps_per_day
    if (savedat == steps_per_day): # post processing
        #
        # Call Postprocessing Routine as needed
        #
        print((dmn1[10,2,2],dmn1[9,2,1]))
        daycount = daycount + 1
        total_days = total_days + 1
        print(['Day = ',total_days])
        if (daycount == tl):
            times_30day = times[total_days-tl+toffset:total_days+toffset]
            postprocessing(disht,divsht,zmnt,dmnt,tmnt,qmnt,wmnt,\
                           phismn,amtrx,times_30day,mw,zw,\
                           kmax,imax,jmax,sl,lats,lons,tl,datapath)
            tmnt = torch.zeros((tl,kmax,mw,zw),dtype=torch.complex128)
            zmnt = torch.zeros((tl,kmax,mw,zw),dtype=torch.complex128)
            dmnt = torch.zeros((tl,kmax,mw,zw),dtype=torch.complex128)
            qmnt = torch.zeros((tl,mw,zw),dtype=torch.complex128)
            wmnt = torch.zeros((tl,kmax,mw,zw),dtype=torch.complex128)
            daycount = 0
        savedat = 0
    #
    # Run model for one time step
    #
    # Spectral to grid transformation of needed fields:
    #   Vorticity, divergence, temperature, U, V, 
    #   grad(ln(Ps)), Q prescibed heating
    #
    #
    vort = disht(zmn2) ### This is the relative vorticity
    div = disht(dmn2)
    temp = disht(tmn2)
    qdot = disht(wmn2)
    u,v = uv(divsht,zmn2,dmn2,mw,zw,kmax,imax,jmax)
    dxq,dyq = gradq(divsht,qmn2,mw,zw,imax,jmax)
    #
    # Non-Linear products
    #
    a,b,e,ut,vt,ri,wj,cbar,dbar = nlprod(u,v,vort,div,temp,dxq,dyq,heat,coriolis,delsig,si,sikap,slkap,\
                                         r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax)
    #
    #
    # Grid to spectral transformation of nlprod results
    #
    ddtdiv,ddtvort = vortdivspec(vsht,a,b,kmax,mw,zw)
    zmn3 = - ddtvort
    dmn3 = ddtdiv - lap_sht(dsht,e,mw,zw)
    _,ttend = vortdivspec(vsht,ut,vt,kmax,mw,zw)
    #
    tmn3 = -ttend + dsht(ri)
    wmn3 = dsht(wj) ### Prescribed heating converted to spectral
    qmn3 = -dsht(cbar) ### Only cbar here since dbar is included in implicit or explicit
    # 
    # Diffusion, Damping, Implicit or Explicit time differencing, Time filter
    #
    zmn3,dmn3,tmn3 = diffsn(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,mw,zw)
    #
    zmn3,dmn3,tmn3,qmn3 = damp_test(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,\
                          tclim,lnpsclim,vortclim,divclim,kmax,mw,zw)
    #
    dt = 86400.0/steps_per_day
    #
    zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,tmn3,qmn1,qmn2,qmn3 = \
                        explicit(dt,amtrx,cmtrx,dmtrx,emtrx,\
                        zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,\
                        tmn3,wmn1,wmn2,wmn3,qmn1,qmn2,qmn3,phismn,\
                        delsig,kmax,mw,zw)
    ####
    ## Reset zmn3, dmn3, tmn3,wmn3 & qmn3
    ###
    zmn3 = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
    dmn3 = zmn3.detach().clone()
    tmn3 = zmn3.detach().clone()
    wmn3 = zmn3.detach().clone()
    qmn3 = torch.zeros((mw,zw),dtype=torch.complex128)
#
# Done


# In[11]:


## Write spectral data for possible restart
##
torch.save(zmn1,datapath+'zmn1.spectral.pt')
torch.save(zmn2,datapath+'zmn2.spectral.pt')
torch.save(dmn1,datapath+'dmn1.spectral.pt')
torch.save(dmn2,datapath+'dmn2.spectral.pt')
torch.save(tmn1,datapath+'tmn1.spectral.pt')
torch.save(tmn2,datapath+'tmn2.spectral.pt')
torch.save(qmn1,datapath+'qmn1.spectral.pt')
torch.save(qmn2,datapath+'qmn2.spectral.pt')

