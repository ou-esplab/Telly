#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import subprocess
import sys
import argparse
import torch_harmonics as th
import torch_harmonics.distributed as dist
import pandas as pd
import matplotlib.pyplot as plt
from subs1_utils import *
import multiprocessing
from dask.distributed import Client
from dask.distributed import LocalCluster
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def worker():
    print("Processing...")

def main():
    process = multiprocessing.Process(target=worker)
    process.start()
    process.join()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows/PyInstaller

    cluster = LocalCluster()
    print(cluster)

    # Parse commend line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname",nargs='?',default=None,help="experiment name to be appended to AC_")
    parser.add_argument("--toffset",nargs='?',default=None,help="number of days that have already run; 0 for cold start")
    parser.add_argument("--ichunk",nargs='?',default=None,help="number of 30-day chunks to run")
    args = parser.parse_args()

    expstub=args.expname
    toffset=int(args.toffset)
    ichunk=int(args.ichunk)

# In[3]:


# For MacOS is higher than 12.3+
    if torch.backends.mps.is_available():
        print("Running on GPU")
   #    print("Running on CPU")
        device = torch.device("mps")
   #    device = torch.device("cpu")
        print("MPS is activated:",torch.backends.mps.is_built()) # Was the current version of PyTorch built with MPS activated?
    else:
        print("Running  on CPU")
        device = torch.device("cpu")

    print(device)


# In[4]:


### This cell initializes the model
###
###     First Define all spectral grids
###
    zw = 63 # zonal wave number
    mw = 63 # meridional wave number
    kmax = 26
    imax = 192
    jmax = 96
    steps_per_day = 216*1.5 ### Changing this number impliws time step changes and should
    
#    ichunk = 120
#    toffset = 2970
#                       be implemented carefully
#
#
# provide experiment name and data path for writing out data
# datapath may need to be edited for your system
#
    expname = 'AC_'+expstub
    foo = str(subprocess.check_output(['whoami']))
    end = len(foo) - 3
    uname = foo[2:end]
    datapath = '/data/esplab/kpegion/projects/AGCM_Experiments/'+expname+'/'
    prepath = '/data/esplab/kpegion/projects/AGCM/AnnualCycle/' #### Assumed to already exist and have preprocessing data
 #
    times = pd.date_range(start = '1950-01-01', end='2100-01-01', freq='D') # The calendar month matters in this version with an
 #                                                                         annual cycle
 #
 # If restart see below settings
 #
 #toffset = 60 # toffset is the number of days that have already run
    datapath_init = datapath #set equal to datapath if restarting in same directory
 #
    if ( toffset == 0): # Cold Start
        subprocess.call(['rm','-r', datapath])
        subprocess.check_output(['mkdir', datapath])
 #
 # Get Day of Year at start of run
 #
    itime_ac = str(times[toffset])
    year_ac = itime_ac[0:4]
    mon_ac = itime_ac[5:7]
    day_ac = itime_ac[8:10]
    if ( mon_ac == '02' ) and ( day_ac == '29' ):
        day_ac = '28'
    print(['Calendar for Annual Cycle',year_ac,mon_ac,day_ac])
    newdate = pd.Timestamp('1951-'+mon_ac+'-'+day_ac)
    daynumber=newdate.day_of_year - 1
    print(['Day of Year Minus 1 at Start of Run',daynumber])
 #
 
 
 # In[5]:
 
 
 # Get the Gaussian latitudes and equally spaced longitudes
 #
    cost_lg, wlg, lats = precompute_latitudes(jmax)
    lats = 90-180*lats/(np.pi)
    lons = np.linspace(0.0,360.0-360.0/imax,imax)
    Lon, Lat = np.meshgrid(lons, lats)
 #
 # Instantiate grid to spectral (dsht) and spectral to grid (disht) distibuted transforms
 #
    vsht = th.RealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    dsht = dist.DistributedRealSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    disht = dist.DistributedInverseRealSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    dvsht = dist.DistributedRealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
    divsht = dist.DistributedInverseRealVectorSHT(jmax, imax, lmax=mw, mmax=zw, grid="legendre-gauss", csphase=False)
 
 
 # In[6]:
 
 
 #
 #
 #
 # Initialize spectral fields (at rest or to be read in)
    def initialize(temp_newton,lnpsclim,kmax,mw,zw):
        for k in range (kmax):
            tmn1[k] = tmn1[k] + temp_newton[k]
            tmn2[k] = tmn2[k] + temp_newton[k]
            tmn3[k] = tmn3[k] + temp_newton[k]
        qmn1 = lnpsclim
        qmn2 = lnpsclim
        qmn3 = lnpsclim
        return tmn1,tmn2,tmn3,qmn1,qmn2,qmn3
 
 
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
    temp_newton = torch.load(prepath+'temp.spectral.pt') # This is only used when initializing from a cold start
 #                                    
 #lnpsclim = torch.load(prepath+'lnps.spectral.pt') # lnps climatology for damping & for initializing a cold start
    lnpsclim = torch.load(prepath+'lnps.spectral.pt') 
 #
 # Read Climatology on Gausian Grid
 #
    tclim_gg = torch.load(prepath+'tsig.ggrid.pt')
    vortclim_gg = torch.load(prepath+'vortsig.ggrid.pt')
    divclim_gg = torch.load(prepath+'divsig.ggrid.pt')
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
        initialize(tclim[daynumber],lnpsclim[daynumber],kmax,mw,zw)
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
    phismn = torch.load(prepath+'topog.spectral.pt')
 #
 #
 # Read shape and scale for gamma rainfall distribution
 #
    shape = torch.load(prepath+'shapeAC.pt')
    scale = torch.load(prepath+'scaleAC.pt')
 #
 
 
 # In[ ]:
 
 
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
 
 
 # In[ ]:
 
 
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
    heatt = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
 #
    ddtdiv = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
    ddtvort = ddtdiv.detach().clone()
    ttend = ddtdiv.detach().clone()
 #
    vort = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    div = vort .detach().clone()
    temp = vort.detach().clone()
    qdot = vort.detach().clone()
 #
 # 
 #   ichunk = 120
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
 #
    heat = latent_heat_release(dsht,disht,shape[daynumber],scale[daynumber],
                               imax,jmax,kmax,Lat,delsig,torch.Tensor.numpy(div[kmax-1]),ii)
 #
    while ii < nstep:
        ii = ii + 1
        savedat = savedat + 1
        zmnt[daycount] = zmnt[daycount] + zmn1/steps_per_day
        dmnt[daycount] = dmnt[daycount] + dmn1/steps_per_day
        tmnt[daycount] = tmnt[daycount] + tmn1/steps_per_day
        qmnt[daycount] = qmnt[daycount] + qmn1/steps_per_day
        wmnt[daycount] = wmnt[daycount] + wmn1/steps_per_day
        heatt[daycount] = heatt[daycount] + heat/steps_per_day
        if (savedat == steps_per_day): # post processing
         #
         #
            heat = latent_heat_release(dsht,disht,shape[daynumber],scale[daynumber],
                                       imax,jmax,kmax,Lat,delsig,torch.Tensor.numpy(div[kmax-1]),ii)
         #
         # Call Postprocessing Routine as needed
         #
            print((dmn1[10,2,2],dmn1[9,2,1]))
            print(('heat',heat[20,47,95]))
            daycount = daycount + 1
            total_days = total_days + 1
            print(['Day = ',total_days])
            print(['Date = ',times[total_days+toffset]])
         #
            itime_ac = str(times[total_days+toffset])
            year_ac = itime_ac[0:4]
            mon_ac = itime_ac[5:7]
            day_ac = itime_ac[8:10]
            if ( mon_ac == '02' ) and ( day_ac == '29' ):
                day_ac = '28'
            print(['Calendar for Annual Cycle',year_ac,mon_ac,day_ac])
            newdate = pd.Timestamp('1951-'+mon_ac+'-'+day_ac)
            daynumber=newdate.day_of_year - 1
            print(['Day of Year Minus 1',daynumber])
 #
            if (daycount == tl):
                times_30day = times[total_days-tl+toffset:total_days+toffset]
                postprocessing(disht,divsht,zmnt,dmnt,tmnt,qmnt,wmnt,heatt,\
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
                           tclim[daynumber],lnpsclim[daynumber],vortclim[daynumber],divclim[daynumber],kmax,mw,zw)
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
 
 
 # In[ ]:
 
 
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
 
 
# In[ ]:




