import numpy as np, xarray as xr, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=6.371e6; Om=7.292e-5
z=xr.open_dataset("era5_z500_tmean_1991_2020.nc").z; u=xr.open_dataset("era5_u500_tmean_1991_2020.nc")
u=u[list(u.data_vars)[0]]
z=z.sortby("latitude"); u=u.sortby("latitude")
P0,P1=30.0,70.0
zb=z.sel(latitude=slice(P0,P1)); ub=u.sel(latitude=slice(P0,P1))
lat=zb.latitude.values; lon=zb.longitude.values
zs=zb-zb.mean("longitude")
w=np.cos(np.deg2rad(lat)); ubar=float((ub.mean("longitude")*w).sum()/w.sum())
phic=50.0; beta=2*Om*np.cos(np.deg2rad(phic))/a; Ks=np.sqrt(beta/ubar)
print("ubar band mean",ubar,"Ks",Ks,"wavelength km",2*np.pi/Ks/1e3,"n_s at 50N",Ks*a*np.cos(np.deg2rad(phic)))
F=np.fft.rfft(zs.values,axis=1)/len(lon)                 # (lat, n)
dphi=lat[-1]-lat[0]; Ly=np.deg2rad(dphi)*a
N=range(1,7); M=range(0,7)
E=np.zeros((len(N),len(M)))
for i,n in enumerate(N):
    for j,m in enumerate(M):
        basis=np.cos(m*np.pi*(lat-P0)/dphi)
        c=np.trapz(F[:,n]*basis,lat)/np.trapz(basis**2,lat)
        E[i,j]=2*abs(c)**2                                # amplitude^2 of the (n,m) wave, m: half-waves across band
k=np.array(list(N))/(a*np.cos(np.deg2rad(phic))); l=np.array(list(M))*np.pi/Ly
KK=np.sqrt(k[:,None]**2+l[None,:]**2)
tot=E.sum(); print("share of variance by n:",np.round(E.sum(1)/tot,2)); print("share by m:",np.round(E.sum(0)/tot,2))
print("energy-weighted K / Ks:",float((E*KK).sum()/tot/Ks))
for r in (0.5,0.7): print("share within Ks*(1±%.1f):"%r,float(E[(KK>Ks*(1-r))&(KK<Ks*(1+r))].sum()/tot))
# figure
fig,ax=plt.subplots(figsize=(8.4,6.4),constrained_layout=True)
Kg=np.linspace(0,2.4e-6,200); 
ulo,uhi=ubar*0.7,ubar*1.3
ax.add_patch(plt.matplotlib.patches.Wedge((0,0),np.sqrt(beta/(ubar*0.7))*1e6,0,90,width=np.sqrt(beta/(ubar*0.7))*1e6-np.sqrt(beta/(ubar*1.3))*1e6,color="#14707C",alpha=0.18,lw=0))
th=np.linspace(0,np.pi/2,100); ax.plot(Ks*1e6*np.cos(th),Ks*1e6*np.sin(th),color="#14707C",lw=3)
sc=ax.scatter((k[:,None]+0*l[None,:]).ravel()*1e6,(0*k[:,None]+l[None,:]).ravel()*1e6,s=(E.ravel()/E.max())*1800+10,color="#B8501A",alpha=0.75,edgecolor="#7a3410",lw=0.8)
ax.set_xlim(0,2.2); ax.set_ylim(0,2.2); ax.set_aspect("equal")
ax.set_xlabel("East–west wavenumber k (10⁻⁶ per metre)",fontsize=14); ax.set_ylabel("North–south wavenumber l (10⁻⁶ per metre)",fontsize=14)
ax.tick_params(labelsize=12)
sec=ax.secondary_xaxis("top",functions=(lambda x:x*1e-6*a*np.cos(np.deg2rad(phic)),lambda n:n/(a*np.cos(np.deg2rad(phic)))*1e6)); sec.set_xlabel("Waves around the 50°N circle",fontsize=13); sec.tick_params(labelsize=12)
ax.text(Ks*1e6*0.707+0.08,Ks*1e6*0.707+0.08,"k² + l² = Ks²",color="#14707C",fontsize=15,fontweight="bold")
ax.text(1.45,0.35,"Orange dots: ERA5\n(bigger = more of the\nstationary-wave variance)",fontsize=13,color="#7a3410")
ax.text(1.33,1.00,"Teal band:\nKs for ū ± 30%",fontsize=13,color="#14707C")
fig.savefig("era5_k_l.png",dpi=150)
