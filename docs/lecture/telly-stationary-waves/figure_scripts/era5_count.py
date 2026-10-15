import sys, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
f=sys.argv[1]
z=xr.open_dataset(f).z.sortby("latitude"); zs=(z-z.mean("longitude"))
zs=zs.assign_coords(longitude=((zs.longitude+180)%360)-180).sortby("longitude")
lon=zs.longitude.values; lat=zs.latitude.values
LO,HI=50,60
band=zs.sel(latitude=slice(LO,HI)).mean("latitude").values
# extrema of the band (smoothed lightly, cyclic)
b=np.convolve(np.r_[band[-8:],band,band[:8]],np.ones(9)/9,mode="same")[8:-8]
ext=[]
for i in range(len(b)):
    p,n=b[i-1],b[(i+1)%len(b)]
    if b[i]>p and b[i]>n and b[i]>15: ext.append((lon[i],band[i],"H"))
    if b[i]<p and b[i]<n and b[i]<-15: ext.append((lon[i],band[i],"L"))
print(ext)
levels=np.arange(-200,201,20); NAVY="#14233A"
fig=plt.figure(figsize=(10.4,6.2),constrained_layout=True)
gs=fig.add_gridspec(2,1,height_ratios=[2.5,1.35])
ax=fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-10,85],crs=ccrs.PlateCarree())
cf=ax.contourf(zs.longitude,zs.latitude,zs,levels=levels,cmap="RdBu_r",extend="both",transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE,linewidth=0.7,edgecolor="k")
ax.add_patch(plt.Rectangle((-180,LO),360,HI-LO,fill=False,ec=NAVY,lw=2.2,ls="--",transform=ccrs.PlateCarree()))
for x,v,t in ext:
    ax.text(x,(LO+HI)/2,t,transform=ccrs.PlateCarree(),ha="center",va="center",fontsize=17,fontweight="bold",color="white" if abs(v)>60 else NAVY,
            bbox=dict(boxstyle="circle,pad=0.25",fc="#B8501A" if t=="H" else "#14707C",ec="white",lw=1.5))
ax.set_xticks(range(-180,181,60),crs=ccrs.PlateCarree()); ax.set_yticks(range(0,81,30),crs=ccrs.PlateCarree()); ax.tick_params(labelsize=11)
ax.text(-178,LO-3,"50°–60°N",fontsize=13,color=NAVY,fontweight="bold",va="top")
cb=fig.colorbar(cf,ax=ax,orientation="vertical",shrink=0.85,pad=0.015,aspect=25); cb.set_label("Height anomaly (m)",fontsize=12); cb.ax.tick_params(labelsize=10)
a2=fig.add_subplot(gs[1])
a2.axhline(0,color="#888",lw=0.8); a2.plot(lon,band,color=NAVY,lw=2.6); a2.fill_between(lon,band,0,where=band>0,color="#E8894A",alpha=0.45); a2.fill_between(lon,band,0,where=band<0,color="#5BA3BE",alpha=0.45)
for x,v,t in ext:
    a2.text(x,v+(18 if t=="H" else -18),t,ha="center",va="center",fontsize=16,fontweight="bold",color="#B8501A" if t=="H" else "#14707C")
a2.set_xlim(-180,180); a2.set_xticks(range(-180,181,60)); a2.tick_params(labelsize=11)
a2.set_ylabel("Height anomaly (m)",fontsize=12); a2.set_xlabel("Longitude",fontsize=12); a2.set_ylim(band.min()-45,band.max()+45)
a2.set_title("Along 50°–60°N, once around the globe",fontsize=13,color=NAVY,loc="left")
fig.savefig("era5_count.png",dpi=150)
