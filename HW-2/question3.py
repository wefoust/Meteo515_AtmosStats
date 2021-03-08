#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question3
"""
#%% Importing Libraries and Ingesting/Formatting Data
from netCDF4 import Dataset
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
file1 = 'FRCP85C5CN.001.2006-2019.CNTRL.cam.h1.PRECT.20060101-20191231.nc'
file2 = 'FRCP85C5CN.001.2086-2099.CNTRL.cam.h1.PRECT.20860101-20991231.nc'
a = Dataset(file1,mode='r')
b = Dataset(file2,mode='r')
dataset1 = xr.open_dataset(xr.backends.NetCDF4DataStore(a))
dataset2 = xr.open_dataset(xr.backends.NetCDF4DataStore(b))



#%% Truncating Data
# Setting lats to 130+180 & -60+180
mask = ((dataset1['lat']>19) &
       (dataset1['lat']<51)  &
       (dataset1['lon']>229) &
       (dataset1['lon']<311))

precip20062019=dataset1.where(mask,drop=True)['PRECT']*1000*60*60*24
precip20862099=dataset2.where(mask,drop=True)['PRECT']*1000*60*60*24
lats = dataset1.where(mask,drop=True)['lat']
lons = dataset1.where(mask,drop=True)['lon']

#precip20062019.where(mask,drop=True)
#precip20862099.where(mask,drop=True)


Conus20102019 = precip20062019.sel(time = slice("2010-01-01",None))                                   
Conus20902099 = precip20862099.sel(time = slice("2090-01-01",None))
DJF20102019 = Conus20102019.where(Conus20102019.time.dt.season.isin(['DJF']),drop=True)
DJF20902099 = Conus20902099.where(Conus20902099.time.dt.season.isin(['DJF']),drop=True)

#Calculating Mean, Diff in Median, and Diff in IQR between datasets
DJF20102019Median = DJF20102019.median(dim='time')
DJF20902099Median = DJF20902099.median(dim='time')
DJFDiff = DJF20902099Median - DJF20102019Median
DJF20102019IQR = DJF20102019.quantile(.75,dim='time') - DJF20102019.quantile(.25,dim='time')
DJF20902099IQR = DJF20902099.quantile(.75,dim='time') - DJF20902099.quantile(.25,dim='time')
DJFDiffquantiles = DJF20902099IQR-DJF20102019IQR



#%% Plotting Figures
dataArray = [[DJF20102019Median,DJF20902099Median,DJFDiff],
             [DJF20102019IQR,DJF20902099IQR,DJFDiffquantiles]]
levels =[[np.linspace(0,5,11),np.linspace(0,5,11),np.linspace(-2.5,2.5,11)],
         [np.linspace(0,15,16),np.linspace(0,15,16),np.linspace(-2.5,2.5,11)]]
titles = [['2010-2019 Median','2090-2099 Median','2090-2099 Diff'],
          ['2010-2019 IQR','2090-2099 IQR ','2090-2099 IQR Diff']]

fig, ax = plt.subplots(3,2,subplot_kw={'projection': ccrs.PlateCarree()})
for i in np.arange(0,3):
    for j in np.arange(0,2):
        print('I value is ',i)
        print('J value is ',j)
        if i == 2:
            colors = 'bwr'
        else:
            colors ='YlGn'
        figure = ax[i,j].contourf(lons,lats,dataArray[j][i],
                     cmap=colors,
                     levels=levels[j][i],
                     transform=ccrs.PlateCarree())
        fig.colorbar(figure,ax=ax[i,j],orientation='horizontal',aspect=12.1,shrink=1)
        ax[i,j].set_title(titles[j][i])
        ax[i,j].coastlines()
        ax[i,j].add_feature(cfeature.STATES, 
                        zorder=1, 
                        linewidth=1.5, 
                        edgecolor='k')
        plt.tight_layout()
fig.savefig('HW2_3B_Plot.png',
            bbox_inches='tight',
            dpi=250)        
#%%
concatData = xr.concat([Conus20102019,Conus20902099],dim='time')
loop = 1000
diffMedian = []
diffIQR = []


for i in np.arange(loop):
    permuteData = np.random.permutation(concatData)
    lower,upper = np.split(permuteData,2)
    median = np.median(upper,axis=0)-np.median(lower,axis=0)
    IQR = (np.quantile(upper,.75,axis=0)-np.quantile(upper,.25,axis=0)) -\
          (np.quantile(lower,.75,axis=0)-np.quantile(lower,.25,axis=0))
    diffMedian.append(median)
    diffIQR.append(IQR)

diffMedian = np.asarray(diffMedian)
diffIQR = np.asarray(diffIQR)

sigMedian = np.percentile(diffMedian,.975,axis=0)
sigIQR = np.percentile(diffIQR,.975,axis=0)

medianHatchRgn = DJFDiff-sigMedian
IQRHatchRgn = DJFDiffquantiles-sigIQR

#%% Last Plots
fig, ax = plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree()})
#"""
ax[0].contourf(lons,lats,medianHatchRgn,
                 cmap=colors,
                 levels=[-2,-1.5,-1,-.5,0,.5,1,1.5,2],
                 hatches = ["","","",'','','xx','xx'],
                 facecolor=None,
                 transform=ccrs.PlateCarree())
#"""

#"""
ax[0].contourf(lons,lats,DJFDiff,
                 cmap=colors,
                 levels=[-2,-1.5,-1,-.5,0,.5,1,1.5,2],
                 #facecolor = None,
                 transform=ccrs.PlateCarree())
#"""

fig.colorbar(figure,ax=ax[0],orientation='horizontal',aspect=12.1,shrink=1)
ax[0].set_title('Median Sig')
ax[0].coastlines()
ax[0].add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')


ax[1].contourf(lons,lats,IQRHatchRgn,
                 cmap=colors,
                 levels=[-1,-.5,0,.5,1],
                 hatches = ["","","xx",'xx','xx'],
                 transform=ccrs.PlateCarree())
fig.colorbar(figure,ax=ax[1],orientation='horizontal',aspect=12.1,shrink=1)
ax[1].set_title('IQR Sig')
ax[1].coastlines()
ax[1].add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')

plt.tight_layout()
#fig.savefig('HW2_3D_Plot.png',
#            bbox_inches='tight',
#            dpi=250)
