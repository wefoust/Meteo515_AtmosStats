#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:29:34 2021

@author: wef5056
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from netCDF4 import Dataset

# Igesting data 
#Import surface data
dataNC = 'air.mon.mean.1948-2020.nc'
a = Dataset(dataNC,mode='r')
dataset1 = xr.open_dataset(xr.backends.NetCDF4DataStore(a))
ts = dataset1['air']
ts = ts.sel(time=ts.time.dt.month.isin([1]))
ts = ts[2:,:,:]
#import ENSO Index
header = ['Year', 'January','Febuary', 'March','April',
          'May','June','July','August',
          'September','October','November','December']
dataCSV = 'nina34.data.climatedatacenter.csv'
df = pd.read_csv(dataCSV,delim_whitespace=False,names=header)
df['Date'] = pd.to_datetime(df['Year'],format = '%Y')
df = df.set_index(['Date'])

# Converting to Jan ENSO and Standardized Jan ENSO
JanEnso = df['January'].to_xarray()
JanEnso = JanEnso.sel(Date = slice("1950-01-01","2020-01-01")) 
stdEnso = (JanEnso - JanEnso.mean(axis=0))/JanEnso.std(axis=0)

#%% Calculations
years = np.arange(1950,2021,1)

# Getting ENSO Regression stats
ensoRegress = stats.linregress(years,JanEnso.values)
ensoRegline = years*ensoRegress[0]+ensoRegress[1]

# Getting Correlation coeffs of temperature trends
squeezeLatLon = len(ts['lat'])*len(ts['lon'])
timeIndexes = len(ts['time'])
coreffmatrix =np.zeros((len(ts['lat']),len(ts['lon'])))
pvalmatrix=np.zeros((len(ts['lat']),len(ts['lon'])))
for i in np.arange(0,len(ts['lat']),1):
    for j in np.arange(0,len(ts['lon']),1):
        coreffmatrix[i,j] = stats.linregress(np.arange(0,timeIndexes,1),ts.values[:,i,j])[0]*100
        pvalmatrix[i,j] = stats.linregress(np.arange(0,timeIndexes,1),ts.values[:,i,j])[3]*1
pvalmatrix[pvalmatrix >.025] = np.nan

# Getting ENSO Datess
LaNinaDates = stdEnso.where(stdEnso<-JanEnso.std(axis=0),drop=True).coords['Date']
ElNinoDates = stdEnso.where(stdEnso>JanEnso.std(axis=0),drop=True).coords['Date']


#Getting Detrended Temperatures
detrendedTs = ts*0
for i in np.arange(0,detrendedTs.shape[1]):
    for j in np.arange(0,detrendedTs.shape[2]):
        regress = stats.linregress(years,ts[:,i,j].values)
        detrendedTs[:,i,j] = ts[:,i,j] - (years*regress[0]+regress[1])
        
#Calculating ENSO Composits
LaNinaCompositTs = detrendedTs.sel(time=LaNinaDates)
sigLaNina = stats.ttest_1samp(LaNinaCompositTs,0,axis=0)[1]
sigLaNina[sigLaNina >.025] = np.nan
LaNinaCompositTs = LaNinaCompositTs.mean(axis=0)


ElNinoCompositTs = detrendedTs.sel(time=ElNinoDates)
ElNinoCompositTs = ElNinoCompositTs.mean(dim='Date')

#Regressing Ts onto Enso
correlationMap = np.zeros((ts.shape[1],ts.shape[2]))
ENSOPredictTs = correlationMap*0
sigENSOTs = correlationMap*0
for i in np.arange(0,ts.shape[1]):
    for j in np.arange(0,ts.shape[2]):
        regress = stats.linregress(stdEnso,detrendedTs[:,i,j])
        correlationMap[i,j] = regress[2]
        sigENSOTs[i,j] = regress[3]
sigENSOTs[sigENSOTs >.025] = np.nan



#Filtering
ENSOFilteredTs = ts*0
for i in np.arange(0,ENSOFilteredTs.shape[1]):
    for j in np.arange(0,ENSOFilteredTs.shape[2]):
        regress = stats.linregress(stdEnso,detrendedTs[:,i,j])
        ENSOFilteredTs[:,i,j] = detrendedTs[:,i,j].values - (stdEnso*regress[0]+regress[1])

#% Calculating Uncorrelated Enso Composits

uncorrLaNinaCompositTs = ENSOFilteredTs.sel(time=LaNinaDates)
siguncorrLaNina = stats.ttest_1samp(uncorrLaNinaCompositTs,0,axis=0)[1]
siguncorrLaNina[siguncorrLaNina >.025] = np.nan
uncorrLaNinaCompositTs = uncorrLaNinaCompositTs.mean(axis=0)
#%% Plotting ENSO Time Series Question 2
xlabels = np.arange(1950,2010,10)
fig, ax = plt.subplots(2,sharex=True)
ax[0].plot(years,JanEnso)
ax[0].plot(years,ensoRegline,color='r')
ax[0].set_title('ENSO 3.4 Index')

ax[1].plot(years,stdEnso)
ax[1].set_title('Normalized ENSO 3.4 Index')
ax[1].set_yticks([-2.5,-JanEnso.std(axis=0),0,JanEnso.std(axis=0),2.5])
ax[1].axhline(JanEnso.std(axis=0),color='k',ls='--')
ax[1].axhline(-JanEnso.std(axis=0),color='k',ls='--')
fig.savefig('HW3_ENSO_TimeSeries.png',
            bbox_inches='tight',
            dpi=250)

#%% Plotting Trend Maps Question 3A
vmin = -4 
vmax = 13
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
figure = ax.contourf(ts['lon'],ts['lat'],coreffmatrix,
                     cmap='bwr',
                     levels = np.arange(-5,14,.5),
                     norm=norm,
                     transform=ccrs.PlateCarree())

#Significance overlay
ax.contourf(ts['lon'],ts['lat'],pvalmatrix,
                 hatches = ['xxx'],
                 colors = 'None',
                 transform=ccrs.PlateCarree())

fig.colorbar(figure,orientation='horizontal',pad=.05,aspect=12.1,shrink=1)
ax.set_title('January Surface Temperature Trends\n (\xb0C/Century)')
ax.coastlines()
ax.add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')
gl = ax.gridlines(draw_labels=True,alpha=0,color='k',linewidth=2)
gl.bottom_labels=False

fig.savefig('HW3_Ts_Trends.png',
            bbox_inches='tight',
            dpi=250)
#%% Plotting Composit Map 4A

vmin = -4 
vmax = 4
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
figure2 = ax2.contourf(ts['lon'],ts['lat'],LaNinaCompositTs,
                     cmap='bwr',
                     levels = np.arange(-4,4.1,.2),
                     norm=norm)

ax2.contourf(ts['lon'],ts['lat'],sigLaNina,
                 hatches = ['xxx'],
                 colors = 'None',
                 transform=ccrs.PlateCarree())

fig2.colorbar(figure2,orientation='horizontal',pad=.05,aspect=12.1,shrink=1)
ax2.set_title('El Nino and Surface Temperature Composite\n (\xb0C/Century)')
ax2.coastlines()
ax2.add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')
gl2 = ax2.gridlines(draw_labels=True,alpha=0,color='k',linewidth=2)
gl2.bottom_labels=False
ax2.set_extent([LaNinaCompositTs['lon'].min('lon'),
               LaNinaCompositTs['lon'].max('lon'),
               15.,
               55.])
fig2.savefig('HW3_ENSO_Ts_Composite.png',
            bbox_inches='tight',
            dpi=250)

#%% Plotting Correlation Maps 4B
vmin = -1 
vmax = 1
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
figure2 = ax2.contourf(ts['lon'],ts['lat'],correlationMap,
                     cmap='bwr',
                     levels = np.arange(-1,1.1,.1),
                     norm=norm)

ax2.contourf(ts['lon'],ts['lat'],sigENSOTs,
                 hatches = ['xxx'],
                 colors = 'None',
                 transform=ccrs.PlateCarree())

fig2.colorbar(figure2,orientation='horizontal',pad=.05,aspect=12.1,shrink=1)
ax2.set_title('ENSO and Surface Temperature Correlation Coefficient\n')
ax2.coastlines()
ax2.add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')
gl2 = ax2.gridlines(draw_labels=True,alpha=0,color='k',linewidth=2)
gl2.bottom_labels=False
fig2.savefig('HW3_ENSO_Ts_Correlation.png',
            bbox_inches='tight',
            dpi=250)



#%% Plotting Composit Map 4c Uncorrelated Composits
lon2, lat2 = np.meshgrid(ts['lon'],ts['lat'])
#x, y = m(lon2, lat2)

vmin = -4 
vmax = 4
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
figure2 = ax2.contourf(ts['lon'],ts['lat'],uncorrLaNinaCompositTs,
                     cmap='bwr',
                     levels = np.arange(-4,4.1,.2),
                     norm=norm)

#ax2.scatter(lon2,lat2,s=siguncorrLaNina,marker='X',color='k')
ax2.contourf(lon2,lat2,siguncorrLaNina,
             hatches = ['xxx'],
             colors = 'None',
             transform=ccrs.PlateCarree())

fig2.colorbar(figure2,orientation='horizontal',pad=.05,aspect=12.1,shrink=1)

ax2.set_title('Uncorrelated El Nino Surface Temperature Composite\n')
ax2.coastlines()
ax2.add_feature(cfeature.STATES, 
                    zorder=1, 
                    linewidth=1.5, 
                    edgecolor='k')
gl2 = ax2.gridlines(draw_labels=True,alpha=0,color='k',linewidth=2)
gl2.bottom_labels=False
ax2.set_extent([LaNinaCompositTs['lon'].min('lon'),
               LaNinaCompositTs['lon'].max('lon'),
               15,
               55])
fig2.savefig('HW3_Ts_Uncorrelated_ElNino.png',
            bbox_inches='tight',
            dpi=250)

