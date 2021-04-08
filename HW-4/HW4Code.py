"""
This script completes HW 4
"""
import numpy as np
import hw4calcs
import matplotlib.pyplot as plt
from scipy import stats

# Ingesting Data
header = ['Year',1,2,3,4,5,6,7,8,9,10,11,12]
filename = 'nina34.data.climatedatacenter.csv'
dates = np.array('1950-01', dtype=np.datetime64)
dates = dates+np.arange(71*12)

ensoTS = hw4calcs.getData(header,filename,71)
detrendENSO = hw4calcs.detrend(ensoTS)

#%% Creating First Plot
fig, ax = plt.subplots(2,sharex=True)
ax[0].plot(dates,ensoTS)
ax[0].set_title('ENSO 3.4 Index')
ax[0].set_xlabel('')
ax[0].set_ylabel('Index')

ax[1].plot(dates,detrendENSO)
ax[1].set_title('Detrended ENSO 3.4 Index')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Index')

fig.savefig('./figures/HW4_2panel_TS.png',
            bbox_inches='tight',
            dpi=250)  

#%%
""" This section does the following via a loop
    1) Sets chunking of N/1, N/3, N/6
    2) Sets constants
    3) Calculates power spectrum of detrended ENSO time series
    4) Calculates power spectrum of ENSO with Hanning Window applied
    5) Creates an AR1 Fit to the ENSO Power spectrum
    6) Creates Significance Thresholds (apriori and aposteriori)
    7) Plots the results 
"""    

plotFreqs = 1/(np.array([50,20,10,5,2,1,.5])*12)
xticklabels = ['50Yr', '20Yr','10Yr','5Yr','2Yr','1Yr','6Mos']
mh = [1, 3, 6]
title =['Mch=N/1','Mch=N/3','Mch=N/6']
for i in np.arange(0,3):
    chunks =int(detrendENSO.size/mh[i]) 
    alpha = .05
    record = detrendENSO.size
    window = np.hanning(record)
    chunfreq,powerENSO = hw4calcs.getPower(detrendENSO,chunks)
    totalPower = sum(powerENSO)
    chunkfreqhann,powerENSOHann = hw4calcs.getPower(window*detrendENSO,chunks)
    AR1Freqs,AR1Fit = hw4calcs.getAR1Fit(detrendENSO,chunks,totalPower,chunfreq)
    aprioriR1,apostoriR1 = hw4calcs.getSpectralSig(AR1Fit,alpha,chunks)
    mask = chunfreq >= 0 

    fig,ax=plt.subplots()
    ax.plot(chunfreq[mask],powerENSO[mask], label='ENSO')
    ax.plot(chunkfreqhann[mask],powerENSOHann[mask],color='C1', label='Hann')
    ax.plot(chunfreq[mask],AR1Fit,color='r', label='AR1')
    ax.plot(chunfreq[mask],aprioriR1, color='k', linestyle='--', label='AR1')
    ax.plot(chunfreq[mask],apostoriR1, color='k', linestyle='--', label='AR1')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_xticks(plotFreqs)
    ax.set_xticklabels(xticklabels,rotation=90)
    ax.set_title(title[i])
    ax.set_xlim(0,.2)
    ax.set_ylim(0,100)
    ax.legend(['ENSO','Hann','AR1','Significance'])
    fig.savefig('./figures/HW4_Power_{}.png'.format(i),
            bbox_inches='tight',
            dpi=250)  

    fig2,ax2=plt.subplots()
    ax2.semilogx(chunfreq[mask],chunfreq[mask]*powerENSO[mask],label='ENSO')
    ax2.semilogx(chunkfreqhann[mask],chunfreq[mask]*powerENSOHann[mask], color='C1', label='Hann')
    ax2.semilogx(chunfreq[mask],chunfreq[mask]*AR1Fit,color='r', label='AR1')
    ax2.semilogx(chunfreq[mask],chunfreq[mask]*aprioriR1,color='k',linestyle='--', label='AR1')
    ax2.semilogx(chunfreq[mask],chunfreq[mask]*apostoriR1, color='k', linestyle='--', label='AR1')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('f * Power')
    ax2.set_xticks(plotFreqs)
    ax2.set_xticklabels(xticklabels,rotation=90)
    ax2.set_title(title[i])
    ax2.set_xlim(0,.2)
    ax2.legend(['ENSO','Hann','AR1','Significance'])
    fig2.savefig('./figures/HW4_LogPower_{}.png'.format(i),
            bbox_inches='tight',
            dpi=250) 

#%% This section confirms Parceval's Theorm
chunfreq,powerENSO = hw4calcs.getPower(detrendENSO,1)
parceval = np.sum(detrendENSO**2) - np.sum(powerENSO)