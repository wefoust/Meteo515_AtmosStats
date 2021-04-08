#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:53:06 2021

@author: wef5056
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def getData(header,filename,years):
    dates = np.array('1950-01', dtype=np.datetime64)
    dates = dates+np.arange(years*12)
    df = pd.read_csv(filename,delim_whitespace=False,names=header)
    ensoTS = df.iloc[:-1,1:].to_numpy()                  # convert to np array
    ensoTS  = ensoTS .reshape(np.size(ensoTS ))          # reshape to 1d
    return ensoTS

def detrend(ts):
    dtrendTS = ts*0
    for i in np.arange(0,ts.shape[0]):
        regress = stats.linregress(range(0,ts.shape[0]),ts)
        dtrendTS[i] = ts[i] - (i*regress[0]+regress[1])
    return dtrendTS

def makeTSPlot(dates,data, ax=None, **kwargs):
    ax=ax
    return ax.plot(dates,data)

def getPower(data,chunks):
    deltaT = 1
    record = len(data)
    numSteps = record/deltaT 
    dataFFT = np.fft.fft(data)
    frequencies = np.fft.fftfreq(data.size)
    power = np.square(np.abs(dataFFT))/numSteps
    chunkfrequencies = frequencies.reshape((-1, chunks),order='F').mean(axis=0)
    chunkpower = power.reshape((-1, chunks),order='F').sum(axis=0)
    return chunkfrequencies,chunkpower
def getAR1Fit(data,chunks,totalpower,chunkfreq):
    Msp = chunks/2
    h=np.arange(0,Msp)              
    phi = np.corrcoef(data[1:],data[:-1])[0,1] 
    rho1 = phi
    AR1Fit =(1-rho1)/(1-2*rho1*np.cos((h*np.pi)/Msp)+rho1**2)
    
    #AR1Fit = (1-(phi*phi))/(1-2*phi*np.cos((h*3.14)/Msp)+(phi**2))
    #AR1power = np.square(np.abs(AR1Fit))/data.size
    AR1power = sum(AR1Fit)
    AR1Fit = AR1Fit*(totalpower/AR1power) 
    return h,AR1Fit

def getSpectralSig(AR1Fit,alpha,chunks):
    Msp= chunks/2 
    alphaAposteriori = 1-(1-alpha)**(1/Msp)
    v = AR1Fit.size/Msp
    aprioriR1 = (AR1Fit/v)*stats.chi2.ppf((1-alpha),v)
    apostoriR1=(AR1Fit/v)*stats.chi2.ppf((1-alphaAposteriori),v)
    return aprioriR1,apostoriR1

def makePowerPlot(frequency,power,ax=None, **kwargs):
    ax=ax
    return ax.plot(frequency,power)

def makeSemiLogXPlot(frequency,power,ax=None, **kwargs):
    ax=ax
    return ax.semilogx(frequency,power)