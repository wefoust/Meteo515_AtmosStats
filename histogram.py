#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:24:51 2021

@author: wef5056
"""
import matplotlib.pyplot  as plt
import numpy as np
from sklearn.neighbors import KernelDensity

def getIQR(data):
    q25,q75 = np.percentile(data,[25,75], interpolation = 'midpoint')
    IQR = (q75-q25)
    return IQR

def getSkewness(data):
    std = np.std(data,ddof=1)
    errors = data-(np.mean(data))
    numerator = np.sum(np.power(errors,3))
    denominator = ((len(data)-1)*std*std*std)
    gamma = numerator/denominator
    return gamma

def getYuleKendall(data):
    q25,q50,q75 = np.percentile(data,[25,50,75])
    gammaYK = ((q75-q50)-(q50-q25)) / (q75-q25)
    return gammaYK
    

def gethist(data,ax=None,**kwargs):
    """This function ingests a 1d array or list and
    returns a plot of object of a basic histogram"""
    ax = ax or plt.gca()
    return ax.hist(data,**kwargs)

    
def getKDF(data,ax=None,**kwargs):
    """This function ingests a 1d array or list and 
    returns a plot object of a kernal
    density function"""
    data=data.values[:,None]
    ax = ax or plt.gca()    
    xValues = np.linspace(min(data),
                          max(data),
                          1000)
    kdensity = KernelDensity(**kwargs)
    kdensity.fit(data)
    logscore = kdensity.score_samples(xValues)
    return ax.plot(xValues,
                   np.exp(logscore),
                   'r--',
                   linewidth=5)
    