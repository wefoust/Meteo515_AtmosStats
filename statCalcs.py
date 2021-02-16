#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:24:51 2021

@author: wef5056
"""
import matplotlib.pyplot  as plt
import numpy as np
from sklearn.neighbors import KernelDensity


def basicStats(data):
    """ A function that ingests an 1-D Xarray dimension and
    computes basic statics and returns a dictionary containing
    max,min,mean, median, standard deviation,interquartile range,
    median absolute deviations, skewness, and Yule-Kendall index"""
    data = data.values
    stats = {}
    stats.update(maxValue = data.max())
    stats.update(minValue = data.min())
    stats.update(meanValue = data.mean())
    stats.update(medianValue= np.median(data))
    stats.update(std = np.std(data,ddof=1))
    stats.update(IQR = getIQR(data))
    stats.update(absMeanDeviation = np.mean(np.absolute(data-np.mean(data))))
    stats.update(skewness = getSkewness(data))
    stats.update(YuleKendall = getYuleKendall(data))
    return stats


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
    

