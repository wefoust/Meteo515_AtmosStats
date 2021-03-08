"""
This module computes statistical calculations
"""
import numpy as np

def basicStats(data):
    """ A function that ingests a 1-D Xarray dimension,
    computes basic stats, and returns a dictionary containing
    max,min,mean, median, standard deviation,interquartile range,
    median absolute deviations, skewness, and the Yule-Kendall index"""
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
    """ A function that gets the Interquartile Range 
    of a vector"""
    q25,q75 = np.percentile(data,
                            [25,75],
                            interpolation = 'midpoint')
    IQR = (q75-q25)
    return IQR

def getSkewness(data):
    """ A function that gets the skewness of a vector"""
    std = np.std(data,ddof=1)
    errors = data-(np.mean(data))
    numerator = np.sum(np.power(errors,3))
    denominator = ((len(data)-1)*std*std*std)
    gamma = numerator/denominator
    return gamma

def getYuleKendall(data):
    """ A function that gets the Yule Kendall Index of a vector"""
    q25,q50,q75 = np.percentile(data,[25,50,75])
    gammaYK = ((q75-q50)-(q50-q25)) / (q75-q25)
    return gammaYK
    

