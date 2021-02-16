"""
This module is used to create plots for the assignment
"""

import matplotlib.pyplot  as plt
import numpy as np
from sklearn.neighbors import KernelDensity

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
                   linewidth=3)
    