""" This completes the first assignment"""

import matplotlib.pyplot as plt
import pandas
from sklearn.neighbors import KernelDensity
import numpy as np
import xarray

import histogram


def getRange(input,min,max):
    """ This function gets range of values"""
    value = input.where(input > min,drop=True)
    value = value.where(value < max,drop=True)
    return value

def getConditionalProb(data,variable1,variable2,limit,cuttoff):
    """Calculates P(A|B) from an Xarray object with 
    user parameters defining variables and thresholds 
    very specific now""" 
    if cuttoff == 'l':
        print('yep')
        filteredData = data.where(data[variable1] < limit[0],drop=True)
    elif cuttoff == 'u':
        filteredData = data.where(data[variable1] > limit[0],drop=True)
    else:
        filteredData = getRange(data[variable1],limit[0],limit[1])
    truncatedData = filteredData[variable2].where(filteredData[variable2] != 0,drop=True)
    conditionalProb = len(truncatedData)/len(filteredData[variable1])
    return conditionalProb


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
    stats.update(IQR = histogram.getIQR(data))
    stats.update(absMeanDeviation = np.mean(np.absolute(data-np.mean(data))))
    stats.update(skewness = histogram.getSkewness(data))
    stats.update(YuleKendall = histogram.getYuleKendall(data))
    return stats
    

#Ingest xls file as Pandas Dataframe
filename = './SC_data_1980_2015.xls'
rawData = pandas.read_excel(filename)
rawData['Date'] = pandas.to_datetime(rawData['Date'],format = '%Y%m%d')
rawData = rawData.set_index(['Date'])
data = rawData[['Tmax','Tmin','PCP']].to_xarray()

#converting PCP values <=0.1 to 0
data['PCP'] = data['PCP'].where(data['PCP']>.01,0)

#splitting data into seasons
dataSeasonal = data.groupby('Date.season')
dataDJF = data.where(data.Date.dt.season.isin(['DJF']),drop=True)
dataMAM = data.where(data.Date.dt.season.isin(['MAM']),drop=True)
dataJJA = data.where(data.Date.dt.season.isin(['JJA']),drop=True)
dataSON = data.where(data.Date.dt.season.isin(['SON']),drop=True)

#filtering based on T1,T2,T3,T4
filterT1 = dataDJF.where((dataDJF['Tmax']<30),drop=True)
filterT2 = dataDJF.where((dataDJF['Tmax']>29) & 
                         (dataDJF['Tmax']<40),drop=True)
filterT3 = dataDJF.where((dataDJF['Tmax']>39) & 
                         (dataDJF['Tmax']<51),drop=True)
filterT4 = dataDJF.where((dataDJF['Tmax']>50),drop=True)
filterPCP = dataDJF.where((dataDJF['PCP']>0),drop=True)

P_T1 = len(filterT1['Tmax']) / len(dataDJF['Tmax'])
P_T2 = len(filterT2['Tmax']) / len(dataDJF['Tmax'])
P_T3 = len(filterT3['Tmax']) / len(dataDJF['Tmax'])
P_T4 = len(filterT4['Tmax']) / len(dataDJF['Tmax'])

P_pcp_given_T1 = len(filterT1.where(filterT1['PCP']>0,drop = True)['PCP']) / len(filterT1['Tmax'])
P_pcp_given_T2 = len(filterT2.where(filterT2['PCP']>0,drop = True)['PCP']) / len(filterT2['Tmax'])
P_pcp_given_T3 = len(filterT3.where(filterT3['PCP']>0,drop = True)['PCP']) / len(filterT3['Tmax'])
P_pcp_given_T4 = len(filterT4.where(filterT4['PCP']>0,drop = True)['PCP']) / len(filterT4['Tmax'])

P_pcp_1 = (P_pcp_given_T1*P_T1) + (P_pcp_given_T2*P_T2) + (P_pcp_given_T3*P_T3) + (P_pcp_given_T4*P_T4)
P_pcp_2 = len(dataDJF['PCP'].where(dataDJF['PCP']>0,drop = True))/len(dataDJF['PCP'])

P_T4_given_PCP = len(filterPCP.where(filterPCP['Tmax']>50,drop = True)['Tmax']) / len(filterPCP['PCP'])



#Schematic Plot
test = basicStats(data['Tmax'])


"""
figure1, ax1 = plt.subplots()
ax1.set_title('MY Data')

temps = [dataDJF['Tmax'],dataMAM['Tmax'],dataJJA['Tmax'],dataSON['Tmax']]
mins = [dataDJF['Tmin'],dataMAM['Tmin'],dataJJA['Tmin'],dataSON['Tmin']]
precips = [dataDJF['PCP'],dataMAM['PCP'],dataJJA['PCP'],dataSON['PCP']]
ax1.boxplot(temps,positions=[1,5,9,13])
ax1.boxplot(mins,positions=[2,6,10,14])
ax1.boxplot(precips,positions=[3,7,11,15])
"""

"""
#PDF Loop
fig=plt.figure()
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
ax3 = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
ax4 = plt.subplot2grid(shape=(2,6), loc=(1,1), colspan=2)
ax5 = plt.subplot2grid(shape=(2,6), loc=(1,3), colspan=2)

ax1.set_title('h=0.5')
ax2.set_title('h=1')
ax3.set_title('h=2')
ax4.set_title('h=5')
ax5.set_title('h=10')

numbins = 20
histogram.gethist(data['Tmax'],ax1,density='True',bins=numbins, edgecolor='k')
histogram.gethist(data['Tmax'],ax2,density='True',bins=numbins, edgecolor='k')
histogram.gethist(data['Tmax'],ax3,density='True',bins=numbins, edgecolor='k')
histogram.gethist(data['Tmax'],ax4,density='True',bins=numbins, edgecolor='k')
histogram.gethist(data['Tmax'],ax5,density='True',bins=numbins, edgecolor='k')

histogram.getKDF(data['Tmax'],ax1,kernel='gaussian',bandwidth=.5)
histogram.getKDF(data['Tmax'],ax2,kernel='gaussian',bandwidth=1.0)
histogram.getKDF(data['Tmax'],ax3,kernel='gaussian',bandwidth=2.0)
histogram.getKDF(data['Tmax'],ax4,kernel='gaussian',bandwidth=5.0)
histogram.getKDF(data['Tmax'],ax5,kernel='gaussian',bandwidth=10.)

fig.suptitle('Kernal Density Functions')
fig.text(0.53,0.00,'Temperature',ha='center')
fig.text(0.00,0.50,'Frequency Density',va='center',rotation='vertical')

plt.tight_layout()
    
"""
