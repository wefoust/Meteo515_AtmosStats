""" This completes the first assignment"""

import matplotlib.pyplot as plt
import pandas
import numpy as np

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

"""
#Binning Data T1,T2,T3,T4
T1 = dataDJF['Tmax'].where(dataDJF['Tmax'] < 30,drop=True)
T2 = getRange(dataDJF['Tmax'],29,40)
T3 = getRange(dataDJF['Tmax'],39,51)
T4 = dataDJF['Tmax'].where(dataDJF['Tmax'] > 50,drop=True)

#calculating probabilities of temp range in DJF
PT1 = len(T1)/len(dataDJF['Tmax'])
PT2 = len(T2)/len(dataDJF['Tmax'])
PT3 = len(T3)/len(dataDJF['Tmax'])
PT4 = len(T4)/len(dataDJF['Tmax'])

#Calculating Conditional Probabilities
PT1givenPrecip = getConditionalProb(dataDJF,'Tmax','PCP',[30],'l')
PT2givenPrecip = getConditionalProb(dataDJF,'Tmax','PCP',[29,40],'r')
PT3givenPrecip = getConditionalProb(dataDJF,'Tmax','PCP',[39,51],'r')
PT4givenPrecip = getConditionalProb(dataDJF,'Tmax','PCP',[50],'u')
"""
# Bad attemt at conditional probabilities
#T1_full = dataDJF.where(dataDJF['Tmax'] < 30,drop=True)
#T2 = getRange(dataDJF['Tmax'],29,40)
#T3 = getRange(dataDJF['Tmax'],39,51)
#T4_full = dataDJF.where(dataDJF['Tmax'] > 50,drop=True)


"""
#Schematic Plot
dataMAM = data.where(data.Date.dt.season.isin(['MAM']),drop=True)
dataJJA = data.where(data.Date.dt.season.isin(['JJA']),drop=True)
dataSON = data.where(data.Date.dt.season.isin(['SON']),drop=True)

figure1, ax1 = plt.subplots()
ax1.set_title('MY Data')

temps = [dataDJF['Tmax'],dataMAM['Tmax'],dataJJA['Tmax'],dataSON['Tmax']]
mins = [dataDJF['Tmin'],dataMAM['Tmin'],dataJJA['Tmin'],dataSON['Tmin']]
precips = [dataDJF['PCP'],dataMAM['PCP'],dataJJA['PCP'],dataSON['PCP']]
ax1.boxplot(temps,positions=[1,5,9,13])
ax1.boxplot(mins,positions=[2,6,10,14])
ax1.boxplot(precips,positions=[3,7,11,15])
"""