#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to calculate the answers for question 1. 
"""
#%% Importing Libraries and Ingesting/Formatting Data

import matplotlib.pyplot as plt
import numpy as np
import pandas
import statPlots
import statsmodels.api as sm
from scipy.stats import chisquare
from scipy.stats import norm

#Ingest xls file as Pandas Dataframe
filename = './SC_data_1980_2015.xls'
rawData = pandas.read_excel(filename)
rawData['Date'] = pandas.to_datetime(rawData['Date'],format = '%Y%m%d')
rawData = rawData.set_index(['Date'])
data = rawData[['Tmax','Tmin','PCP']].to_xarray()
data['PCP'] = data['PCP'].where(data['PCP']>.01) #Convert PCP <= 0.1 to NaN

# Creating daily, 5-day, and monthly precip
PCPDaily = data['PCP'].dropna(dim='Date')
PCP5day = data['PCP'].resample(Date='5D',keep_attrs='True').mean(skipna='True')
PCPMonthly = data['PCP'].resample(Date='1M',keep_attrs='True').mean(skipna='True')

x = np.linspace(0, 1, 100) #for plotting gaussian PDF
muDaily = PCPDaily.mean(skipna='True')
stdDaily = PCPDaily.std(skipna='True') 
mu5day = PCP5day.mean(skipna='True')
std5day = PCP5day.std(skipna='True')
muMonthly = PCPMonthly.mean(skipna='True')
stdMonthly = PCPMonthly.std(skipna='True')

#%% Question 1A)
#muDaily, stdDaily = norm.fit(PCPDaily.dropna(dim='Date'))
xDaily = np.linspace(0, 1, 100)
pDaily = norm.pdf(xDaily, muDaily, stdDaily)


#mu5day, std5day= norm.fit(PCP5day.dropna(dim='Date'))
x5day = np.linspace(0, 1, 100)
p5day = norm.pdf(x5day, mu5day, std5day)

#muMonthly, stdMonthly = norm.fit(PCPMonthly.dropna(dim='Date'))
xMonthly = np.linspace(0, 1, 100)
pMonthly = norm.pdf(xMonthly, muMonthly, stdMonthly)


#
binwidth=.1
fig, ax = plt.subplots(1,3,sharex=True)
statPlots.gethist(PCPDaily,ax=ax[0],range=[0,1],edgecolor='k',
                  bins=np.arange(0, 1.1,binwidth))
statPlots.gethist(PCP5day,ax=ax[1],range=[0,1],edgecolor='k',
                  bins=np.arange(0, 1.1,binwidth))
statPlots.gethist(PCPMonthly,ax=ax[2],range=[0,1],edgecolor='k',
                  bins=np.arange(0, 1.1,binwidth))


ax[0].plot(xDaily, pDaily*(len(PCPDaily)*.1), 'r', linewidth=2)
ax[1].plot(x5day,  p5day*(len(PCP5day)*.1), 'r', linewidth=2)
ax[2].plot(xMonthly, pMonthly*(len(PCPMonthly)*.1), 'r', linewidth=2)

ax[0].set_title('mu={:.2f}\nstd={:.2f}'.format(muDaily.values,stdDaily.values))
ax[1].set_title('mu={:.2f}\nstd={:.2f}'.format(mu5day.values,std5day.values))
ax[2].set_title('mu={:.2f}\nstd={:.2f}'.format(muMonthly.values,stdMonthly.values))
ax[1].set_xlabel('PCP mm/day')
ax[0].set_ylabel('Count')
fig.suptitle('Daily, 5-day, and Monthly Precip')
plt.tight_layout()



PCPDailyStd = (PCPDaily-PCPDaily.mean(dim='Date',skipna='True'))*\
              (1/float(PCPDaily.std(dim='Date',skipna='True').values))

PCP5dayStd = (PCP5day-PCP5day.mean(dim='Date',skipna='True'))*\
              (1/float(PCP5day.std(dim='Date',skipna='True').values))

PCPMonthlyStd = (PCPMonthly-PCPMonthly.mean(dim='Date',skipna='True'))*\
              (1/float(PCPMonthly.std(dim='Date',skipna='True').values))
              
fig2, ax = plt.subplots(1,3)
sm.qqplot(PCPDaily,line='45',ax=ax[0])
sm.qqplot(PCP5dayStd,line='45',ax=ax[1])
sm.qqplot(PCPMonthlyStd,line='45',ax=ax[2])

ax[0].set_title('Daily Precip')
ax[1].set_title('5-Day Precip')
ax[2].set_title('Monthly Precip')
plt.tight_layout()


#%% Question 3
""" This code tests the Goodness of fit for the Daily, 5-day, and monthly
    precip datasets to a gaussian fit. This code creates bins based on 
    0-20, 20-40, 40-60, 60-80, and 80-100 percentile of each dataset. It 
    then calcualtes the Z score at each bin edge. The Z score is used to find
    the area under a standard normal curve. The area*count of each bin is the 
    expectency. The counts are then converted to frequencies. The frequencies 
    and counts are then used as parameters for the chisquare function where it
    returns a multi-dimensional list containing Chi2 values and P values. 
    If P value is less than alpha (.05 in this case) then we reject
    the null and conclude that the distribution is NOT normal. 
"""    

percentiles = [0,20,40,60,80,100]
data = [PCPDaily[~np.isnan(PCPDaily)],
        PCP5day[~np.isnan(PCP5day)],
        PCPMonthly[~np.isnan(PCPMonthly)]]

Chi2 = []
for i in np.arange(0,len(data)):
    percntVals = np.percentile(data[i],percentiles)
    Counts,rightedgs = np.histogram(data[i],bins=percntVals)
    Z = (percntVals-np.nanmean(data[i]))/np.nanstd(data[i])
    areas =Z[1:]-Z[:-1]
    expectency = np.multiply(Counts,areas)
    FreqDaily = np.ndarray.tolist(np.divide(Counts,sum(Counts)))
    Chi2.append(chisquare(FreqDaily,f_exp=expectency))
