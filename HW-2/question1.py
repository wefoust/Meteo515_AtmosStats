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


#%% Question 1C)
numbins = 10
histrange = [-2.5, 2.5]
#Getting counts of histograms and position of bin edges
CountsDaily,rightedgs = np.histogram(PCPDailyStd,bins=numbins,range=histrange)
Counts5day,rightedgs = np.histogram(PCP5dayStd,bins=numbins,range=histrange)
CountsMonthly,rightedgs = np.histogram(PCPMonthlyStd,bins=numbins,range=histrange)

#Converting Counts to Frequencies
FreqDaily = np.ndarray.tolist(np.divide(CountsDaily,sum(CountsDaily)))
Freq5day = np.ndarray.tolist(np.divide(Counts5day,sum(Counts5day)))
FreqMonthly = np.ndarray.tolist(np.divide(CountsMonthly,sum(CountsMonthly)))

#Calculating expectencies. This will be the same for all cases 
#since data is standardized and bin widths and edges are equal.
Expectency = np.zeros(numbins)
for i in np.arange(1,numbins-1,1):
    print(i)
    Expectency[i] = abs(norm.cdf(abs(rightedgs[i])) - norm.cdf(abs(rightedgs[i-1])))
Expectency[0] = norm.cdf(histrange[0]) 
Expectency[numbins-1] = 1 - norm.cdf(histrange[0]) 

#Calulating Chi2 and P values
Chi2Daily = chisquare(FreqDaily,f_exp=Expectency)
Chi25day = chisquare(Freq5day,f_exp=Expectency)
Chi2Monthly = chisquare(FreqMonthly,f_exp=Expectency)
