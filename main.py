""" 
This is a runner script to complete the HW-1 assignment. 
It imports a .xls file into a Pandas dataframe and then 
converts it to an Xarray object. It then complets 
calculations and plots as necessary"""

#importing libraries
import matplotlib.pyplot as plt
import pandas
#importing modules (see statCalcs.py and statPlots.py for code)
import statCalcs
import statPlots

def getRange(input,min,max):
    """ This function gets a range values"""
    value = input.where(input > min,drop=True)
    value = value.where(value < max,drop=True)
    return value

def getConditionalProb(data,variable1,variable2,limit,cuttoff):
    """Calculates P(A|B) from variables in an Xarray object 
    cutoff is uppper and lower   """ 
    if cuttoff == 'l':
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

#Converting PCP values <=0.1 to 0
data['PCP'] = data['PCP'].where(data['PCP']>.01,0)

""" This section performs calculations that correspond to 
    Question 1 of the assignment"""
#Splitting data into seasons
dataSeasonal = data.groupby('Date.season')
dataDJF = data.where(data.Date.dt.season.isin(['DJF']),drop=True)
dataMAM = data.where(data.Date.dt.season.isin(['MAM']),drop=True)
dataJJA = data.where(data.Date.dt.season.isin(['JJA']),drop=True)
dataSON = data.where(data.Date.dt.season.isin(['SON']),drop=True)


#filtering based on T1,T2,T3,T4, & PCP days
djf_T1 = dataDJF.where((dataDJF['Tmax']<30),drop=True)
djf_T2 = dataDJF.where((dataDJF['Tmax']>29) & 
                         (dataDJF['Tmax']<40),drop=True)
djf_T3 = dataDJF.where((dataDJF['Tmax']>39) & 
                         (dataDJF['Tmax']<51),drop=True)
djf_T4 = dataDJF.where((dataDJF['Tmax']>50),drop=True)
djf_PCPDays = dataDJF.where((dataDJF['PCP']>0),drop=True)


#Calculating probability of T1,T2,T3,T4
P_T1 = len(djf_T1['Tmax']) / len(dataDJF['Tmax'])
P_T2 = len(djf_T2['Tmax']) / len(dataDJF['Tmax'])
P_T3 = len(djf_T3['Tmax']) / len(dataDJF['Tmax'])
P_T4 = len(djf_T4['Tmax']) / len(dataDJF['Tmax'])

#Calculating Conditional Probabilities
P_pcp_given_T1 = len(djf_T1.where(djf_T1['PCP']>0,drop = True)['PCP']) / len(djf_T1['Tmax'])
P_pcp_given_T2 = len(djf_T2.where(djf_T2['PCP']>0,drop = True)['PCP']) / len(djf_T2['Tmax'])
P_pcp_given_T3 = len(djf_T3.where(djf_T3['PCP']>0,drop = True)['PCP']) / len(djf_T3['Tmax'])
P_pcp_given_T4 = len(djf_T4.where(djf_T4['PCP']>0,drop = True)['PCP']) / len(djf_T4['Tmax'])

#Calculating probability of PCP via two methods as a check on one another
P_pcp_1 = (P_pcp_given_T1*P_T1) + (P_pcp_given_T2*P_T2) + (P_pcp_given_T3*P_T3) + (P_pcp_given_T4*P_T4)
P_pcp_2 = len(dataDJF['PCP'].where(dataDJF['PCP']>0,drop = True))/len(dataDJF['PCP'])

P_T4_given_PCP = len(djf_PCPDays.where(djf_PCPDays['Tmax']>50,drop = True)['Tmax'])\
                /len(djf_PCPDays['PCP'])


""" This section performs calculations that correspond to 
    Question 2 of the assignment"""
#Collecting Summary Statistics for all precipitating days in record
#See statCalcs.py for code
pcpDays = data.where((data['PCP']>0),drop=True)
summaryStatisticsTmax = statCalcs.basicStats(pcpDays['Tmax'])
summaryStatisticsTmin = statCalcs.basicStats(pcpDays['Tmin'])
summaryStatisticsPCP = statCalcs.basicStats(pcpDays['PCP'])

#Grouping precipitating days by season
djf_PCPDays = pcpDays.where(pcpDays.Date.dt.season.isin(['DJF']),drop=True)
mam_PCPDays = pcpDays.where(pcpDays.Date.dt.season.isin(['MAM']),drop=True)
jja_PCPDays = pcpDays.where(pcpDays.Date.dt.season.isin(['JJA']),drop=True)
son_PCPDays = pcpDays.where(pcpDays.Date.dt.season.isin(['SON']),drop=True)

#Plotting Schematic Plots
figure1, ax1 = plt.subplots()
Tmax_list = [djf_PCPDays['Tmax'],mam_PCPDays['Tmax'],jja_PCPDays['Tmax'],son_PCPDays['Tmax']]
Tmin_list = [djf_PCPDays['Tmin'],mam_PCPDays['Tmin'],jja_PCPDays['Tmin'],son_PCPDays['Tmin']]
PCP_list =  [djf_PCPDays['PCP'],mam_PCPDays['PCP'],jja_PCPDays['PCP'],son_PCPDays['PCP']]
tmaxConfig = dict(color='r',)
tminConfig = dict(color='b')
pcpConfig = dict(color='g')

#Tmax Schematic Plot
box1 = ax1.boxplot(Tmax_list,
                   positions=[1,4,7,10],
                   widths=0.5,
                   boxprops=tmaxConfig)
#Tmin Schematic Plot
box2 = ax1.boxplot(Tmin_list,
                   positions=[1.55,4.55,7.55,10.55],
                   widths=0.5,
                   boxprops=tminConfig)
#PCP Schematic Plot
ax2 = ax1.twinx()
box3 = ax2.boxplot(PCP_list,
                   positions=[2.05,5.05,8.05,11.05],
                   boxprops=pcpConfig)

ax2.set_xticks([1.5,4.5,7.5,10.5])
ax2.set_xticklabels(['DJF','MAM','JJA','SON'])
ax2.set_ylabel('Precip (inches/day)')
ax2.set_title('Daily Temperature and Precipitation')

ax1.set_ylabel('Temperature ($^\circ$F)')
ax1.legend([box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],
           ['Tmax','Tmin','PCP'])

#Saving figure
figure1.savefig('HW1_SchematicPlot.png',
                bbox_inches='tight',
                dpi=200)

"""
This section creates a figure containing 5 histograms with KDFs overlayed
This section callls the module statPlot.py to create the histograms 
and the KDFs
"""

fig=plt.figure()
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
ax3 = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
ax4 = plt.subplot2grid(shape=(2,6), loc=(1,1), colspan=2)
ax5 = plt.subplot2grid(shape=(2,6), loc=(1,3), colspan=2)

ax1.set_title('w/h=0.5')
ax2.set_title('w/h=1')
ax3.set_title('w/h=2')
ax4.set_title('w/h=5')
ax5.set_title('w/h=10')

pltrng = max(data['Tmax'])-min(data['Tmax'])
statPlots.gethist(data['Tmax'],ax1,density='True',bins=int(pltrng//.5), edgecolor='k')
statPlots.gethist(data['Tmax'],ax2,density='True',bins=int(pltrng//1), edgecolor='k')
statPlots.gethist(data['Tmax'],ax3,density='True',bins=int(pltrng//2), edgecolor='k')
statPlots.gethist(data['Tmax'],ax4,density='True',bins=int(pltrng//5), edgecolor='k')
statPlots.gethist(data['Tmax'],ax5,density='True',bins=int(pltrng//10),edgecolor='k')

statPlots.getKDF(data['Tmax'],ax1,kernel='gaussian',bandwidth=0.5)
statPlots.getKDF(data['Tmax'],ax2,kernel='gaussian',bandwidth=1.0)
statPlots.getKDF(data['Tmax'],ax3,kernel='gaussian',bandwidth=2.0)
statPlots.getKDF(data['Tmax'],ax4,kernel='gaussian',bandwidth=5.0)
statPlots.getKDF(data['Tmax'],ax5,kernel='gaussian',bandwidth=10.)
ax5.set_ylim(0,.045)
ax2.set_ylim(0,.025)
ax3.set_ylim(0,.025)
ax4.set_ylim(0,.025)
ax5.set_ylim(0,.025)

fig.suptitle('Tmax Histograms and KDFs')
fig.text(0.53,0.00,'Temperature ($^\circ$F)',ha='center')
fig.text(0.00,0.50,'Frequency Density',va='center',rotation='vertical')

plt.tight_layout()
fig.savefig('HW1_KDF_Plot.png',
            bbox_inches='tight',
            dpi=250)

