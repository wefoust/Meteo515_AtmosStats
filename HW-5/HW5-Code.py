# -*- coding: utf-8 -*-
"""
HW 5
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from netCDF4 import Dataset
from sklearn import preprocessing
from sklearn.decomposition import PCA

filename1 = 'sine_wave_data1.nc'
filename2 = '2D_bulls_eyes.nc'
a = Dataset(filename1,mode='r')
b = Dataset(filename2,mode='r')
dataset1 = xr.open_dataset(xr.backends.NetCDF4DataStore(a))
dataset2 = xr.open_dataset(xr.backends.NetCDF4DataStore(b))
sinData = dataset1['data'].T
bullseyeData = dataset2['data']

#%% Question 1
#Preprocessing Data
scalar = preprocessing.StandardScaler()           # Scaling Object
standardizedData = scalar.fit_transform(sinData)  # Fit calcs mean & std. transform standardizes
sinwavePCA = PCA()                                # Create PCA object
sinwavePCA.fit(standardizedData)                  # Do PCA and get EOFs as object

# Calculating PCs
PCs = sinwavePCA.transform(standardizedData)                        # Return PCs
stdPC1 = (PCs[:,0]-np.mean(PCs[:,0]))/np.std(PCs[:,0])              # Get  PC1
stdPC2 = (PCs[:,1]-np.mean(PCs[:,1]))/np.std(PCs[:,1])              # Get  PC2
varExplained = np.round(sinwavePCA.explained_variance_ratio_*100,1) # Get variance explained %
cumvarExplained = np.cumsum(varExplained)                           # Get cummulative var explained    
eigenvalues = sinwavePCA.explained_variance_                        # Get eigenvalues

#Calculate EOFs
EOFs = sinwavePCA.components_              # Get EOFs
EOF1 = EOFs[0,:]*np.sqrt(eigenvalues[0])   # Get EOF 1 & scale it
EOF2 = EOFs[1,:]*np.sqrt(eigenvalues[1])   # Get EOF 2 & scale it

EOF1_reg = stdPC1 @ standardizedData                       # Get EOF 1 by regressing PC on data
EOF1_reg = (EOF1_reg - np.mean(EOF1_reg))/np.std(EOF1_reg) # Standardize EOF1
EOF2_reg = stdPC2 @ standardizedData                       # Get EOF 2 by regressing PC on data
EOF2_reg = (EOF2_reg - np.mean(EOF2_reg))/np.std(EOF2_reg) # Standardize EOF 2


#Plotting
fig,ax = plt.subplots(2,2)
ax[0,0].plot(stdPC1,label='PC1 ({}%)'.format(varExplained[0]))
ax[0,0].plot(stdPC2,label='PC2 ({}%)'.format(varExplained[1]))
ax[0,0].set_title('PC Time Series')
ax[0,0].set_xlabel('time')
ax[0,0].legend(loc='upper right')

ax[0,1].scatter(np.arange(1,11),varExplained[0:10],color='r')
ax[0,1].plot(np.arange(1,11),varExplained[0:10])
ax[0,1].set_xlabel('PC')
ax[0,1].set_title('Variance Explained (%)')

ax[1,0].plot(EOF1, label='EOF1')
ax[1,0].plot(EOF2, label='EOF2')
ax[1,0].set_title('EOFs by Eigenvector')
ax[1,0].legend(loc='upper right')

ax[1,1].plot(EOF1_reg, label='EOF1')
ax[1,1].plot(EOF2_reg, label='EOF2')
ax[1,1].set_title('EOFs by PC Regressed Onto Data')
plt.tight_layout()
fig.savefig('./figures/HW5_PCA_Sinwave.png',
        bbox_inches='tight',
        dpi=250) 

#%% Question 2
DATA = np.reshape(bullseyeData.T.values,(bullseyeData.shape[2], bullseyeData.shape[0] * bullseyeData.shape[1]), order='F')
scalar = preprocessing.StandardScaler()
stdDATA = scalar.fit_transform(DATA) #fit cacls mean & std, transform applies the standardization
patternPCA = PCA()
patternPCA.fit(stdDATA)

#Calc PCs
PCs = patternPCA.transform(stdDATA)
stdPC1 = (PCs[:,0]-np.mean(PCs[:,0]))/np.std(PCs[:,0])
stdPC2 = (PCs[:,1]-np.mean(PCs[:,1]))/np.std(PCs[:,1])
varExplained = np.round(patternPCA.explained_variance_ratio_*100,1)
cumvarExplained = np.cumsum(varExplained)
eigenvalues = patternPCA.explained_variance_

#Calulate EOFs

scalar = preprocessing.StandardScaler()
stdDATA = scalar.fit_transform(DATA.T) #fit cacls mean & std, transform applies the standardization
patternPCA = PCA()
patternPCA.fit(stdDATA.T)
EOFs = patternPCA.components_
EOF1 = np.reshape(EOFs[0,:],(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF1 = EOF1*np.sqrt(eigenvalues[0])
EOF2 = np.reshape(EOFs[1,:],(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF2 = EOF2*np.sqrt(eigenvalues[1])

EOF1_reg = stdPC1 @ stdDATA.T
EOF1_reg = (EOF1_reg - np.mean(EOF1_reg))/np.std(EOF1_reg)
EOF1_reg = np.reshape(EOF1_reg,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')

EOF2_reg = stdPC2 @ stdDATA.T
EOF2_reg = np.reshape(EOF2_reg,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF2_reg = (EOF2_reg - np.mean(EOF2_reg))/np.std(EOF2_reg)

#Plotting
fig,ax = plt.subplots(3,2)
plotmap = ax[0,0].plot(stdPC1,label='PC1 ({}%)'.format(varExplained[0]))
plotmap = ax[0,0].plot(stdPC2,label='PC2 ({}%)'.format(varExplained[1]))
ax[0,0].set_title('PC Time Series')
#ax[0,0].set_xlabel('time')
ax[0,0].legend(loc='upper right')

plotmap = ax[0,1].scatter(np.arange(1,11),varExplained[0:10],color='r')
plotmap = ax[0,1].plot(np.arange(1,11),varExplained[0:10])
#ax[0,1].set_xlabel('PC')
ax[0,1].set_title('Variance Explained (%)')

plotmap = ax[1,0].contourf(EOF1,cmap='bwr')
ax[1,0].set_title('EOF1 by Eigenvectors')

plotmap = ax[1,1].contourf(EOF1_reg,cmap='bwr')
ax[1,1].set_title('EOF1 by Regression')

plotmap = ax[2,0].contourf(EOF2,cmap='bwr')
ax[2,0].set_title('EOF2 by Eigenvectors')

plotmap = ax[2,1].contourf(-EOF2_reg,cmap='bwr')
ax[2,1].set_title('EOF2 by Regression')


fig.subplots_adjust(top=1)
axCbar = fig.add_axes([0.33, .0, 0.4, 0.03])
fig.colorbar(plotmap, cax=axCbar,orientation="horizontal")

plt.tight_layout()

fig.savefig('./figures/HW5_PCA_ToyPattern.png',
        bbox_inches='tight',
        dpi=250) 

