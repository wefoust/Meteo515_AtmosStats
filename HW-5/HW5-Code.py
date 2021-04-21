# -*- coding: utf-8 -*-
"""
HW 5 Code
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eofs.standard import Eof
from netCDF4 import Dataset

def cumSUM(data,threshold):
    cumVarExplained = 0
    i = 0
    while cumVarExplained <= threshold:
        i += 1
        cumVarExplained = np.sum(data[0:i])        
    return i

filename1 = 'sine_wave_data1.nc'
filename2 = '2D_bulls_eyes.nc'
a = Dataset(filename1,mode='r')
b = Dataset(filename2,mode='r')
dataset1 = xr.open_dataset(xr.backends.NetCDF4DataStore(a))
dataset2 = xr.open_dataset(xr.backends.NetCDF4DataStore(b))
sinData = dataset1['data'].T
sinData = (sinData - sinData.mean(axis=0))/sinData.std(axis=0)
sinData = sinData.values
bullseyeData = dataset2['data']

#%% EOF analysis
solver = Eof(sinData)
eigenvalues = solver.eigenvalues()               # Get eigenvalues
EOFs = solver.eofs(eofscaling=0)                 # Get EOFs
EOFs_reg = solver.eofsAsCorrelation()            # Get EOFs as correlation b/w PCs &  orig data
PCs = solver.pcs(pcscaling=1)                    # Get PCs

# Get variance explained and # of PCs
VarExplain = np.round(solver.varianceFraction()*100,1)
numPCs2Keep = cumSUM(VarExplain,90)

# Calculate EOFs
EOF1 = EOFs[0,:] * np.sqrt(eigenvalues[0])   # Get EOF 1 & scale it
EOF2 = EOFs[1,:] * np.sqrt(eigenvalues[1])   # Get EOF 2 & scale it
EOF1_reg = EOFs_reg[0,:]
EOF2_reg = EOFs_reg[1,:]
stdPC1 = PCs[:,0]
stdPC2 = PCs[:,1]

# Alt method of getting EOF 1 by regressing PC on data
#EOF1_reg = np.expand_dims(stdPC1,0) @ sinData                                  
#EOF1_reg = (EOF1_reg - np.mean(EOF1_reg))/np.std(EOF1_reg) # Standardize EOF1

# Alt method of getting EOF 1 by regressing PC on data
#EOF2_reg =  np.expand_dims(stdPC2,0) @ sinData                                 
#EOF2_reg = (EOF2_reg - np.mean(EOF2_reg))/np.std(EOF2_reg) # Standardize EOF 2

fig,ax = plt.subplots(2,2)
ax[0,0].plot(stdPC1,label='PC1 ({}%)'.format(VarExplain[0]))
ax[0,0].plot(stdPC2,label='PC2 ({}%)'.format(VarExplain[1]))
ax[0,0].set_title('PC Time Series')
ax[0,0].set_xlabel('time')
ax[0,0].legend(loc='upper right')
ax[0,0].set_ylabel('index')

ax[0,1].scatter(np.arange(1,11),VarExplain[0:10],color='r')
ax[0,1].plot(np.arange(1,11),VarExplain[0:10])
ax[0,1].set_xlabel('PC')
ax[0,1].set_title('Variance Explained (%)')
ax[0,1].grid()
ax[0,1].set_xticks(([1,2,3,4,5,6,7,8,9]))


ax[1,0].plot(EOF1, label='EOF1')
ax[1,0].plot(EOF2, label='EOF2')
ax[1,0].set_title('EOFs by Eigenvector')
ax[1,0].legend(loc='upper right')
ax[1,0].set_xlabel('space')
ax[1,0].set_ylabel('index')


ax[1,1].plot(EOF1_reg, label='EOF1')
ax[1,1].plot(EOF2_reg, label='EOF2')
ax[1,1].set_title('EOFs by PC Regressed Onto Data')
ax[1,1].set_xlabel('space')

plt.tight_layout()

fig.savefig('./figures/HW5_PCA_Sinwave.png',
             bbox_inches='tight',
             dpi=250) 

#%% Question 2
DATA = np.reshape(bullseyeData.T.values,(bullseyeData.shape[2], bullseyeData.shape[0] * bullseyeData.shape[1]), order='F')
solver = Eof(DATA)
eigenvalues = solver.eigenvalues()
EOFs = solver.eofs(eofscaling=2)
EOFs_reg = solver.eofsAsCorrelation() 
PCs = solver.pcs(pcscaling=1)
VarExplain = np.round(solver.varianceFraction()*100,1)
numPCs2Keep = cumSUM(VarExplain,95)

# Reteive Specific EOFs
EOF1 = EOFs[0,:]   # Get EOF 1 & scale it
EOF2 = EOFs[1,:]   # Get EOF 2 & scale it
EOF1_reg = EOFs_reg[0,:]
EOF2_reg = EOFs_reg[1,:]

EOF1 = np.reshape(EOF1,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF2 = np.reshape(EOF2,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF1_reg = np.reshape(EOF1,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')
EOF2_reg = np.reshape(EOF2,(bullseyeData.shape[0],bullseyeData.shape[1]),order='F')

stdPC1 = PCs[:,0]
stdPC2 = PCs[:,1]

# Plotting
fig,ax = plt.subplots(3,2)
plotmap = ax[0,0].plot(stdPC1,label='PC1 ({}%)'.format(VarExplain[0]))
plotmap = ax[0,0].plot(stdPC2,label='PC2 ({}%)'.format(VarExplain[1]))
ax[0,0].set_title('PC Time Series')
#ax[0,0].set_xlabel('time')
ax[0,0].legend(loc='upper right')

plotmap = ax[0,1].scatter(np.arange(1,11),VarExplain[0:10],color='r')
plotmap = ax[0,1].plot(np.arange(1,11),VarExplain[0:10])
#ax[0,1].set_xlabel('PC')
ax[0,1].set_title('Variance Explained (%)')
ax[0,1].grid()
ax[0,1].set_xticks(([1,2,3,4,5,6,7,8,9]))

plotmap = ax[1,0].contourf(EOF1,cmap='bwr')
ax[1,0].set_title('EOF1 by Eigenvectors')

plotmap = ax[1,1].contourf(EOF1_reg,cmap='bwr')
ax[1,1].set_title('EOF1 by Regression')

plotmap = ax[2,0].contourf(EOF2,cmap='bwr')
ax[2,0].set_title('EOF2 by Eigenvectors')

plotmap = ax[2,1].contourf(EOF2_reg,cmap='bwr')
ax[2,1].set_title('EOF2 by Regression')

fig.subplots_adjust(top=1)
axCbar = fig.add_axes([0.33, .0, 0.4, 0.03])
fig.colorbar(plotmap, cax=axCbar,orientation="horizontal")
plt.tight_layout()

fig.savefig('./figures/HW5_PCA_ToyPattern.png',
             bbox_inches='tight',
             dpi=250) 
