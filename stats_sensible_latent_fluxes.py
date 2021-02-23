# Souza & Ramos da Silva,

# Ocean-Land Atmosphere Model (OLAM) performance for major extreme
#   meteorological events near the coastal region of southern Brazil,

# Climate Research, in revision 2020
''' 
Created on Sun Feb  14 20:29:00 2021

Compute the mean and standard deviation for 
    sensible and Latent heat flux from MERRA and OLAM model
    
@author: Renato and Danilo

''' 

import statistics
from prepare_data import (FluxData, unacc_olam_prec)

dates = ['2018-01-11T15:00:00.000000000',
            '2017-06-04T15:00:00.000000000',
            '2011-09-08T15:00:00.000000000',
            '2011-05-28T15:00:00.000000000',
            '2011-01-21T15:00:00.000000000',
            '2010-04-10T15:00:00.000000000',
            '2008-11-22T15:00:00.000000000',
            '2008-05-03T15:00:00.000000000',
            '2007-07-28T15:00:00.000000000',
            '2006-09-03T15:00:00.000000000',
            '2005-08-10T15:00:00.000000000',
            '2004-03-28T15:00:00.000000000']


print('-------------------')
print('Sensible Heat Fluxes \n')

shf_olam = []
for i in range(1,13):
    data = FluxData(i).shf/10800
    data = unacc_olam_prec(data)
    data = data.sel(time=dates[i-1])
    shf_olam.append(statistics.mean(data.values.ravel())) 
    
print("Mean of the OLAM SHF is % s " %(statistics.mean(shf_olam))) 
print("Median of the OLAM SHF is % s " %(statistics.median(shf_olam))) 
print("Standard Deviation of the OLAM SHF is % s " %(statistics.stdev(shf_olam)))
print('')

shf_merra = [70, 20, 30, 93, 77, 70, 83, 57, 139, 181, 78, 93]

print("Mean of the MERRA-2 SHF is % s " %(statistics.mean(shf_merra))) 
print("Median of the MERRA-2 SHF is % s " %(statistics.median(shf_merra))) 
print("Standard Deviation of the MERRA-2 SHF is % s " %(statistics.stdev(shf_merra)))
print('')

print('-------------------')
print('Latent Heat Fluxes \n')

lhf_olam = []
for i in range(1,13):
    data = FluxData(i).lhf/10800
    data = unacc_olam_prec(data) 
    data = data.sel(time=dates[i-1])
    lhf_olam.append(statistics.mean(data.values.ravel()))    

print("Mean of the OLAM LHF is % s " %(statistics.mean(lhf_olam))) 
print("Median of the OLAM LHF is % s " %(statistics.median(lhf_olam))) 
print("Standard Deviation of the OLAM LHF is % s " %(statistics.stdev(lhf_olam)))
print('')


lhf_merra = [167, 120, 106, 252, 206, 221, 315, 269, 255, 302, 185, 146]

print("Mean of the MERRA-2 LHF is % s " %(statistics.mean(lhf_merra))) 
print("Median of the MERRA-2 LHF is % s " %(statistics.median(lhf_merra))) 
print("Standard Deviation of the MERRA-2 LHF is % s " %(statistics.stdev(lhf_merra)))
print('')
