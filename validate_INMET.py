# Souza & Ramos da Silva,

# Ocean-Land Atmosphere Model (OLAM) performance for major extreme
#   meteorological events near the coastal region of southern Brazil,

# Climate Research, in revision 2020
"""
Created on Wed Dec  4 13:57:21 2019

This script will validade OLAM modeled data against INMET station data

@author: Danilo Couto de Souza
"""
import csv
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from scipy import stats

# Create arrays for making comparisons
#  Will help to separate data from each station
si = [83844, 83897, 83948, 83995, 83997] # station index
sn = ['pga', 'fps', 'trs', 'rgd', 'svp'] # station name

# ----------------
# Functions bellow are for getting precipitation or temperature data
#  from OLAM or INMET
def GetINMETPrecData(event):    
    if event < 10:
        with open("Station_data/inmet/precipDaily_inmet_E0"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)         
    else:
        with open("Station_data/inmet/precipDaily_inmet_E"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)                        
    return data

def GetINMETTempData(event):    
    if event < 10:
        with open("Station_data/inmet/inmet_temp_E0"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)         
    else:
        with open("Station_data/inmet/inmet_temp_E"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)                        
    return data


def GetOLAMPrecData(event):    
    if event < 10:
        with open("Station_data/olam/precipDaily_OLAM_E0"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)       
    else:
        with open("Station_data/olam/precipDaily_OLAM_E"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)
    return data

def GetOLAMTempData(event):    
    if event < 10:
        with open("Station_data/olam/olam_temp_E0"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)       
    else:
        with open("Station_data/olam/olam_temp_E"+str(event)+".csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            data = []
            for row in reader: # each 
                        data.append(row)
    return data

# ----------------
def GetEachStationData(event,source,variable):
    '''
    Get and separate data for distinct stations
    
    Parameters
    ----------
    event : integer
        corresponding event number
    
    source : float
        station or olam
        
    variable: float
        prec (precipitation) or temp (temperature)
        
     Returns
    -------
    list of lists
        A list containing data from each station
    '''
    pga, fps, trs, rgd, svp = [], [], [], [], []
    if variable == 'prec' and source == 'station':
        get = GetINMETPrecData
    elif variable == 'temp' and source == 'station':
        get = GetINMETTempData
    if variable == 'prec' and source == 'olam':
        get = GetOLAMPrecData
    elif variable == 'temp' and source == 'olam':
        get = GetOLAMTempData        
    if source == 'station':
        col = 3
    elif source == 'olam':
        col = 1
    tmp = get(event)
    for row in tmp:
        if row[0] == si[0]:
            pga.append(row[col])
        elif row[0] == si[1]:
            fps.append(row[col])
        elif row[0] == si[2]:
            trs.append(row[col])
        elif row[0] == si[3]:
            rgd.append(row[col])
        elif row[0] == si[4]:
            svp.append(row[col])
                
    return pga, fps, trs, rgd, svp

def accINMET(inmet_data):
    '''
    Accumulate precipitation data from INMET
    '''
    tmp = []
    for station in range(len(inmet_data)):
        tmp.append(list(np.cumsum(inmet_data[station])))
    return tmp

def linear_regression(x, y):
    '''
    Perform a linear reggression
    '''
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    B0 = y_mean - (B1*x_mean)
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    return (B0, B1, reg_line)

def PlotObsOlam(variable):
    '''
    Make a plot comparing observed and simulated data,
        including reggression line
    
    Parameters
    ----------
    variable: float
        prec (precipitation) or temp (temperature)
    
    '''
    # Make figure
    fig = plt.figure(constrained_layout=False,figsize=(8,10))
    # Colors for each event
    cols = ['#96ceb4','#ffeead','#ff6f69','#ffcc5c','#88d8b0',
              '#66545e','#a39193','#aa6f73','#eea990','#eea990',
              '#8caba8','#7ddc1f']
    # Text box params         
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # this is for computating corrleation
    data_all_obs = []
    data_all_olam = []
    for event in range(1,13):
        # And this is for separating each event
        #   1) create arrays
        data_obs = []
        data_olam =[]
        #   2) separate data
        obs = GetEachStationData(event,'station',variable)
        olam = GetEachStationData(event,'olam',variable)
        if variable == 'prec':
            obs = accINMET(obs)
        # 3) get only data that matches the same size
        # so we can avoid the missing data form INMET
        for i in range(len(obs)): 
            if len(obs[i]) == len(olam[i]):
                # append the list containing all data from each event
                data_obs.append(obs[i])
                data_olam.append(olam[i])                 
                for j in range(len(obs)):
                    # append each data individuallt
                    data_all_obs.append(obs[i][j])
                    data_all_olam.append(olam[i][j]) 
        # choose color          
        c = cols[event-1]
        # fig
        ax1 = fig.add_subplot(211)
        # plot data
        if event < 10:
            ax1.scatter([],[],label='E0'+str(event))
        else:
            ax1.scatter([],[],label='E'+str(event))
        for station in range(len(data_obs)):
            ax1 = fig.add_subplot(211)
            ax1.scatter(data_olam[station],data_obs[station],color=c)
        # legend
        handles, labels = ax1.get_legend_handles_labels()        
        lgd = ax1.legend(handles, labels,fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')    
    # regression calc
    max_ = np.max([np.amax(data_all_obs),np.amax(data_all_olam)]) 
    min_ = np.min([np.amin(data_all_obs),np.amin(data_all_olam)]) 
    B0, B1, reg_line = linear_regression(data_all_obs, data_all_olam)
    R = stats.spearmanr(data_all_obs, data_all_olam) [0]      
    text = ''' R^2: {} \n y = {} + {}X'''.format(round(R**2, 2),
                           round(B0, 2),
                           round(B1, 2))    
    x = np.ma.masked_where(data_all_obs == 0 , data_all_obs)
    y = np.ma.masked_where(data_all_olam == 0 , data_all_olam)    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(np.log(x),np.log(y))            
    x1=np.linspace(0,max_,10)
    y1=gradient*x1+intercept 
    ax1.loglog(x1,y1,"k")
    # cosmedics       
    plt.grid(linewidth=0.25,color='gray',alpha=0.7)
    ax1.text(0.7,0.1, s=text, fontsize = 12, transform=ax1.transAxes, bbox=props)
    if variable == 'prec':
        plt.xlabel('INMET Acc. Prec. (mm)', fontsize = 16)
        plt.ylabel('OLAM Acc. Prec. (mm)', fontsize = 16)
        plt.xticks([1e0,1e1,1e2]) 
        ax1.set_yscale('log')
        ax1.set_xscale('log') 
    elif variable == 'temp':
        plt.xlabel('INMET Temp (C)', fontsize = 16)
        plt.ylabel('OLAM Temp (C)', fontsize = 16)
        plt.xlim(min_,max_)
        plt.ylim(min_,max_)         
    # Save figure
    pl.savefig('./figures/station/obsxmodel_'+str(variable)+'.eps', bbox_extra_artist=lgd, bbox_inches='tight', format='eps', dpi=300)
    pl.savefig('./figures/station/obsxmodel_'+str(variable)+'.jpg', bbox_extra_artist=lgd, bbox_inches='tight', format='png')
        
if __name__ == "__main__":     
    PlotObsOlam('prec')
    PlotObsOlam('temp')