#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:32:34 2021

@author: Danilo
"""
import statistics_Danilo as st
from scipy.stats import mannwhitneyu
from prepare_data import (regrid, GetPrecData, unacc_olam_prec)
import numpy as np
import pandas as pd
# plotting packages
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

            
def mean_corrl(event):
    olam,obs = GetPrecData(event)[0],GetPrecData(event)[1]
    if event < 3:
        obs = obs.resample(time='3H').sum(dim='time')
    obs = obs.cumsum('time')
    olam_r = regrid(obs.lon,obs.lat,olam,event)     
    meanobs = []
    meanolam = []
    up_obs = []
    down_obs = []
    up_olam = []
    down_olam = []
    corr = []
    for t in range(len(olam.time)):   
        meanobs.append(np.mean(obs[t]))
        meanolam.append(np.mean(olam[t]))
        up_obs.append(np.amax(obs[t]))
        down_obs.append(np.amin(obs[t]))
        up_olam.append(np.amax(olam[t]))
        down_olam.append(np.amin(olam[t]))
        corr.append(st.Scorr(obs[t], olam_r[t])[0])        
    return (meanobs, meanolam, corr, olam.time,
            up_obs, down_obs, up_olam, down_olam)

def ptot_corrl(event):
    olam,obs = GetPrecData(event)[0],GetPrecData(event)[1]
    if event < 3:
        obs = obs.resample(time='3H').sum(dim='time')
    obs = obs.cumsum('time')
    olam_r = regrid(obs.lon,obs.lat,olam,event)    
    obspt = []
    olampt = []
    corr = []
    for t in range(len(olam.time)):
        
        s_obs = obs[t].sum('lon').sum('lat')
        s_olam = olam[t].sum('lon').sum('lat')       
        obspt.append(s_obs)
        olampt.append(s_olam)
        corr.append(st.Scorr(obs[t], olam_r[t])[0])

    return obspt, olampt, corr, olam.time

def plot_panel(which):
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.1, wspace=0.3)
    axs = []
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]           
        if which == 'pt':
            tmp = ptot_corrl(i)
            ax1.set_yscale('log')
        elif which == 'mean':
            tmp = mean_corrl(i)
            obst, obsb, olabt, olamb = tmp[4],tmp[5],tmp[6],tmp[7]            
        obs, olam, corr, time = tmp[0], tmp[1], tmp[2], tmp[3]                
        ax1.plot(time,obs, linewidth=4,
                        c='#0077b6', label='Reanalysis')
        ax1.plot(time,olam, linewidth=4,
                        c='#69140E',  label='OLAM')   
        ax2 = ax1.twinx()  
        ax2.scatter(time, corr, alpha=0.7,
                        c='#41521F', label='Correlation')                    
        if which == 'mean':
            ax1.fill_between(time,obst,obsb, color='#0077b6',alpha=0.2)                                
            ax1.fill_between(time,olabt,olamb, color='#69140E',alpha=0.2)                                     
#        ax1.tick_params(axis = 'x',rotation=20)
        ax1.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
#        ax1.tick_params(axis = 'x',labelsize=12)  
        plt.xticks([])
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if which == 'mean': 
            loc = 'upper right'
        elif which == 'pt':
            loc = 'lower right'
        if i == 1:
            ax2.legend(lines + lines2, labels + labels2, loc=loc)   
#        lim = ax2.get_ylim()[1]
        ax1.text(0.85,0.85, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        ax1.set_xlim(time[0],time[-1])        
        ax2.set_ylim(0,1)
    pl.savefig('./figures/accprec_time_evo/accprec_'+str(which)+'_time_evo.jpg', format='jpg')
    pl.savefig('./figures/accprec_time_evo/accprec_'+str(which)+'_time_evo.eps', format='eps', dpi=300)


def GetDailyAccPrec(event):
    
    tmp = GetPrecData(event)
    olam,obs = tmp[0],tmp[1]    
    if event > 2:
       dtr = pd.date_range(start=olam.time[0].values, periods=len(obs.time), freq='3h')
       obs = obs.assign_coords(time=dtr)
    olam = unacc_olam_prec(olam)
    olam = olam.resample(time='1D').sum('time')
    obs = obs.resample(time='1D').sum('time')
    if event > 2 and event !=9:
        obs = obs[:-1]

    return obs, olam

def DataToDataFrame(data):
    
    list_ = []
    for t in data.time:
        l = list(data.sel(time=t).values.ravel())
        list_.append(np.reshape(l,len(l)))
    df = pd.DataFrame(list_).transpose()
    
#    dates = data['time'].dt.strftime("%D")
#    df.columns = dates
    df.columns = range(1,len(data.time)+1)
    
    return df 

def TestMannWithneyU_DailyAccPrec():
    MWUT = {}
    for i in range(1,13):
        MWUT['Event '+str(i)] = []
        tmp = GetDailyAccPrec(i)
        obs,olam = tmp[0],tmp[1] 
#            olam_df = DataToDataFrame(olam)
#            obs_df = DataToDataFrame(obs) 
        #daily corrl
        olam_r = regrid(obs.lon,obs.lat,olam,i)
        olam_r, obs_acc = olam_r.cumsum('time'), obs.cumsum('time')
        for d in olam.time:
            stat, p = mannwhitneyu(obs_acc.sel(time=d).values.ravel(), olam_r.sel(time=d).values.ravel())
            alpha = 0.05
            if p > alpha:
                MWUT['Event '+str(i)].append('Same distribution (fail to reject H0)')
            else:
                	MWUT['Event '+str(i)].append('Different distribution (reject H0)')
    
    return MWUT

def PlotBoxPLot():

    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.25, wspace=0.25)
    axs = []
    c1 ='#0077b6'
    c2 = '#ff6961'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
    for i in range(1,13):
        tmp = GetDailyAccPrec(i)
        obs,olam = tmp[0],tmp[1] 
        olam_df = DataToDataFrame(olam)
        obs_df = DataToDataFrame(obs) 
        #daily corrl
        olam_r = regrid(obs.lon,obs.lat,olam,i)
        olam_r, obs_acc = olam_r.cumsum('time'), obs.cumsum('time')
        corrl = []
        for d in olam.time:
            corrl.append(st.Scorr(obs_acc.sel(time=d), olam_r.sel(time=d))[0])
            for ct in range(3):
                corrl.append(np.nan)
        xs = range(len(corrl))
        s1 = pd.Series(corrl, index=xs)                        
        # figure
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]
        axs.append(ax1)
        ax2 = axs[-1].twinx()
        
        ticks = np.arange(0,len(obs_df.columns))
             
        ax1.boxplot(obs_df, labels=obs_df.columns,
        positions=np.array(ticks)*4-1, widths=1.6,
        showfliers=False,
        notch=True, patch_artist=True,
        boxprops=dict(facecolor=c1), 
        medianprops=dict(color='k', linewidth=2),
        flierprops=dict(color='k'))
        
        ax1.boxplot(olam_df, labels=olam_df.columns,
        positions=np.array(ticks)*4+1,widths=1.6,                
        showfliers=False,
        notch=True, patch_artist=True,
        boxprops=dict(facecolor=c2), 
        medianprops=dict(color='k', linewidth=2),
        flierprops=dict(color='k'))
        
        ax2.plot(s1.dropna(),c='#028e2c',marker='v',
                 alpha=0.85,  label='Corrl.', linewidth=2)
        ax2.set_ylim(0,1)
        ax2.grid(linewidth=0.25,c='gray')
        
#        max_ = np.max([np.amax(obs.values),np.amax(olam.values)])            
#        ax1.set_ylim(0,max_)
        
        if i == 1:
            plt.plot([], c=c1, label='Reanalysis')
            plt.plot([], c=c2, label='OLAM')
            plt.legend(fontsize=14,loc='upper right')
        
        plt.xticks(np.arange(0, (len(ticks)) * 4, 4), range(1,len(obs_df.columns)+1))
        
        ax1.text(.05,0.8, str(i), fontsize = 14,
                 transform=ax1.transAxes, bbox=props)
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        plt.xlim(-3, (len(ticks)*4)+1)
        if i > 9:        
            ax1.set_xlabel('Time (days)', fontsize = 16)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Daily Acc. Prec. (mm)',fontsize = 16)
        if i  == 3 or i == 6 or i == 9 or i == 12:
                ax2.set_ylabel('Corrl I.',fontsize = 16)    
    pl.savefig('./figures/accprec_time_evo/daily_boxplot.jpg', format='jpg')
    pl.savefig('./figures/accprec_time_evo/daily_boxplot.eps', format='eps', dpi=300)
      

          
#if __name__ == "__main__":
#    plot_panel('mean')
#    plot_panel('pt')
#    PlotBoxPLot()