#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:54:33 2021

@author: Danilo
"""
import statistics_Danilo as st
from prepare_data import (regrid, GetPrecData)
from scipy import stats
from scipy.stats import mannwhitneyu
#
import numpy as np
import csv
# plotting packages
import pylab as pl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
# ----------
def GetTotalAcc(event):
    olam,obs = GetPrecData(event)[0],GetPrecData(event)[1]           
#    obs = regrid(olam.lon,olam.lat,obs,event)
#    obs = obs.cumsum('time')     
#    if event < 3:
#        obs = obs[-48] 
#    else:
#        obs = obs[-9]
    
    obs_acc = obs[0]*0    
    if event < 3:    
        for t in obs.time[:-48]:
            obs_acc = obs_acc + obs.sel(time=t)        
    else:    
        for t in obs.time[:-9]:
            obs_acc = obs_acc + obs.sel(time=t) 
        
    olam = regrid(obs.lon,obs.lat,olam,event)        
    olam = olam[-1]
    
    return obs_acc, olam
# ----------
# params for plotting
# --
# bounding box
min_lon, max_lon, min_lat, max_lat = -54, -45, -34, -26
# --
# Make state boundaries feature
states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
# Make country borders feature
country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_countries',    
                                               scale='50m', facecolor='none')       
# --
def plot_background(ax):
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax

def plot_accprec_panel():
    
    # create colormap
    col_hcl = [
     [0.9921568627450981, 0.6588235294117647, 0.7058823529411765],       
     [0.9294117647058824, 0.4392156862745098, 0.6627450980392157],       
     [0.8, 0.16470588235294117, 0.6470588235294118], 
     [0.5294117647058824, 0.058823529411764705, 0.5254901960784314],  
     [0.36470588235294116,0.1568627450980392, 0.39215686274509803],  
     [0.3215686274509804, 0.2549019607843137, 0.4549019607843137],      
     [0.1843137254901961, 0.4627450980392157, 0.5725490196078431], 
     [0.0, 0.5843137254901961, 0.6862745098039216],
     [0.09411764705882353, 0.7411764705882353, 0.6901960784313725],
     [0.9450980392156862, 0.9450980392156862, 0.9450980392156862]
     ]   
    col_hcl.reverse()
    cmap = LinearSegmentedColormap.from_list(
        'MyMap', col_hcl, N=20)
    cmap.set_under('white')
    # figure params
    fig = plt.figure(figsize=(10,15) , constrained_layout=False)
    gs1 = gridspec.GridSpec(6, 2, hspace=0.25, wspace=0.15, left=0.01, right=0.45)
    gs2 = gridspec.GridSpec(6, 2, hspace=0.25, wspace=0.15, left=0.50, right=0.94)
    axs = []
    datacrs = ccrs.PlateCarree()
    #
    ev = 1
    panel1 = 0
    panel2 = 0
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range (1,25):
        # get data
        if i % 2 != 0:
            tmp = GetTotalAcc(ev)
            obs = tmp[0].transpose('lat','lon')
            olam = tmp[1].transpose('lat','lon')
            if np.max(obs.values) > np.max(olam.values):
#                max_ = float(np.mean(obs).values + np.std(obs).values)
                max_ = float(np.amax(obs).values)
            else:
#                max_ = float(np.mean(olam).values + np.std(olam).values)
                max_ = float(np.amax(olam).values)
            clevs_prec = np.arange(1, round(max_,-1), round(max_,-1)/10)
        # figure
        if ev % 2 != 0:
            panel1 += 1
            axs.append(fig.add_subplot(gs1[panel1 - 1], projection=datacrs))
        if ev % 2 == 0:
            panel2 += 1
            axs.append(fig.add_subplot(gs2[panel2 - 1], projection=datacrs))
        ax1 = axs[-1]
        axs.append(ax1)         
        # reanalysis data
        lons, lats = obs.lon, obs.lat
        if i % 2 != 0: 
            ax1.contourf(lons, lats, obs, clevs_prec, vmin=1,
                               cmap=cmap,extend= 'max') 
            ax1.contour(lons, lats, obs, clevs_prec,colors='grey', linewidths=1) 
            ax1.text(-53,-27.5,str(ev), fontsize = 18, bbox=props)
            if ev < 3:
                ax1.text(-51,-25.8,'OBS.', fontsize=16)
        else:
            cf = ax1.contourf(lons, lats, olam, clevs_prec, vmin=1,
                               cmap=cmap,extend= 'max') 
            ax1.contour(lons, lats, olam, clevs_prec,colors='grey', linewidths=1) 
            # colorbar
            pos = ax1.get_position()
            cbar_ax = fig.add_axes([pos.x1+0.01, pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(cf, cax=cbar_ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=12)
            for label in cbar.ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            if ev < 3:
                ax1.text(-51,-25.8,'OLAM.', fontsize=16) 
            ev += 1
        # map cosmedics 
        plot_background(ax1)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = False
        if i % 2 == 0: 
            gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator(range(-54,-40,2))
        gl.xlabel_style = {'size': 14, 'color': 'gray'}
        gl.ylabel_style = {'size': 14, 'color': 'gray'}
        ax1.outline_patch.set_edgecolor('gray')
            
    pl.savefig('./figures/accprec/accprec_panel.jpg', format='jpg')
    pl.savefig('./figures/accprec/accprec_panel.eps', format='eps', dpi=300)
        
    
def obs_model_panel():
    
        # create colormap
    col_hcl = [
     [0.9921568627450981, 0.6588235294117647, 0.7058823529411765],       
     [0.9294117647058824, 0.4392156862745098, 0.6627450980392157],       
     [0.8, 0.16470588235294117, 0.6470588235294118], 
     [0.5294117647058824, 0.058823529411764705, 0.5254901960784314],  
     [0.36470588235294116,0.1568627450980392, 0.39215686274509803],  
     [0.3215686274509804, 0.2549019607843137, 0.4549019607843137],      
     [0.1843137254901961, 0.4627450980392157, 0.5725490196078431], 
     [0.0, 0.5843137254901961, 0.6862745098039216],
     [0.09411764705882353, 0.7411764705882353, 0.6901960784313725],
     ]   
    col_hcl.reverse()
    cmap = LinearSegmentedColormap.from_list(
        'MyMap', col_hcl, N=20)
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.2, wspace=0.2)
    axs = []
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]           
        tmp = GetTotalAcc(i)
        obs, olam = tmp[0].transpose('lat','lon'),tmp[1].transpose('lat','lon')
        B0, B1, reg_line = st.linear_regression(obs.values, olam.values)
        R = st.Scorr(obs, olam)[0]        
        text = ''' R^2: {}
        y = {} + {}X'''.format(round(R**2, 2),
                               round(B0, 2),
                               round(B1, 2))
        gradient, intercept, r_value, p_value, std_err = stats.linregress(olam.values.ravel(),obs.values.ravel())            
        max_ = np.max([np.amax(obs.values),np.amax(olam.values)])            
        min_ = np.min([np.amin(obs.values),np.amin(olam.values)])
        x1=np.linspace(min_,max_,500)
        y1=gradient*x1+intercept 
        ax1.scatter(olam,obs,c=obs,cmap=cmap)
#        ax1.plot(obs, B0 + B1*obs, c = 'r', linewidth=5, alpha=.5, solid_capstyle='round')
        ax1.loglog(x1,y1,"k")
        if i > 9:        
            ax1.set_xlabel('Olam', fontsize = 18)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Reanalysis',fontsize = 18)
        ax1.text(0.1,0.85, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        ax1.text(0.3,0.1, s=text, fontsize = 12, transform=ax1.transAxes, bbox=props)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.tick_params(labelsize=14)
        ax1.set_xlim(0,max_)
        ax1.set_ylim(0,max_)
        ax1.set_aspect('equal', 'datalim')
        
    pl.savefig('./figures/accprec/obs_x_olam_panel.jpg', format='jpg')    
    pl.savefig('./figures/accprec/obs_x_olam_panel.eps', format='eps', dpi=300)    

        
def TestMannWithneyU_TotalAccPrec():
    for i in range(1,13):

        obs, olam = GetTotalAcc(i)
        stat, p = mannwhitneyu(obs.values.ravel(), olam.values.ravel())
        alpha = 0.05
        fig, axs = plt.subplots(1,2)
        axs[0].hist(olam.values.ravel(), bins=20, color= 'r')
        plt.suptitle('Event '+str(i))
        axs[1].hist(obs.values.ravel(), bins=20, color = 'b') 
        if p > alpha:
           plt.title('Same distribution (fail to reject H0)')
        else:
            	plt.title('Different distribution (reject H0)')  
    
def histogram():
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.2, wspace=0.2)
    axs = []    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13): 
        obs, olam = GetTotalAcc(i)
        obs1d, olam1d = np.reshape(obs.values,obs.size), np.reshape(olam.values,obs.size) 
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]  
        maxp = np.amax([np.amax(obs1d),np.amax(olam1d)])
        if maxp > 300:
            interval = 50
        elif maxp > 100 and maxp < 300:
            interval = 20
        else:
            interval = 10
        histo = ax1.hist(obs1d, bins=range(0, int(round(maxp,-1)) + interval, interval))
        valueso, binso = histo[0],  histo[1]
        histm =  ax1.hist(olam1d, bins=range(0, int(round(maxp,-1)) + interval, interval))
        valuesm, binsm =  histm[0], histm[1]
        ax1.clear()
        ax1.plot(binso[:-1],valueso, c='#0077b6',linestyle='--', label='Reanalysis', linewidth=4)
        ax1.plot(binsm[:-1],valuesm,  c='#69140E', label='OLAM', linewidth=4)
        ax1.set_yscale('log')
        if maxp > 100:
            ax1.set_xscale('log')
            ax1.set_aspect('equal', 'datalim')
        ax1.tick_params(labelsize=14)
        ax1.set_xlim(0,maxp+(maxp/10))
        ax1.grid(linewidth=0.5, color= 'grey')
        ax1.text(0.85,0.85, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        if i == 1:
            ax1.legend(fontsize=14)
        if i > 9:        
            ax1.set_xlabel('Precipitation (mm)', fontsize = 18)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Num. of grid points',fontsize = 17)
        
        pl.savefig('./figures/accprec/histogram.jpg', format='jpg')    
        pl.savefig('./figures/accprec/histogram.eps', format='eps', dpi=300)    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap  

def export_stats(): 
# export statistics to csv file     
    arr = []
    for event in range(1,13):
        print('------------------------------------------------')
        print('making statistics for event '+str(event)+'...')
        ev = []
        obs, olam = GetTotalAcc(event)
        Ptot_model = float(olam.sum('lon').sum('lat').values)
        Ptot_obs = float(obs.sum('lon').sum('lat').values)
        ev.append(str(event))
        ev.append(round(Ptot_obs))
        ev.append(np.std(obs).values)
        ev.append(round(Ptot_model)) 
        ev.append(np.std(olam).values)        
        ev.append(round(Ptot_obs - Ptot_model))
        ev.append(st.BIAS(obs, olam))
        ev.append(st.MAE(obs, olam))
        ev.append(st.MSE(obs, olam))
        ev.append(st.MSE_diss(obs, olam)) 
        ev.append(st.MSE_disp(obs, olam))
        ev.append(st.Scorr(obs, olam)[0]) 
        ev.append(st.Scorr(obs, olam)[1]) 
        ev.append(st.Scorr(obs, olam)[2])         
        ev.append(st.RMSE(obs, olam))
        ev.append(st.RMSE_bias(obs, olam))
        ev.append(np.std(st.di_acc(obs, olam)).values)
        ev.append(st.S_sqr(obs, olam))
        ev.append(st.oncordanceIndex(obs, olam)) 
        ev.append(st.D_pielke(obs, olam))
        arr.append(ev)
        

    with open('accprec_validation_stats.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['Event', 'Ptot_obs','obs_std','Ptot_model','model_std', 'DeltaPtot', 'Bias',
                         'MAE','MSEtot','MSE_diss','MSE_disp',
                         'Corr_Pearson', 'Corr_Spearmann', 'Corr_Kendall', 
                         'RMSE','RMSE_bias','diff_std','S_sqr','ConcordanceIndex',
                         'D_pielke'])
        writer.writerows(arr)
        
# ----------
#if __name__ == "__main__": 
#        plot_accprec_panel()
#        histogram()
#        obs_model_panel()
#        export_stats()
#                