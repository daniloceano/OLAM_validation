#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:34:55 2021

@author: Danilo
"""
import statistics_Danilo as st
from prepare_data import (regridSLPWind, GetSLPWindData, regrid)
from stats_accprec_time_evo import (mean_corrl, DataToDataFrame)
from scipy import stats

#
import numpy as np
import pandas as pd
import csv
# plotting packages
import pylab as pl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
# ----------
def Get_SLP_Wind_At_EvPeak(event, re):
    olam,obs = GetSLPWindData(event)[0],GetSLPWindData(event)[1]           

    dates = ['2018-01-11T18:00:00.000000000',
            '2017-06-04T21:00:00.000000000',
            '2011-09-08T09:00:00.000000000',
            '2011-05-28T06:00:00.000000000',
            '2011-01-21T21:00:00.000000000',
            '2010-04-10T09:00:00.000000000',
            '2008-11-22T12:00:00.000000000',
            '2008-05-03T21:00:00.000000000',
            '2007-07-28T12:00:00.000000000',
            '2006-09-03T18:00:00.000000000',
            '2005-08-10T09:00:00.000000000',
            '2004-03-28T06:00:00.000000000']
    
    obs = obs.sel(time=dates[event-1])        
    olam = olam.sel(time=dates[event-1]).assign(lev=obs.lev)
        
    if re == 'y':
        olam = regridSLPWind(obs.lon,obs.lat,olam,event)
    
    return obs, olam



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

def plot_slp_panel():
    
    cmap = cmocean.cm.balance
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
            tmp = Get_SLP_Wind_At_EvPeak(ev,'n')
            obs = tmp[0].transpose('lat','lon','lev')
            olam = tmp[1].transpose('lat','lon', 'lev')
            # reanalysis data
            lonsre, latsre = obs.lon, obs.lat
            obs_slp = obs.SLP/100
            ure, vre = obs.U, obs.V
            # olam data
            lonso, latso = olam.lon, olam.lat
            olam_slp = olam.sslp/100
            uo, vo = olam.uwnd, olam.vwnd   
#            clevs_slp = np.arange(990, 1051,2)
            minslp = round(np.amin([int(np.amin(obs_slp.values)), int(np.amin(olam_slp.values))])-5,-1)
            maxslp = round(np.amax([int(np.amax(obs_slp.values)), int(np.amax(olam_slp.values))])+5,-1)
            if minslp > 1014:
                minslp = 1010
            if maxslp < 1025:
                maxslp = 1030            
            norm = colors.DivergingNorm(vmin=minslp, vcenter=1014, vmax=maxslp)
        # figure
        if ev % 2 != 0:
            panel1 += 1
            axs.append(fig.add_subplot(gs1[panel1 - 1], projection=datacrs))
        if ev % 2 == 0:
            panel2 += 1
            axs.append(fig.add_subplot(gs2[panel2 - 1], projection=datacrs))
        ax1 = axs[-1]
        axs.append(ax1)        
        
        if i % 2 != 0: 
            cf1 = ax1.contourf(lonsre, latsre, obs_slp, norm=norm,
                               cmap=cmap) 
            ax1.contour(lonsre, latsre, obs_slp, cf1.levels,colors='grey', linewidths=1) 
            qv = ax1.quiver(lonsre[::2],latsre[::2],
                       ure.sel(lev=1000)[::2,::2],vre.sel(lev=1000)[::2,::2],
                       color= 'k')
            ax1.text(-53,-27.5,str(ev), fontsize = 18, bbox=props)  
            ax1.quiverkey(qv,1.1, 1.07, 20, r'$10 \frac{m}{s}$', labelpos = 'E',
                           coordinates='axes', labelsep = 0.05,
                           fontproperties={'size': 14, 'weight': 'bold'})
            if ev < 3:
                ax1.text(-51,-25.8,'OBS.', fontsize=16)
        else:
            cf2 = ax1.contourf(lonso, latso, olam_slp, norm=norm,
                               cmap=cmap) 
            ax1.contour(lonso, latso, olam_slp, cf2.levels,colors='grey', linewidths=1) 
            ax1.quiver(lonso[::20],latso[::20],uo[::20,::20],vo[::20,::20],
                       color= 'k')
            # colorbar
            pos = ax1.get_position()
            cbar_ax = fig.add_axes([pos.x1+0.005, pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(cf1, cax=cbar_ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=12)
            for label in cbar.ax.xaxis.get_ticklabels()[::1]:
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
        gl.xlabel_style = {'size': 13, 'color': 'gray'}
        gl.ylabel_style = {'size': 14, 'color': 'gray'}
        ax1.outline_patch.set_edgecolor('gray')
            
    pl.savefig('./figures/slp_wind/slp_wind_panel.jpg', format='jpg')
    pl.savefig('./figures/slp_wind/slp_wind_panel.eps', format='eps', dpi=300)
 
    

def plot_wind_panel():
    
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
            tmp = Get_SLP_Wind_At_EvPeak(ev,'n')
            obs = tmp[0].transpose('lat','lon','lev')
            olam = tmp[1].transpose('lat','lon', 'lev')
            # reanalysis data
            lonsre, latsre = obs.lon, obs.lat
            ure, vre = obs.U, obs.V
            wsre = np.sqrt(ure**2 + vre**2)
            wsre = wsre.sel(lev=1000)
            # olam data
            lonso, latso = olam.lon, olam.lat
            uo, vo = olam.uwnd, olam.vwnd   
            wso = np.sqrt(uo**2 + vo**2)
#            clevs_slp = np.arange(990, 1051,2)
            maxws = round(np.amax([int(np.amax(wso.values)), int(np.nanmax(wsre.values))]),-1)           
            norm = colors.Normalize(vmin=0, vmax=maxws)
        # figure
        if ev % 2 != 0:
            panel1 += 1
            axs.append(fig.add_subplot(gs1[panel1 - 1], projection=datacrs))
        if ev % 2 == 0:
            panel2 += 1
            axs.append(fig.add_subplot(gs2[panel2 - 1], projection=datacrs))
        ax1 = axs[-1]
        axs.append(ax1)        
        
        if i % 2 != 0: 
            cf1 = ax1.contourf(lonsre, latsre, wsre, norm=norm,
                               cmap=cmap) 
            ax1.contour(lonsre, latsre, wsre, cf1.levels,colors='grey', linewidths=1) 
            ax1.quiver(lonsre[::2],latsre[::2],
                       ure.sel(lev=1000)[::2,::2],vre.sel(lev=1000)[::2,::2],
                       color= 'k') 
            ax1.text(-53,-27.5,str(ev), fontsize = 18, bbox=props)  
            if ev < 3:
                ax1.text(-51,-25.8,'OBS.', fontsize=16)
        else:
            cf2 = ax1.contourf(lonso, latso, wso, norm=norm,
                               cmap=cmap) 
            ax1.contour(lonso, latso, wso, cf2.levels,colors='grey', linewidths=1) 
            ax1.quiver(lonso[::20],latso[::20],uo[::20,::20],vo[::20,::20],
                       color= 'k')
            # colorbar
            pos = ax1.get_position()
            cbar_ax = fig.add_axes([pos.x1+0.005, pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(cf1, cax=cbar_ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=12)
            for label in cbar.ax.xaxis.get_ticklabels()[::5]:
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
        gl.xlabel_style = {'size': 13, 'color': 'gray'}
        gl.ylabel_style = {'size': 14, 'color': 'gray'}
        ax1.outline_patch.set_edgecolor('gray')
            
    pl.savefig('./figures/slp_wind/windspeed_panel.jpg', format='jpg')
    pl.savefig('./figures/slp_wind/windspeed_panel.eps', format='eps', dpi=300)


def obs_model_panel_slp():
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.2, wspace=0.4)
    axs = []
    cmap = cmocean.cm.balance
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]           
        tmp = Get_SLP_Wind_At_EvPeak(i,'y')
        obs, olam = tmp[0].transpose('lat','lev','lon'),tmp[1].transpose('lat','lev','lon')
        obs, olam = obs.SLP/100, olam.sslp/100
        # regression data
        B0, B1, reg_line = st.linear_regression(obs.values, olam.values)
        R = st.Scorr(obs, olam)[0]        
        text = ''' R^2: {}
        y = {} + {}X'''.format(round(R**2, 2),
                               round(B0, 2),
                               round(B1, 2))
        gradient, intercept, r_value, p_value, std_err = stats.linregress(olam.values.ravel(),obs.values.ravel())            
        minslp = round(np.amin([int(np.amin(obs.values)), int(np.amin(olam.values))])-5,-1)
        maxslp = round(np.amax([int(np.amax(obs.values)), int(np.amax(olam.values))])+5,-1)
        if minslp > 1014:
            minslp = 1010
        if maxslp < 1025:
            maxslp = 1030
        x1=np.linspace(minslp,maxslp,500)
        y1=gradient*x1+intercept             
        norm = colors.DivergingNorm(vmin=minslp, vcenter=1014, vmax=maxslp)            
        ax1.scatter(olam,obs,c=obs,cmap=cmap,norm=norm,alpha=0.85)
        ax1.loglog(x1,y1,"k")
        if i > 9:        
            ax1.set_xlabel('Olam', fontsize = 18)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Reanalysis',fontsize = 18)
        ax1.text(0.1,0.85, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        ax1.text(0.3,0.1, s=text, fontsize = 12, transform=ax1.transAxes, bbox=props)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20)
        ax1.set_xlim(minslp,maxslp)
        ax1.set_ylim(minslp,maxslp)
        ax1.set_aspect('equal', 'datalim')
        
    pl.savefig('./figures/slp_wind/obs_x_olam_slp_panel.jpg', format='jpg')    
    pl.savefig('./figures/slp_wind/obs_x_olam_slp_panel.eps', format='eps', dpi=300)    


def obs_model_panel_wind():
    
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
    
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.2, wspace=0.4)
    axs = []
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]           
        tmp = Get_SLP_Wind_At_EvPeak(i,'y')
        obs, olam = tmp[0].transpose('lat','lev','lon'),tmp[1].transpose('lat','lev','lon')
        obs = np.sqrt(obs.U**2 + obs.V**2).sel(lev=1000)
        olam = np.sqrt(olam.uwnd**2 + olam.vwnd**2)
        # regression data
        mask = ~np.isnan(olam.values) & ~np.isnan(obs.values)
        B0, B1, reg_line = st.linear_regression(obs.values[mask], olam.values[mask])
        R = st.Scorr(obs, olam)[0]        
        text = ''' R^2: {}
        y = {} + {}X'''.format(round(R**2, 2),
                               round(B0, 2),
                               round(B1, 2))
        mask = ~np.isnan(olam.values.ravel()) & ~np.isnan(obs.values.ravel())
        gradient, intercept, r_value, p_value, std_err = stats.linregress(olam.values.ravel()[mask],obs.values.ravel()[mask])         
        minws = round(np.amin([int(np.nanmin(obs.values)), int(np.nanmin(olam.values))])-5,-1)
        maxws = round(np.amax([int(np.nanmax(obs.values)), int(np.nanmax(olam.values))])+5,-1)        
        x1=np.linspace(minws,maxws,500)
        y1=gradient*x1+intercept          
        ax1.scatter(olam,obs,c=obs,cmap=cmap,alpha=0.85)
        ax1.loglog(x1,y1,"k")        
        if i > 9:        
            ax1.set_xlabel('Olam', fontsize = 18)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Reanalysis',fontsize = 18)
        ax1.text(0.1,0.85, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        ax1.text(0.3,0.1, s=text, fontsize = 12, transform=ax1.transAxes, bbox=props)
        ax1.set_xlim(minws,maxws)
        ax1.set_ylim(minws,maxws)
        ax1.set_aspect('equal', 'datalim')
        
    pl.savefig('./figures/slp_wind/obs_x_olam_wind_panel.jpg', format='jpg')    
    pl.savefig('./figures/slp_wind/obs_x_olam_wind_panel.eps', format='eps', dpi=300)    
                
                
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap  


def export_stats_SLP(): 
# export statistics to csv file     
    arr = []
    for event in range(1,13):
        print('------------------------------------------------')
        print('making statistics for event '+str(event)+'...')
        ev = []
        
        obs, olam = Get_SLP_Wind_At_EvPeak(event,'y')
        obs, olam = obs.SLP/100, olam.sslp/100
        
        ev.append(str(event))
        ev.append(np.amax(obs).values)
        ev.append(np.amin(obs).values)        
        ev.append(np.std(obs).values)
        ev.append(np.amax(olam).values) 
        ev.append(np.amin(olam).values) 
        ev.append(np.std(olam).values)        
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
        ev.append(st.ConcordanceIndex(obs, olam)) 
        ev.append(st.D_pielke(obs, olam))
        arr.append(ev)
        


    with open('SLP_validation_stats.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['Event', 'max_obs', 'min_obs','obs_std','max_model',
                         'min_model', 'model_std', 'Bias',
                         'MAE','MSEtot','MSE_diss','MSE_disp',
                         'Corr_Pearson', 'Corr_Spearmann', 'Corr_Kendall', 
                         'RMSE','RMSE_bias','diff_std','S_sqr','ConcordanceIndex',
                         'D_pielke'])
        writer.writerows(arr)

def Create_df(obs,olam):
        # This is for computating correlation with NaN values
        data_obs = []
        for i in obs.values:
                for j in i:
                    data_obs.append(j)        
        data_olam = []
        for i in olam.values:
                for j in i:
                    data_olam.append(j)                    
        diff = []
        for i in range(len(data_obs)):
                diff.append(data_obs[i]-data_olam[i])
        df = pd.DataFrame({'Obs': data_obs,
                           'olam': data_olam,
                           'diff': diff})               
        return df    
        
def export_stats_wind(): 
# export statistics to csv file     
    arr = []
    for event in range(1,13):
        print('------------------------------------------------')
        print('making statistics for event '+str(event)+'...')
        ev = []
        
        obs, olam = Get_SLP_Wind_At_EvPeak(event,'y')
        obs = np.sqrt(obs.U**2 + obs.V**2).sel(lev=1000)
        olam = np.sqrt(olam.uwnd**2 + olam.vwnd**2)
        
        df = Create_df(obs,olam)
        
        mean_model = float(olam.mean('lon').mean('lat').values)
        mean_obs = float(obs.mean('lon').mean('lat').values)
        ev.append(str(event))
        ev.append(round(mean_obs))
        ev.append(np.amax(obs).values)        
        ev.append(np.std(obs).values)
        ev.append(round(mean_model)) 
        ev.append(np.amax(olam).values)         
        ev.append(np.std(olam).values)        
        ev.append(round(mean_obs - mean_model))
        ev.append(st.BIAS(obs, olam))
        ev.append(st.MAE(obs, olam))
        ev.append(st.MSE(obs, olam))
        ev.append(st.MSE_diss(obs, olam)) 
        ev.append(st.MSE_disp(obs, olam))
        ev.append(df.corr('pearson')['Obs']['olam']) 
        ev.append(df.corr('spearman')['Obs']['olam']) 
        ev.append(df.corr('kendall')['Obs']['olam'])         
        ev.append(st.RMSE(obs, olam))
        ev.append(st.RMSE_bias(obs, olam))
        ev.append(np.std(st.di_acc(obs, olam)).values)
        ev.append(st.S_sqr(obs, olam))
        ev.append(st.ConcordanceIndex(obs, olam)) 
        ev.append(st.D_pielke(obs, olam))
        arr.append(ev)
        


    with open('wind_validation_stats.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['Event', 'mean_obs','max_obs','obs_std','mean_model',
                         'max_model','model_std', 'Deltamean', 'Bias',
                         'MAE','MSEtot','MSE_diss','MSE_disp',
                         'Corr_Pearson', 'Corr_Spearmann', 'Corr_Kendall', 
                         'RMSE','RMSE_bias','diff_std','S_sqr','ConcordanceIndex',
                         'D_pielke'])
        writer.writerows(arr)


def CorrlEachtime(event):
    
    tmp = GetSLPWindData(event)
    olam,obs = tmp[0],tmp[1]
    
    corrl_ws = []
    corrl_slp = []
    for t in olam.time:
        
        obs_ = obs.sel(time=t)        
        olam_ = olam.sel(time=t).assign(lev=obs.lev)        
        olam_ = regridSLPWind(obs.lon,obs.lat,olam_,event)
        
        obs_ws = np.sqrt(obs_.U**2 + obs_.V**2).sel(lev=1000)
        olam_ws = np.sqrt(olam_.uwnd**2 + olam_.vwnd**2)
        
        df = Create_df(obs_ws,olam_ws)
        corrl_ws.append(df.corr('pearson')['Obs']['olam'])
        
        corrl_slp.append(st.Scorr(obs_.SLP, olam_.sslp)[0])
        
    return corrl_ws, corrl_slp, olam.time

def corrl_time_evo(prec):
    
    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.225, wspace=0.1)
    axs = []
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]           
        
        tmp = CorrlEachtime(i)
        corrl_ws, corrl_slp = tmp[0], tmp[1]
        time = tmp[2]
                   
        ax1.plot(time,corrl_ws,c='#836953',marker='o',
                 alpha=0.85,label='Wind Spd.', linewidth=2)
        ax1.plot(time,corrl_slp,c='#028e2c',marker='v',
                 alpha=0.85,  label='SLP', linewidth=2)
        
        if prec == 'y':
            corrl_prec = mean_corrl(i)[2]
            ax1.plot(time,corrl_prec,c='#54718F',marker='s',
                 alpha=0.85,  label='Acc. Prec.', linewidth=2)
        
        if i  == 1 or i == 4 or i == 7 or i == 10:
            ax1.set_ylabel('Corrl. I.',fontsize = 16)
        if i == 1:
            ax1.legend(fontsize=16)
        ax1.text(0.8,0.1, str(i), fontsize = 18, transform=ax1.transAxes, bbox=props)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20)
        ax1.set_ylim(0,1)
        ax1.grid(linestyle='--',linewidth=0.25,c='gray')
    
    if prec == 'n':    
        pl.savefig('./figures/slp_wind/time_corrl.jpg', format='jpg')    
        pl.savefig('./figures/slp_wind/time_corrl.eps', format='eps', dpi=300)    
    elif prec == 'y':    
        pl.savefig('./figures/slp_wind/time_corrl_slp_wind_prec.jpg', format='jpg')    
        pl.savefig('./figures/slp_wind/time_corrl_slp_wind_prec.eps', format='eps', dpi=300)    


def TestMannWithneyU_DailySLP():
    MWUT = {}
    for i in range(1,13):
        MWUT['Event '+str(i)] = []
        tmp = GetSLPWindData(i)
        olam,obs = tmp[0],tmp[1]
        olam = olam.sslp.resample(time='1D').mean('time')
        obs = obs.SLP.resample(time='1D').mean('time')
        #daily corrl
        olam_r = regrid(obs.lon,obs.lat,olam,i)
        for d in olam.time:
            stat, p = stats.mannwhitneyu(obs.sel(time=d).values.ravel(), olam.sel(time=d).values.ravel())
            alpha = 0.05
            if p > alpha:
                MWUT['Event '+str(i)].append('Same distribution (fail to reject H0)')
            else:
                	MWUT['Event '+str(i)].append('Different distribution (reject H0)')
    
    return MWUT

def TestMannWithneyU_allSLP():
    MWUT = {}
    for i in range(1,13):
        MWUT['Event '+str(i)] = []
        tmp = GetSLPWindData(i)
        olam,obs = tmp[0].sslp,tmp[1].SLP
        #daily corrl
        olam_r = regrid(obs.lon,obs.lat,olam,i)
        for d in olam.time:
            stat, p = stats.mannwhitneyu(obs.sel(time=d).values.ravel(), olam.sel(time=d).values.ravel())
            alpha = 0.05
            if p > alpha:
                MWUT['Event '+str(i)].append('Same distribution (fail to reject H0)')
            else:
                	MWUT['Event '+str(i)].append('Different distribution (reject H0)')
    
    return MWUT

def PlotBoxPLotSLP(which):

    fig = plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(4, 3, hspace=0.125, wspace=0.3)
    axs = []
    c1 ='#0077b6'
    c2 = '#ff6961'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
    for i in range(1,13):
        tmp = GetSLPWindData(i)
        olam,obs = tmp[0],tmp[1]
        olam = olam.resample(time='1D').mean('time')
        obs = obs.resample(time='1D').mean('time')
        if which == 'slp':
            obs_data, olam_data = obs.SLP/100, olam.sslp/100 
        elif which == 'wind':
            obs_data = np.sqrt(obs.U**2 + obs.V**2)   
            olam_data = np.sqrt(olam.uwnd**2 + olam.vwnd**2)
        olam_df = DataToDataFrame(olam_data)
        obs_df = DataToDataFrame(obs_data) 
        if which == 'wind':
            mask = ~np.isnan(obs_df.values) 
            obs_df = obs_df[mask]
        #daily corrl
        olam_datar = regrid(obs.lon,obs.lat,olam_data,i)
        corrl = []
        for d in olam.time:
            corrl.append(st.Scorr(obs_data.sel(time=d), olam_datar.sel(time=d))[0])
            for ct in range(3):
                corrl.append(np.nan)
        xs = range(len(corrl))
        s1 = pd.Series(corrl, index=xs)                      
#       # figure
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
        
        if i == 1:
            plt.plot([], c=c1, label='Reanalysis')
            plt.plot([], c=c2, label='OLAM')
            plt.legend(fontsize=14,loc='upper right')
        
        plt.xticks(np.arange(0, (len(ticks)) * 4, 4), range(1,len(obs_df.columns)+1))
        
        ax1.text(.05,0.05, str(i), fontsize = 14,
                 transform=ax1.transAxes, bbox=props)
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        plt.xlim(-3, (len(ticks)*4)+1)
        if i > 9:        
            ax1.set_xlabel('Time (days)', fontsize = 16)
        if i  == 1 or i == 4 or i == 7 or i == 10:
            if which == 'slp':
                ax1.set_ylabel('SLP (hPa)',fontsize = 16)
            elif which == 'wind':
                ax1.set_ylabel('Wind Speed (m s-1)',fontsize = 16)
        if i  == 3 or i == 6 or i == 9 or i == 12:
                ax2.set_ylabel('Corrl I.',fontsize = 16)                
    pl.savefig('./figures/slp_wind/daily_boxplot'+str(which)+'.jpg', format='jpg')
    pl.savefig('./figures/slp_wind/daily_boxplot'+str(which)+'.eps', format='eps', dpi=300)
     
    
## ----------
#if __name__ == "__main__": 
#    plot_slp_panel()
#    plot_wind_panel()
#    obs_model_panel_slp()
#    obs_model_panel_wind()
#    export_stats_SLP()
#    export_stats_wind()
#    corrl_time_evo('n')
#    corrl_time_evo('y')
#    PlotBoxPLotSLP('slp')
#    PlotBoxPLotSLP('wind')