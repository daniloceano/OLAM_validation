#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:45:09 2021

@author: Danilo
"""
from prepare_data import (open_olam, open_gpm_trmm, open_merra,
                          unacc_olam_prec, GPM_to_3h, regrid)
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature

def GetData(event):
    min_lon, max_lon, min_lat, max_lat = -54, -44.05, -34, -25.05
    olam_data = open_olam(event,'pmic').pmic + open_olam(event,'pcon').pcon
    if event < 3:
        re_data = open_gpm_trmm(event).precipitationCal
    else: 
        re_data = open_gpm_trmm(event).precipitation
    # Slice in order to maintain same spatial domain
    olam_data = olam_data.sel(lat=slice(min_lat,max_lat),
                              lon=slice(min_lon,max_lon))
    re_data = re_data.sel(lat=slice(min_lat,max_lat),
                              lon=slice(min_lon,max_lon))
    return olam_data, re_data

                 
def CheckOLAM():
 
    #### Check OLAM unaccumulate algorithm             
    for event in range(1,13):        
        olam_data = GetData(event)[0]   
        olam_unacc = unacc_olam_prec(olam_data)
        plt.figure()
        for t in olam_data.time:
            plt.scatter(olam_data.time.sel(time=t),
                        olam_data.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.).values,
                        color='r')
            plt.scatter(olam_unacc.time.sel(time=t),
                        olam_unacc.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.15).values,
                        color='b')
            plt.title('OLAM accumulated (red) x unaccumulated (blue) data for Fpolis')

def GPM3h():

    ##### Check Reanalysis algorithm for 3h
    for event in range(1,3):
        olam_data,re_data = GetData(event)[0],GetData(event)[1]    
        re_3h = GPM_to_3h(olam_data.time,re_data)
        plt.figure()        
        for t in olam_data.time:
            plt.scatter(re_data.time.sel(time=t),
                        re_data.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.15).values,
                        color='r')
            plt.scatter(re_3h.time.sel(time=t),
                        re_3h.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.15).values,
                        color='b') 
        plt.figure() 
        re_data_acc = re_data.cumsum('time')
        re_3h_acc = re_data.cumsum('time')
        for t in olam_data.time:
            plt.scatter(re_data_acc.time.sel(time=t),
                        re_data_acc.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.15).values,
                        color='r')
            plt.scatter(re_3h_acc.time.sel(time=t),
                        re_3h_acc.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.15).values,
                        color='b')                          
            
 
def CheckEachEventAcc():
    
    for event in range(1,13):        
        olam_data = GetData(event)[0]
        if event < 3:
            re_data = GetData(event)[1]
            re_data = GPM_to_3h(olam_data.time,re_data)
            re_data = re_data.cumsum('time')
        else: 
            re_data2 = GetData(event)[1][:len(re_data)]
            re_data2 = re_data2.cumsum('time')
        plt.figure()
        ct = 0
        for t in olam_data.time:
            plt.scatter(olam_data.time.sel(time=t),
                        olam_data.sel(time=t,
                                      lat=slice(-27,-26.95),lon=-48.).values,
                        color='r')
            if event < 3:
                plt.scatter(re_data.time.sel(time=t),
                            re_data.sel(time=t,
                                          lat=slice(-27,-26.95),lon=-48.15).values,
                            color='b')
                plt.title('OLAM (red) x Reanalyis (blue) acc prec data for Fpolis')
            else:
                plt.scatter(olam_data.time.sel(time=t),
                            re_data2.sel(time=ct,
                                          lat=slice(-27.125,-27),lon=-48.125).values,
                            color='b')
                plt.title('OLAM (red) x Reanalyis (blue) acc prec data for Fpolis')
           
        
def CheckRegrid():
    
    min_lon, max_lon, min_lat, max_lat = -53.95, -45.05, -26.05, -33.95

    for event in range(1,13):
        olam_data = GetData(event)[0]
        if event < 3:
            re_data = GetData(event)[1]
        else: 
            re_data = GetData(event)[1]
        re_data_reg = regrid(olam_data.lon,olam_data.lat,re_data,event)
        re_data = re_data.transpose('time','lat','lon')
        re_data = re_data.cumsum('time')
        re_data_reg = re_data_reg.cumsum('time')
        
        lons_high, lats_high = olam_data.lon, olam_data.lat
        lons_low, lats_low = re_data.lon, re_data.lat
        #
        plt.figure(figsize=(10, 10))
        datacrs = ccrs.PlateCarree()
        gs = gridspec.GridSpec(1, 2, height_ratios=[1],
                           bottom=0, top=1, wspace=0,hspace = 0)
        ax1 = plt.subplot(gs[0, 0], projection=datacrs)
        ax2 = plt.subplot(gs[0, 1], projection=datacrs)
        axs = [ax1,ax2]
#        cbar_ax1 = fig.add_axes([0.92, 0.335, 0.028, 0.338]) 
        for ax in axs:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5,linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False    
            gl.xlabel_style = {'size': 18, 'color': 'gray'}
            gl.ylabel_style = {'size': 18, 'color': 'gray'}
            ax1.outline_patch.set_edgecolor('gray')
            if ax == ax2:
                gl.ylabels_left = False 
        # contour orginal precipitation
        ax1.contourf(lons_low, lats_low, re_data[-1],cmap='magma_r',extend= 'max')        
        ax1.contour(lons_low, lats_low, re_data[-1],colors='grey', linewidths=1) 
        # contour regridded precipitation
        ax2.contourf(lons_high, lats_high, re_data_reg[-1],cmap='magma_r',extend= 'max')        
        ax2.contour(lons_high, lats_high, re_data_reg[-1],colors='grey', linewidths=1) 
        plot_background(ax1,min_lon, max_lon, min_lat, max_lat)
        plot_background(ax2,min_lon, max_lon, min_lat, max_lat)
        
        
def plot_background(ax,min_lon, max_lon, min_lat, max_lat):
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lines',
                                                    scale='50m', facecolor='none')
    country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                                   name='admin_0_countries',    
                                                   scale='50m', facecolor='none')       
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax        
        
#    plt.figure()
    
#if __name__ == "__main__":    
#    CheckEachEventAcc()           