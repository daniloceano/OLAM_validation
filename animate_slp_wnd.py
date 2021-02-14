#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:36:04 2021

@author: Danilo
"""
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
import xarray as xr
from celluloid import Camera
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean as cmo
import cartopy.feature as cfeature
import cartopy.crs as ccrs
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
# --
# Make country borders feature
country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_countries',
                                               scale='50m', facecolor='none')       
# --
clevs_slp = np.arange(990, 1040, 5)
# ----------
# custom functions
def open_file(event,var):    
    if event < 10:
        file = xr.open_dataset('./OLAM_netcdf/OLAM_alltimes_E0'
                                +str(event)+'_'+str(var)+'.nc')
    else:
         file = xr.open_dataset('./OLAM_netcdf/OLAM_alltimes_E'
                                 +str(event)+'_'+str(var)+'.nc')         
    return file
# --
def plot_background(ax):
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax
# --   
def make_gif(event):    
    # open files and get data
    file_slp = open_file(event,'sslp')
    file_uwnd = open_file(event,'uwnd')
    file_vwnd = open_file(event,'vwnd')
    # --
    slp = file_slp.sslp
    uwnd = file_uwnd.uwnd
    vwnd = file_vwnd.vwnd
    # --
    lons, lats = file_slp.lon, file_slp.lat     
    # ----------
    # fig params    
    fig = plt.figure(figsize=(10, 10))
    datacrs = ccrs.PlateCarree()
    gs = gridspec.GridSpec(1, 1, height_ratios=[1],
                       bottom=0, top=1, wspace=0,hspace = 0)
    ax1 = plt.subplot(gs[0, 0], projection=datacrs)
    cbar_ax1 = fig.add_axes([0.925, 0.16, 0.028, 0.68])
    camera = Camera(fig)
    # ----------
    # plot each time           
    for t in file_slp.time:
        cf1 = ax1.contourf(lons,lats,slp.sel(time=t)/100, clevs_slp,
                           cmap = cmo.cm.balance)
        ax1.contour(lons,lats,slp.sel(time=t)/100, clevs_slp,
                           colors = 'gray', linewidths=1)
        ax1.quiver(lons[::10],lats[::10],uwnd.sel(time=t)[::10,::10],
                  vwnd.sel(time=t)[::10,::10], color= 'k') 
        plot_background(ax1)
        plt.colorbar(cf1, orientation='vertical',cax=cbar_ax1)    
        ax1.set_title('SLP (hPa) and 30m Wind Vectors (m/s)')
        camera.snap()
    
    animation = camera.animate(interval = 200, repeat = True,
                                   repeat_delay = 100000)
    animation.save('E0'+str(event)+'_wind_slp.gif')    
# ----------
# make gif for all events                           
for event in range(1,2):
     make_gif(event)            
            

    