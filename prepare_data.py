#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:49:19 2021

@author: Danilo
"""
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
# ----------
# custom functions
def open_olam(event,var):    
    if event < 10:
        file = xr.open_dataset('./OLAM_netcdf/OLAM_alltimes_E0'
                                +str(event)+'_'+str(var)+'.nc')
    else:
         file = xr.open_dataset('./OLAM_netcdf/OLAM_alltimes_E'
                                 +str(event)+'_'+str(var)+'.nc')         
    return file
# --
def open_gpm_trmm(event):
    if event < 3:
        file = xr.open_mfdataset('./GPM_TRMM_netcdf/E0'+str(event)+'/*.nc4')        
    elif 3 <=  event < 10:    
        file = xr.open_dataset('./GPM_TRMM_netcdf/E0'+str(event)+'/E0'+str(event)+'.nc')
    else:
        file = xr.open_dataset('./GPM_TRMM_netcdf/E'+str(event)+'/E'+str(event)+'.nc')        
    return file
# --
def open_merra(event):
    if event < 10:
        file = xr.open_mfdataset('./MERRA_netcdf/E0'+str(event)+'/MERRA2*.nc')
    else:
        file = xr.open_mfdataset('./MERRA_netcdf/E'+str(event)+'/MERRA2*.nc')
    return file                 

# --
def unacc_olam_prec(accpt):
    pt = accpt*np.nan # prec olam
    times = accpt.time
    for t in range(len(times)):
        if t == 0:
            pt.loc[dict(time=times[0])] = accpt.sel(time=times[0])
        else:
            pt.loc[dict(time=times[t])] = accpt.sel(time=times[t]) - accpt.sel(time=times[t-1])
    return pt

# --
def regrid(input_lon,input_lat,input_file,event):
    input_file = input_file.transpose('time','lat','lon')

    output_file = xr.DataArray(coords=[input_file.time, input_lat, input_lon],
                                   dims=["time", "lat","lon"])
    regridder = xe.Regridder(input_file, output_file, 'bilinear')
    regridder.clean_weight_file()
    regridder
    output_file = regridder(input_file) 
    return output_file

# --
def regridSLPWind(input_lon,input_lat,input_file,event):
    input_file = input_file.transpose('lat','lev','lon')

    output_file = xr.DataArray(coords=[input_lat, input_file.lev, input_lon],
                                   dims=["lat","lev","lon"])
    regridder = xe.Regridder(input_file, output_file, 'bilinear')
    regridder.clean_weight_file()
    regridder
    output_file = regridder(input_file) 
    return output_file

def GetPrecData(event):
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

def GetSLPWindData(event):
    min_lon, max_lon, min_lat, max_lat = -54, -44.05, -34, -25.05
    olam_data = open_olam(event,'uwnd').assign(open_olam(event,'vwnd')).assign(open_olam(event,'sslp'))
    re_data = open_merra(event)    
    # Slice in order to maintain same spatial domain
    olam_data = olam_data.sel(lat=slice(min_lat,max_lat),
                              lon=slice(min_lon,max_lon))
    re_data = re_data.sel(lat=slice(min_lat,max_lat),
                              lon=slice(min_lon,max_lon))   
    return olam_data, re_data       