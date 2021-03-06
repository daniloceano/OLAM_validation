# Souza & Ramos da Silva,

# Ocean-Land Atmosphere Model (OLAM) performance for major extreme
#   meteorological events near the coastal region of southern Brazil,

# Climate Research, in revision 2020


"""
Created on Wed Jan  6 17:36:04 2021

@author: Danilo


Script for creating animations showing OLAM and Reanalysis 
precipitation, slp and wind vectors, for each event

"""
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from prepare_data import (open_olam, open_gpm_trmm, open_merra, unacc_olam_prec)
    
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
def plot_background(ax):
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax
# ----------
def make_gif(event): 
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
    # ----------
    # open OLAM files and get data
    file = open_olam(event,'sslp')  
    slp = open_olam(event,'sslp').sslp/100
    uwnd = open_olam(event,'uwnd').uwnd
    vwnd = open_olam(event,'vwnd').vwnd
    pmic = open_olam(event,'pmic').pmic
    pcon = open_olam(event,'pcon').pcon
    # --
    lons, lats = file.lon, file.lat
    X,Y = np.meshgrid(lons,lats)      
    # --
    # Sum OLAM microphysicis and convective prec
    # and unaccumulate OLAM prec
    accpt = pmic+pcon
    times = accpt.time
    pt = unacc_olam_prec(accpt)
    # ----------
    # Open reanalysis files get data
    # --   
    if event < 3:
        # GPM file and variables    
        pr = open_gpm_trmm(event).precipitationCal
        pr3h = pr.resample(time='3h').sum() 
    else:
        pr3h = open_gpm_trmm(event).precipitation
    lonsrep, latsrep = pr3h.lon, pr3h.lat 
    # --   
    # MERRA-2 file and variables
    mfile = open_merra(event)
    slpm = mfile.SLP/100
    um = mfile.U
    vm = mfile.V
    lonsre, latsre = mfile.lon, mfile.lat   
    Xre,Yre = np.meshgrid(lonsre,latsre)              
    # ----------
    # fig params    
    fig = plt.figure(figsize=(10, 10))
    datacrs = ccrs.PlateCarree()
    gs = gridspec.GridSpec(1, 2, height_ratios=[1],
                       bottom=0, top=1, wspace=0,hspace = 0)
    ax1 = plt.subplot(gs[0, 0], projection=datacrs)
    ax2 = plt.subplot(gs[0, 1], projection=datacrs)
    axs = [ax1,ax2]
    cbar_ax1 = fig.add_axes([0.92, 0.335, 0.028, 0.338])
    camera = Camera(fig)
    # --    
#    clevs_prec = np.linspace(np.amin(pr3h)/10, np.amax(pr3h)+(np.amax(pr3h)/10), 15)    
    clevs_prec = np.linspace(0, 50, 11)        
    # ----------
    # plot each time          
    ct = 0
    for t in times: 
        print()
        # -------------
        # Grid lines
        if ct == 0:
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
        # -------------
        # Plot OLAM data
        # --
        # contour slp
        high = slp.sel(time=t) >= 1014
        low = slp.sel(time=t) <= 1014
        # --
        ch = ax1.contour(np.ma.masked_where(high, X),np.ma.masked_where(high, Y),
                    np.ma.masked_where(high, slp.sel(time=t)), levels = 6,colors = ('b'),
                    linewidths = 2,linestyles = 'dashed', eXtend='both')
        cl = ax1.contour(np.ma.masked_where(low, X),np.ma.masked_where(low, Y),
                    np.ma.masked_where(low, slp.sel(time=t)), levels = 6,colors = ('r'),
                    linewidths =2,extend='both')
        ax1.clabel(ch, inline=1, fontsize=10, fmt='%1.0f')
        ax1.clabel(cl, inline=1, fontsize=10, fmt='%1.0f')        
        # --
        # contour precipitation
        cf1 = ax1.contourf(lons, lats, pt.sel(time=t), clevs_prec,cmap=cmap,extend= 'max')        
        ax1.contour(lons, lats, pt.sel(time=t), clevs_prec,colors='grey', linewidths=1) 
        # --
        # wind vectors
        ax1.quiver(lons[::10],lats[::10],uwnd.sel(time=t)[::10,::10],
                  vwnd.sel(time=t)[::10,::10], color= 'k')
       
        # -------------
        # Plot Reanalysis data
        # --
        # contour slp
        high = slpm.sel(time=t) >= 1014
        low = slpm.sel(time=t) <= 1014    
        # --        
        ch = ax2.contour(np.ma.masked_where(high, Xre),np.ma.masked_where(high, Yre),
                    np.ma.masked_where(high, slpm.sel(time=t)), levels = 6,colors = ('b'),
                    linewidths = 2,linestyles = 'dashed', eXtend='both')
        cl = ax2.contour(np.ma.masked_where(low, Xre),np.ma.masked_where(low, Yre),
                    np.ma.masked_where(low, slpm.sel(time=t)), levels = 6,colors = ('r'),
                    linewidths =2,extend='both')
        ax2.clabel(ch, inline=1, fontsize=10, fmt='%1.0f')
        ax2.clabel(cl, inline=1, fontsize=10, fmt='%1.0f')         
        # --       
        # contour precipitation
        if event < 3:
            prec = pr3h.sel(time=t)
            prec = prec.transpose('lat','lon')
        else:
            prec = pr3h.sel(time=ct)
        cf1 = ax2.contourf(lonsrep, latsrep, prec, clevs_prec,cmap=cmap,extend= 'max')        
        ax2.contour(lonsrep, latsrep, prec, clevs_prec,colors='grey', linewidths=1)   
        # --
        # wind vectors
        ax2.quiver(lonsre,latsre,um.sel(time=t)[0],vm.sel(time=t)[0],color= 'k')        
        # -------------
        # Cosmedics 
        # --
        plot_background(ax1)
        plot_background(ax2)
        plt.colorbar(cf1, orientation='vertical',cax=cbar_ax1) 
        ax1.text(max_lon-2.5,max_lat+.75,str(t.dt.strftime('%m/%d/%Y %HZ').values), fontsize = 20)       
        ax1.text(min_lon+3.5,max_lat+.25,'OLAM', fontsize = 18)        
        ax2.text(min_lon+3,max_lat+.25,'Reanalysis', fontsize = 18)       
        # -------------       
        camera.snap()
        ct += 1
    # -------------    
    animation = camera.animate(interval = 200, repeat = True,
                                   repeat_delay = 100000)
    animation.save('/animations/wind_prec/E0'+str(event)+'_wind_prec.gif')    
# ----------
# make gif for all events 
if __name__ == "__main__":    
    for event in range(1,13):
        print('making figure for event: '+str(event))
        make_gif(event)            
            

    