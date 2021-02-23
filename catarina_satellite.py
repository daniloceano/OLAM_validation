# Souza & Ramos da Silva,

# Ocean-Land Atmosphere Model (OLAM) performance for major extreme
#   meteorological events near the coastal region of southern Brazil,

# Climate Research, in revision 2020
"""
Created on Thu Feb 20 19:45:02 2020

This script will plot a satelite img of the Catarina Hurricane
 in the background and the modeled OLAM wind data upon it

@author: Danilo Couto de Souza

"""
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from prepare_data import GetSLPWindData
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker


# Make state boundaries feature
states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
# Make country borders feature
country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_countries',    
                                               scale='50m', facecolor='none')       
# This cames from the satellite img,
#  if not correct, will plot wrong positions
min_lon, max_lon, min_lat, max_lat = -53.9985, -45.0882, -33.9845, -26.0135

def plot_background(ax):
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    ax.coastlines('50m', edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
    return ax

def map_satellite_olam():
    
    # slice original data to match the image limits
    olam = GetSLPWindData(12)[0].sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
    lon, lat = olam.lon, olam.lat
    # dates choosen and the figures were name accordingly
    dates = ['2004-03-25','2004-03-26','2004-03-27','2004-03-28']
    # model time steps matching the dates above, but for 15 UTC
    times = [61,69,77,85]
    # fig params
    axs = []
    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.1)
    datacrs = ccrs.PlateCarree()
    panel = 0
    for day, time in zip(dates,times):
        # loop through dates and time steps
        axs.append(fig.add_subplot(gs[panel], projection=datacrs))
        ax1 = axs[-1]
        panel +=1
        # data
        u = olam.uwnd.sel(time=olam.time[time])
        v = olam.vwnd.sel(time=olam.time[time]) 
        slp = olam.sslp.sel(time=olam.time[time])/100 
        # plot only minumum SLP
        min_ = np.amin(slp)+2
        slp_f = slp.where(slp < min_)
        # get the path of the file.
        fname =  './figures/satellite/'+day+'.png'    
        img_extent = (-53.9985, -45.0882, -33.9845, -26.0135)
        img = plt.imread(fname)    
        # set a margin around the data
        ax1.set_xmargin(0.05)
        ax1.set_ymargin(0.10)    
        # Add the image. 
        #  Because this image was a tif, the "origin" of the image is in the
        #  upper left corner
        ax1.imshow(img, origin='upper', extent=img_extent,
                  transform=ccrs.PlateCarree(), alpha = 0.8)
        ax1.coastlines(resolution='50m', color='black', linewidth=1)
        ax1.add_feature(states_provinces, edgecolor='black', linewidth=1)
        # add slp contours
        cs = ax1.contour(lon, lat, slp_f,colors='#0077b6',
                    linewidths=2, linestyles='dashed')
        ax1.clabel(cs, cs.levels[::2], inline=True, fmt = '%d',
                   fontsize=12, colors = 'k')
        # for t in tl:
        #     t.set_bbox(dict(boxstyle="round",fc="y", facecolor='wheat', alpha=0.5))
        # add wind vectors
        skip = 10
        ax1.quiver(lon[::skip],lat[::skip],u[::skip,::skip],v[::skip,::skip],
                       color= 'k') 
        # cosmedics
        plot_background(ax1)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlocator = mticker.FixedLocator(range(-54,-40,2))
        gl.xlabel_style = {'size': 13, 'color': 'gray'}
        gl.ylabel_style = {'size': 14, 'color': 'gray'}
        ax1.outline_patch.set_edgecolor('gray')
        ax1.text(0.25,1.05, day, fontsize = 18,
                 transform=ax1.transAxes)
        
    pl.savefig('./figures/satellite/catarina_snapshot.eps', format='eps', dpi=300)
    pl.savefig('./figures/satellite/catarina_snapshot.png', format='png')
    

if __name__ == "__main__": 
    map_satellite_olam()

