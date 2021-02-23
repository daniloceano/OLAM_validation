# Souza & Ramos da Silva,

# Ocean-Land Atmosphere Model (OLAM) performance for major extreme
#   meteorological events near the coastal region of southern Brazil,

# Climate Research, in revision 2020
"""
Created on Wed Dec  4 18:57:06 2019

This script will plot OLAM wind and sensible heat fluxes  fields


@author: Danilo Couto de Souza
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean 
import pylab as pl
from matplotlib import colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from prepare_data import (FluxData, unacc_olam_prec)
from stats_slp_wind import Get_SLP_Wind_At_EvPeak

# dates corresponding to each event peak time
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


# this will simply plot the values for a grid point,
#  so we cann actually "see" the data
def CheckValues(n):
    fig = plt.figure(figsize=(8,10), constrained_layout=False)
    gs1 = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.4)
    axs = []
    for i in range(1,13):
        axs.append(fig.add_subplot(gs1[i - 1]))
        ax1 = axs[-1]
        if n == 1:
            data = FluxData(i).shf
            text = 'Acc SHF (J m-2) for Fpolis'
            name = 'shf1'
        elif n == 2:
            data = FluxData(i).shf/10800
            data = unacc_olam_prec(data)
            text = 'SHF (W m-2) for Fpolis'
            name = 'shf2'
        elif n == 3:
            data = FluxData(i).lhf
            text = 'Acc LHF (J m-2) for Fpolis'
            name = 'lhf1'
        elif n == 4:
            data = FluxData(i).lhf/10800
            data = unacc_olam_prec(data)
            text = 'LHF (W m-2) for Fpolis'
            name = 'lhf2'
        ax1.plot(data.time,data.sel(lat=slice(-27.55,-27.5), lon=slice(-48.5,-48.45)).values.ravel())
        ax1.text(0.8,0.1, str(i), fontsize = 18, transform=ax1.transAxes)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20) 
        if i == 2:
            ax1.text(0,1.1, text, fontsize = 18, transform=ax1.transAxes)
    pl.savefig('./figures/lat_sens_heat/check_'+str(name)+'_panel.jpg', format='jpg')    

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

def clippedcolorbar(CS, **kwargs):
    # from https://stackoverflow.com/questions/43150687/colorbar-limits-are-not-respecting-set-vmin-vmax-in-plt-contourf-how-can-i-more
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    fig = CS.ax.get_figure()
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    return fig.colorbar(m, **kwargs)



def MapFlux(which):
    # skipping vectors
    skip = 10
    # figure
    datacrs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10,13), constrained_layout=False)
    gs1 = gridspec.GridSpec(4, 3, hspace=0.2, wspace=0.2)
    axs = []
    # box for plotting texts
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1,13):          
        axs.append(fig.add_subplot(gs1[i - 1], projection=datacrs))
        ax1 = axs[-1] 
        # flux data 
        if which == 'lat':
            data = FluxData(i).lhf/10800
            data = unacc_olam_prec(data)
            levels = 15
            data = data.sel(time=dates[i-1])
            norm=colors.Normalize(vmin=0, vmax=np.amax(data))
            ticks = np.linspace(0,np.amax(data),5) # see exp. at cbar
            cmap = cmocean.cm.mattercmap = cmocean.cm.matter
        elif which == 'sens':
            data = FluxData(i).shf/10800
            data = unacc_olam_prec(data)
            levels = 30                 
            data = data.sel(time=dates[i-1])
            min_ = np.amin(data)
            max_ = np.amax(data)
            # if only positive values set range from 0 to max
            if min_ >= -40:
                norm=colors.Normalize(vmin=0, vmax=max_)
                ticks = np.linspace(0,np.amax(data),5) # see exp. at cbar
                cmap = cmocean.cm.mattercmap = cmocean.cm.matter
            # if negative values, try to set the midpoint at 0, but didnt work
            else:
                norm = colors.DivergingNorm(vmin=-max_, vcenter=0, vmax=max_)
                t1 = np.linspace(-max_,0,3)
                t2 = np.linspace(-t1[-2],max_,2)
                ticks = np.concatenate((t1,t2)) # see exp. at cbar
                colors1 = cmocean.cm.deep_r(np.linspace(0., 1, 128))
                colors2 = cmocean.cm.matter(np.linspace(0, 1, 128))
                mycolors = np.vstack((colors1, colors2))
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap', mycolors)
        # wind data
        wnd = Get_SLP_Wind_At_EvPeak(i,'n')[1]
        wnd = wnd.transpose('lat','lon','lev')
        u, v = wnd.uwnd, wnd.vwnd
        # contour flux data
        cf1 = ax1.contourf(data.lon, data.lat, data, norm=norm,
                           levels = levels, cmap=cmap)   
        # wind vectors
        ax1.quiver(data.lon[::skip],data.lat[::skip],
                   u[::skip,::skip],v[::skip,::skip],
                   color= 'k') 
        ax1.text(-53,-28,str(i), fontsize = 16, bbox=props)
        # Colorbar
        #  as I was not able to hide every nth labels,
        #  the 'ticks' works a workatround
        pos = ax1.get_position()
        cbar_ax1 = fig.add_axes([pos.x1+0.005, pos.y0, 0.02, pos.height])
        cbar1 = clippedcolorbar(cf1, extend='neither', cax=cbar_ax1, orientation='vertical')
        cbar1.ax.tick_params(labelsize=12)
        cbar1.set_ticks(ticks)
        # map cosmedics 
        plot_background(ax1)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False    
        gl.xlocator = mticker.FixedLocator(range(-54,-40,2))
        gl.xlabel_style = {'size': 13, 'color': 'gray'}
        gl.ylabel_style = {'size': 14, 'color': 'gray'}
        ax1.outline_patch.set_edgecolor('gray')
        
    pl.savefig('./figures/lat_sens_heat/'+str(which)+'_panel.jpg', format='jpg')    
    pl.savefig('./figures/lat_sens_heat/'+str(which)+'_panel.eps', format='eps', dpi=300)    
#                

if __name__ == "__main__":
        MapFlux('lat')
        MapFlux('sens')
