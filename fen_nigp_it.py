# uses conda environment gpflow6_0

from memory_profiler import profile

# generic
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
import time

# plotting

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D 
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gpflow as gpf
from gpflow.ci_utils import ci_niter, ci_range
from gpflow.utilities import print_summary

from fen_functions import *

# tensorflow
import tensorflow as tf
import argparse

@profile

def readv():
    
    
    parser = argparse.ArgumentParser(description='import vars via c-line')
    parser.add_argument("--mod", default='d6g_h6g_')
    parser.add_argument("--lith", default='l71C')
    parser.add_argument("--um", default="p2")
    parser.add_argument("--lm", default="3")
    parser.add_argument("--tmax", default=8000)
    parser.add_argument("--tmin", default=7000)
    parser.add_argument("--place", default="fennoscandia")
    parser.add_argument("--zeros", default="no")
    parser.add_argument("--kernels", default=[2500, 10000, 100, 6000])

    args = parser.parse_args()
    
    ice_model = args.mod
    lith = args.lith
    um = args.um
    lm = args.lm
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place
    zeros = args.zeros
    k1 = int(args.kernels[0])
    k2 = int(args.kernels[1])
    k3 = int(args.kernels[2])
    k4 = int(args.kernels[3])
    
    ####################  Initialize parameters #######################

    agemax = round(tmax, -3) + 100
    agemin = round(tmin, -3) - 100

    ages = np.arange(agemin, agemax, 100)[::-1]

    locs = {'europe': [-20, 15, 35, 70],
            'fennoscandia': [-15, 50, 45, 75],
            'norway': [0, 50, 50, 75],
           }
    extent = locs[place]

    ##Get Norway data sheet from Google Drive
    sheet = 'Norway_isolation'
    df_nor = load_nordata_fromsheet(sheet)

    #import khan dataset
    path = '../data/GSL_LGM_120519_.csv'
    df_place = import_rsls(path, df_nor, tmin, tmax, extent)
    print(f'number of datapoints = {df_place.shape}')

    # add zeros at present-day.  
    if zeros == 'Yes':
        nout = 50
        df_place = add_presday_0s(df_place, nout)
        print('new number of zero points  = {df_place.shape}')

    ####################  Make xarray template  #######################

    filename = '../data/WAISreadvance_VM5_6ka_1step.mat'
    ds_template = xarray_template(filename, ages, extent)

    ####################    Load GIA datasets   #######################

    ds = make_mod(ice_model, lith, ages, extent)

    #make mean of runs
    ds_giamean = ds.mean(dim='modelrun').load().chunk((-1,-1,-1)).interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()
    ds_giastd = ds.std(dim='modelrun').load().chunk((-1,-1,-1)).interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()

    df_place['rsl_giaprior'] = df_place.apply(lambda row: ds_select(ds_giamean, row), axis=1)
    df_place['rsl_giaprior_std'] = df_place.apply(lambda row: ds_select(ds_giastd, row), axis=1)
    df_place['rsl_realresid'] = df_place.rsl - df_place.rsl_giaprior
    df_place['gia_diffdiv'] = df_place.rsl_realresid / df_place.rsl_giaprior_std


    print('number of datapoints = ', df_place.shape)

    ##################	  RUN GP REGRESSION 	#######################
    ##################  --------------------	 ######################
    
 
    k1 = 2500
    k2 = 10000
    k3 = 100
    k4 = 6000

    start = time.time()
    
    mean, ds_giapriorinterp, da_zp, ds_priorplusgpr, ds_varp, m, df_place, k1_l, k2_l, k3_l, k4_l = run_gpr(ds_giamean, ages, k1, k2, k3, k4, df_place)
    print(f'time = {time.time()-start}')

    print_summary(m, fmt='notebook')
    
    ##################	  Plot 3D maps 	         #######################
    ##################  --------------------	 ######################
    
    for i, age in enumerate(ages):
        fig, ax = plt.subplots(1, 3, figsize=(12, 12), subplot_kw=dict(projection=ccrs.LambertConformal(central_longitude=-10)))
        ax = ax.flatten()

        ax[0].coastlines(resolution='50m')
        ax[0].set_extent(extent)

        pc1 = ds_giapriorinterp.rsl.sel(age=age, 
                                        method='nearest').transpose().plot(ax=ax[0], 
                                                                           cmap='coolwarm', 
                          transform=ccrs.PlateCarree(),
                         add_colorbar=False,
                          extend='both',
                                                                          vmin=ds_giapriorinterp.rsl.sel(age=age,
                                                                                                         method='nearest').min(),
                                                                          vmax=ds_giapriorinterp.rsl.sel(age=age,
                                                                                                         method='nearest').max())
        cbar = fig.colorbar(pc1,ax=ax[0],shrink=shrink,label='ice thickness (m)', extend='both')
        ax[0].set_title(f'GIA model of RSL @ {age} yrs')


        ax[1].coastlines(resolution='50m')
        ax[1].set_extent(extent)

        pc1 = da_zp[i,:,:].plot(ax=ax[1], cmap='coolwarm', 
                          transform=ccrs.PlateCarree(),
                         add_colorbar=False,
                          extend='both', 
                               vmin=-15,
                               vmax=10)
        cbar = fig.colorbar(pc1,ax=ax[1],shrink=shrink,label='ice thickness (m)', extend='both')
        ax[1].set_title(f'GPR-learned difference b/w \n GIA prior & data, @ {age} yrs')


        pc2 = ds_priorplusgpr.rsl[i,:,:].plot(ax=ax[2], cmap='coolwarm', 
                                        transform=ccrs.PlateCarree(),
                                       add_colorbar=False,
                          extend='both',
                                             vmin=ds_giapriorinterp.rsl.sel(age=age,method='nearest').min(),
                                            vmax=ds_giapriorinterp.rsl.sel(age=age,method='nearest').max())
        ax[2].set_title(f'Posterior RSL prediction @ {age} yrs')

        cbar = fig.colorbar(pc2,ax=ax[2],shrink=shrink,label='ice thickness (m)', extend='both')
        ax[2].coastlines(resolution='50m')
        ax[2].set_extent(extent)

        if i >= 6:
            break

    number=11

    df_nufsamps = locs_with_enoughsamples(df_place, place, number)
    nufsamp = df_nufsamps.locnum.unique()


    ##################	  Plot timeseries 	     ######################
    ##################  --------------------	 ######################

    fig, ax = plt.subplots(1,len(nufsamp), figsize=(26, 6), subplot_kw=dict(projection=proj))
    ax = ax.ravel()

    da_zeros = xr.zeros_like(ds_giamean.rsl[:,:,0])

    for i, site in enumerate(df_nufsamps.groupby('locnum')):
        ax[i].set_extent(extent)
        ax[i].coastlines(color='k')
        ax[i].plot(site[1].lon.unique(),
                   site[1].lat.unique(),
                   c=colormark[0],
                   ms=7,
                   marker='o',
                   transform=proj)
        ax[i].plot(site[1].lon.unique(),
                   site[1].lat.unique(),
                   c=colormark[0],
                   ms=25,
                   marker='o',
                   transform=proj,
                   mfc="None",
                   mec='red',
                   mew=4)
        da_zeros.plot(ax=ax[i], cmap='Greys', add_colorbar=False)
    #     ax[i].set_title(site[0], fontsize=fontsize)
        ax[i].set_title('')
    #     if i > 6:
    #         break



    proj = ccrs.PlateCarree()

    colormark = ['dodgerblue', 'chocolate', 'darkred', 'crimson', 'olivedrab']
    cmaps = cmap_codes('viridis', len(df_nufsamps))

    num = 6
    fig, ax = plt.subplots(1, len(nufsamp), figsize=(24, 6))
    ax = ax.ravel()

    for i, site in enumerate(df_nufsamps.groupby('locnum')):

        plt.suptitle(f"Prior GIA RSL predictions vs. Posterior GPR regressions", fontsize=20)

    #     #slice data for each site
        prior_it = slice_dataset(ds_giamean) # [:,:,:-4]
        var_itprior = slice_dataset(ds_giastd)
        top_prior = prior_it + var_itprior * 2
        bottom_prior = prior_it - var_itprior * 2


        post_it = slice_dataset(ds_priorplusgpr)
        var_it = slice_dataset(np.sqrt(ds_varp))
        top = post_it + var_it * 2
        bottom = post_it - var_it * 2

        site_err = 2 * (site[1].rsl_er)
        age_err = 2 * site[1].age_er

        prior_it.plot(ax=ax[i], c=colormark[0], alpha=0.4, label='ICE6G GIA prior')
        ax[i].fill_between(prior_it.age, bottom_prior.squeeze(), top_prior.squeeze(), color=colormark[0], alpha=0.1) 


        post_it.plot(ax=ax[i], lw=2, c=colormark[1], alpha=1, label='ICE6G GPR posterior')
        ax[i].fill_between(post_it.age, bottom.squeeze(), top.squeeze(), color=colormark[3], alpha=0.4) 


        ax[i].scatter(site[1].age, site[1].rsl, c=colormark[2], s=4, lw=2,label='RSL data')
        ax[i].errorbar(site[1].age, site[1].rsl, yerr=site_err, xerr=age_err, c=colormark[2], fmt='none', capsize=-.1,lw=1.5)

        ax[i].set_xlim(0, 11500)
    #     ax[i].set_ylim(-15,10)
        ax[i].set_title('')
    #     if i > num:
    #         break

    lines = [ Line2D([0], [0], color=colormark[0], linewidth=3, linestyle='-'),
             Line2D([0], [0], color=colormark[1], linewidth=3, linestyle='-'),
             Circle([0], 0.1, color=colormark[2], linewidth=3, ec="none")]

    labels = ['ICE6G GIA prior', 'ICE6G GPR posterior', 'RSL data']
    ax[i].legend(lines, labels, loc='lower left')


    plt.show()
            

if __name__ == '__main__':
    readv()