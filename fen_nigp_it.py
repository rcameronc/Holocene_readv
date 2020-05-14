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
    if zeros == 'yes':
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
    
    mean, ds_giapriorinterp, da_zp, ds_priorplusgpr, ds_varp, m, df_place, k1_l, k2_l, k3_l, k4_l = run_gpr(ds_giamean, ds_giastd, ages, k1, k2, k3, k4, df_place)
    print(f'time = {time.time()-start}')

    print_summary(m, fmt='notebook')
    print(k1_l, k2_l, k2_l, type(k4_l))

    
    path_gen = f'output/{place}_{ice_model}{ages[0]}_{ages[-1]}'
    da_zp.to_netcdf(path_gen + '_dazp')
    ds_giapriorinterp.to_netcdf(path_gen + '_giaprior')
    ds_priorplusgpr.to_netcdf(path_gen + '_posterior')
    ds_varp.to_netcdf(path_gen + '_gpvariance')

#     #store log likelihood in dataframe
#     df_out = pd.DataFrame({'modelrun': 'average_likelihood',
#                      'log_marginal_likelihood': loglikelist})

#     writepath = f'output/{path_gen}_loglikelihood'
#     df_out.to_csv(writepath, index=False)

            

if __name__ == '__main__':
    readv()