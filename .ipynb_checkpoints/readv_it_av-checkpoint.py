# uses conda environment gpflow6_0

# #!/anaconda3/envs/gpflow6_0/env/bin/python

from memory_profiler import profile

# generic
import numpy as np
import pandas as pd
import xarray as xr
# import dask.array as da
import scipy.io as io
from itertools import product
import glob
import time
import os

# plotting
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import seaborn as sns

# gpflow
import gpflow as gpf
from gpflow.utilities import print_summary
from gpflow.logdensities import multivariate_normal
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from typing import Optional, Tuple, List
from gpflow.config import default_jitter

# tensorflow
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
import argparse

@profile

def readv():

    # set the colormap and centre the colorbar
    class MidpointNormalize(Normalize):
        """Normalise the colorbar.  e.g. norm=MidpointNormalize(mymin, mymax, 0.)"""
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


    ####################  Initialize parameters #######################
    #################### ---------------------- #######################

    parser = argparse.ArgumentParser(description='import vars via c-line')
    parser.add_argument("--tmax", default="12010")
    parser.add_argument("--tmin", default="3990")
    parser.add_argument("--place", default="fennoscandia")

    args = parser.parse_args()
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place
    
    ice_model =  ['d6g_h6g_', 'glac1d_'] 

    locs = {'europe': [-20, 15, 35, 70],
            'atlantic':[-85,50, 25, 73],
            'fennoscandia': [-15, 50, 45, 75]
           }
    extent = locs[place]
    tmax, tmin, tstep = int(tmax), int(tmin), 100

    ages_lgm = np.arange(100, 26000, tstep)[::-1]

    #import khan dataset
    path = 'data/GSL_LGM_120519_.csv'

    df = pd.read_csv(path, encoding="ISO-8859-15", engine='python')
    df = df.replace('\s+', '_', regex=True).replace('-', '_', regex=True).\
            applymap(lambda s:s.lower() if type(s) == str else s)
    df.columns = df.columns.str.lower()
    df.rename_axis('index', inplace=True)
    df = df.rename({'latitude': 'lat', 'longitude': 'lon'}, axis='columns')
    dfind, dfterr, dfmar = df[(df.type == 0)
                              & (df.age > 0)], df[df.type == 1], df[df.type == -1]
    np.sort(list(set(dfind.regionname1)))

    #select location
    df_place = dfind[(dfind.age > tmin) & (dfind.age < tmax) &
                     (dfind.lon > extent[0])
                     & (dfind.lon < extent[1])
                     & (dfind.lat > extent[2])
                     & (dfind.lat < extent[3])][[
                         'lat', 'lon', 'rsl', 'rsl_er_max', 'age']]

    ####################  Make 3D fingerprint  #######################
    #################### ---------------------- #######################

    filename = 'data/WAISreadvance_VM5_6ka_1step.mat'

    waismask = io.loadmat(filename, squeeze_me=True)
    ds_mask = xr.Dataset({'rsl': (['lat', 'lon', 'age'], waismask['RSL'])},
                         coords={
                             'lon': waismask['lon_out'],
                             'lat': waismask['lat_out'],
                             'age': np.round(waismask['ice_time_new'])
                         })
    fingerprint = ds_mask.sel(age=ds_mask.age[0])


    def make_fingerprint(start, end, maxscale):

        #palindromic scaling vector
        def palindrome(maxscale, ages):
            """ Make palindrome scale 0-maxval with number of steps. """
            half = np.linspace(0, maxscale, 1 + (len(ages) - 1) // 2)
            scalefactor = np.concatenate([half, half[::-1]])
            return scalefactor

        ages_readv = ages_lgm[(ages_lgm < start) & (ages_lgm >= end)]
        scale = palindrome(maxscale, ages_readv)

        #scale factor same size as ice model ages
        pre = np.zeros(np.where(ages_lgm == start)[0])
        post = np.zeros(len(ages_lgm) - len(pre) - len(scale))

        readv_scale = np.concatenate([pre, scale, post])

        #scale factor into dataarray
        da_scale = xr.DataArray(readv_scale, coords=[('age', ages_lgm)])

        # broadcast fingerprint & scale to same dimensions;
        fingerprint_out, fing_scaled = xr.broadcast(fingerprint.rsl, da_scale)

        # mask fingerprint with scale to get LGM-pres timeseries
        ds_fingerprint = (fingerprint_out *
                          fing_scaled).transpose().to_dataset(name='rsl')

        # scale dataset with fingerprint to LGM-present length & 0-max-0 over x years
        xrlist = []
        for i, key in enumerate(da_scale):
            mask = ds_fingerprint.sel(age=ds_fingerprint.age[i].values) * key
            mask = mask.assign_coords(scale=key,
                                      age=ages_lgm[i]).expand_dims(dim=['age'])
            xrlist.append(mask)
        ds_readv = xr.concat(xrlist, dim='age')

        ds_readv.coords['lon'] = pd.DataFrame((ds_readv.lon[ds_readv.lon >= 180] - 360)- 0.12) \
                                .append(pd.DataFrame(ds_readv.lon[ds_readv.lon < 180]) + 0.58) \
                                .reset_index(drop=True).squeeze()
        ds_readv = ds_readv.swap_dims({'dim_0': 'lon'}).drop('dim_0')

        # Add readv to modeled RSL at locations with data
        ##### Need to fix this, as currently slice does not acknowledge new coords #########
        ds_readv = ds_readv.sel(age=slice(tmax, tmin),
                                lon=slice(df_place.lon.min() + 180 - 2,
                                          df_place.lon.max() + 180 + 2),
                                lat=slice(df_place.lat.max() + 2,
                                          df_place.lat.min() - 2))
        return ds_readv

    #Make deterministic readvance fingerprint
    start, end = 6100, 3000
    maxscale = 2.25
    ds_readv = make_fingerprint(start, end, maxscale)


    ####################  Build  GIA models 	#######################
    #################### ---------------------- #######################

    #Use either glac1d or ICE6G
    def build_dataset(path, ice_model):
        """download model runs from local directory."""
        files = f'{path + ice_model}/*.nc'
        basefiles = glob.glob(files)
        modelrun = [
           key.split('output_', 1)[1][:-3].replace('.', '_')
        for key in basefiles]
       dss = xr.open_mfdataset(files,
                            chunks=None,
                            concat_dim='modelrun',
                            combine='nested')
       lats, lons, times = dss.LAT.values[0], dss.LON.values[
           0], dss.TIME.values[0]
       ds = dss.drop(['LAT', 'LON', 'TIME'])
       ds = ds.assign_coords(lat=lats,
                          lon=lons,
                          time=times,
                          modelrun=modelrun).rename({
                              'time': 'age',
                              'RSL': 'rsl'})
       return ds

   def one_mod(path, ice_model):
       """Organize model runs into xarray dataset."""
       path1 = path + f'{ice_model[0]}/output_'
       path2 = path + f'{ice_model[1]}/output_'

       ds1 = build_dataset(path, ice_model[0])
       ds2 = build_dataset(path, ice_model[1])
       ds2 = ds2.interp(age=ds1.age, lat=ds1.lat, lon=ds1.lon)

       ds = xr.concat([ds1, ds2], dim='modelrun')

       ds['age'] = ds['age'] * 1000
       ds = ds.roll(lon=256, roll_coords=True)
       ds.coords['lon'] = pd.DataFrame((ds.lon[ds.lon >= 180] - 360)- 0.12 ) \
                            .append(pd.DataFrame(ds.lon[ds.lon < 180]) + 0.58) \
                            .reset_index(drop=True).squeeze()
       ds.coords['lat'] = ds.lat[::-1]
       ds = ds.swap_dims({'dim_0': 'lon'}).drop('dim_0')

       return ds

    #make composite of a bunch of GIA runs, i.e. GIA prior
    # path = f'data/{ice_model}/output_'
    path = f'data/'
    ds_sliced_in = one_mod(path,ice_model).rsl
    ds_sliced = ds_sliced_in.assign_coords({'lat':ds_sliced_in.lat.values[::-1]}).sel(
            age=slice(tmax, tmin),
            lon=slice(df_place.lon.min() - 2,
                      df_place.lon.max() + 2),
            lat=slice(df_place.lat.max() + 2,
                      df_place.lat.min() - 2))
    ds_area = ds_sliced.mean(dim='modelrun').load().to_dataset().interp(
                                            age=ds_readv.age, lon=ds_readv.lon, lat=ds_readv.lat)
    ds_areastd = ds_sliced.std(dim='modelrun').load().to_dataset().interp(
                                            age=ds_readv.age, lon=ds_readv.lon, lat=ds_readv.lat)

    #sample each model at points where we have RSL data
    def ds_select(ds):
        return ds.rsl.sel(age=[row.age],
                          lon=[row.lon],
                          lat=[row.lat],
                          method='nearest').squeeze().values

    #select points at which RSL data exists
    for i, row in df_place.iterrows():
        df_place.loc[i, 'rsl_realresid'] = df_place.rsl[i] - ds_select(ds_area)
        df_place.loc[i, 'rsl_giaprior'] = ds_select(ds_area)
        df_place.loc[i, 'rsl_giaprior_std'] = ds_select(ds_areastd)

    print('number of datapoints = ', df_place.shape)


    ##################	  RUN GP REGRESSION 	#######################
    ##################  --------------------	 ######################
    start = time.time()

    def run_gpr():

        Data = Tuple[tf.Tensor, tf.Tensor]
        likelihood = df_place.rsl_er_max.ravel()**2 # + df_place.rsl_giaprior_std.ravel()**2  # here we define likelihood

        class GPR_diag(gpf.models.GPModel):
            r"""
            Gaussian Process Regression.
            This is a vanilla implementation of GP regression with a pointwise Gaussian
            likelihood.  Multiple columns of Y are treated independently.
            The log likelihood of this models is sometimes referred to as the 'marginal log likelihood',
            and is given by
            .. math::
               \log p(\mathbf y \,|\, \mathbf f) =
                    \mathcal N\left(\mathbf y\,|\, 0, \mathbf K + \sigma_n \mathbf I\right)
            """
            def __init__(self,
                         data: Data,
                         kernel: Kernel,
                         mean_function: Optional[MeanFunction] = None,
                         likelihood=likelihood):
                likelihood = gpf.likelihoods.Gaussian(variance=likelihood)
                _, y_data = data
                super().__init__(kernel,
                                 likelihood,
                                 mean_function,
                                 num_latent=y_data.shape[-1])
                self.data = data

            def log_likelihood(self):
                """
                Computes the log likelihood.
                """
                x, y = self.data
                K = self.kernel(x)
                num_data = x.shape[0]
                k_diag = tf.linalg.diag_part(K)
                s_diag = tf.convert_to_tensor(self.likelihood.variance)
                jitter = tf.cast(tf.fill([num_data], default_jitter()),
                                 'float64')  # stabilize K matrix w/jitter
                ks = tf.linalg.set_diag(K, k_diag + s_diag + jitter)
                L = tf.linalg.cholesky(ks)
                m = self.mean_function(x)

                # [R,] log-likelihoods for each independent dimension of Y
                log_prob = multivariate_normal(y, m, L)
                return tf.reduce_sum(log_prob)

            def predict_f(self,
                          predict_at: tf.Tensor,
                          full_cov: bool = False,
                          full_output_cov: bool = False):
                r"""
                This method computes predictions at X \in R^{N \x D} input points
                .. math::
                    p(F* | Y)
                where F* are points on the GP at new data points, Y are noisy observations at training data points.
                """
                x_data, y_data = self.data
                err = y_data - self.mean_function(x_data)

                kmm = self.kernel(x_data)
                knn = self.kernel(predict_at, full=full_cov)
                kmn = self.kernel(x_data, predict_at)

                num_data = x_data.shape[0]
                s = tf.linalg.diag(tf.convert_to_tensor(
                    self.likelihood.variance))  #changed from normal GPR

                conditional = gpf.conditionals.base_conditional
                f_mean_zero, f_var = conditional(
                    kmn, kmm + s, knn, err, full_cov=full_cov,
                    white=False)  # [N, P], [N, P] or [P, N, N]
                f_mean = f_mean_zero + self.mean_function(predict_at)
                return f_mean, f_var


        def normalize(df):
            return np.array((df - df.mean()) / df.std()).reshape(len(df), 1)


        def denormalize(y_pred, df):
            return np.array((y_pred * df.std()) + df.mean())


        def bounded_parameter(low, high, param):
            """Make parameter tfp Parameter with optimization bounds."""
            affine = tfb.AffineScalar(shift=tf.cast(low, tf.float64),
                                      scale=tf.cast(high - low, tf.float64))
            sigmoid = tfb.Sigmoid()
            logistic = tfb.Chain([affine, sigmoid])
            parameter = gpf.Parameter(param, transform=logistic, dtype=tf.float64)
            return parameter


        class HaversineKernel_Matern52(gpf.kernels.Matern52):
            """
            Isotropic Matern52 Kernel with Haversine distance instead of euclidean distance.
            Assumes n dimensional data, with columns [latitude, longitude] in degrees.
            """
            def __init__(
                self,
                lengthscale=1.0,
                variance=1.0,
                active_dims=None,
            ):
                super().__init__(
                    active_dims=active_dims,
                    variance=variance,
                    lengthscale=lengthscale,
                )

            def haversine_dist(self, X, X2):
                pi = np.pi / 180
                f = tf.expand_dims(X * pi, -2)  # ... x N x 1 x D
                f2 = tf.expand_dims(X2 * pi, -3)  # ... x 1 x M x D
                d = tf.sin((f - f2) / 2)**2
                lat1, lat2 = tf.expand_dims(X[:, 0] * pi, -1), \
                            tf.expand_dims(X2[:, 0] * pi, -2)
                cos_prod = tf.cos(lat2) * tf.cos(lat1)
                a = d[:, :, 0] + cos_prod * d[:, :, 1]
                c = tf.asin(tf.sqrt(a)) * 6371 * 2
                return c

            def scaled_squared_euclid_dist(self, X, X2):
                """
                Returns (dist(X, X2ᵀ)/lengthscales)².
                """
                if X2 is None:
                    X2 = X
                dist = da.square(self.haversine_dist(X, X2) / self.lengthscale)
        #             dist = tf.convert_to_tensor(dist)
                return dist


        class HaversineKernel_Matern32(gpf.kernels.Matern32):
            """
            Isotropic Matern52 Kernel with Haversine distance instead of euclidean distance.
            Assumes n dimensional data, with columns [latitude, longitude] in degrees.
            """
            def __init__(
                self,
                lengthscale=1.0,
                variance=1.0,
                active_dims=None,
            ):
                super().__init__(
                    active_dims=active_dims,
                    variance=variance,
                    lengthscale=lengthscale,
                )

            def haversine_dist(self, X, X2):
                pi = np.pi / 180
                f = tf.expand_dims(X * pi, -2)  # ... x N x 1 x D
                f2 = tf.expand_dims(X2 * pi, -3)  # ... x 1 x M x D
                d = tf.sin((f - f2) / 2)**2
                lat1, lat2 = tf.expand_dims(X[:, 0] * pi, -1), \
                            tf.expand_dims(X2[:, 0] * pi, -2)
                cos_prod = tf.cos(lat2) * tf.cos(lat1)
                a = d[:, :, 0] + cos_prod * d[:, :, 1]
                c = tf.asin(tf.sqrt(a)) * 6371 * 2
                return c

            def scaled_squared_euclid_dist(self, X, X2):
                """
                Returns (dist(X, X2ᵀ)/lengthscales)².
                """
                if X2 is None:
                    X2 = X
                dist = tf.square(self.haversine_dist(X, X2) / self.lengthscale)

                return dist


        ########### Section to Run GPR######################
        ##################################3#################

        # Input space, rsl normalized to zero mean, unit variance
        X = np.stack((df_place['lon'], df_place['lat'], df_place['age']), 1)
        RSL = normalize(df_place.rsl_realresid)

        #define kernels  with bounds

        k1 = HaversineKernel_Matern32(active_dims=[0, 1])
        k1.lengthscale = bounded_parameter(5000, 30000, 10000)  #hemispheric space
        k1.variance = bounded_parameter(0.1, 100, 2)

        k2 = HaversineKernel_Matern32(active_dims=[0, 1])
        k2.lengthscale = bounded_parameter(1, 5000, 1000)  #GIA space
        k2.variance = bounded_parameter(0.1, 100, 2)

        k3 = gpf.kernels.Matern32(active_dims=[2])  #GIA time
        k3.lengthscale = bounded_parameter(8000, 20000, 10000)
        k3.variance = bounded_parameter(0.1, 100, 1)

        k4 = gpf.kernels.Matern32(active_dims=[2])  #shorter time
        k4.lengthscale = bounded_parameter(1, 8000, 1000)
        k4.variance = bounded_parameter(0.1, 100, 1)

        k5 = gpf.kernels.White(active_dims=[2])
        k5.variance = bounded_parameter(0.01, 100, 1)

        kernel = (k1 * k3) + (k2 * k4) + k5

        #build & train model
        m = GPR_diag((X, RSL), kernel=kernel, likelihood=likelihood)
        print('model built, time=', time.time() - start)

        @tf.function(autograph=False)
        def objective():
            return - m.log_marginal_likelihood()

        o = gpf.optimizers.Scipy()
        o.minimize(objective, variables=m.trainable_variables)
        print('model minimized, time=', time.time() - start)

        # output space
        nout = 70
        lat = np.linspace(min(ds_area.lat), max(ds_area.lat), nout)
        lon = np.linspace(min(ds_area.lon), max(ds_area.lon), nout)
        ages = ages_lgm[(ages_lgm < tmax) & (ages_lgm > tmin)]
        xyt = np.array(list(product(lon, lat, ages)))

        #query model & renormalize data
        y_pred, var = m.predict_f(xyt)
        y_pred_out = denormalize(y_pred, df_place.rsl_realresid)

        #reshape output vectors
        Zp = np.array(y_pred_out).reshape(nout, nout, len(ages))
        varp = np.array(var).reshape(nout, nout, len(ages))

        #print kernel details
        print_summary(m, fmt='notebook')
        print('time elapsed = ', time.time() - start)

        print('negative log marginal likelihood =',
              m.neg_log_marginal_likelihood().numpy())

        loglikelist = []
        loglikelist.append(m.neg_log_marginal_likelihood().numpy())


        ##################	  INTERPOLATE MODELS 	#######################
        ##################  --------------------	 ######################

        # turn GPR output into xarray dataarray
        da_zp = xr.DataArray(Zp, coords=[lon, lat, ages],
                             dims=['lon', 'lat',
                                   'age']).transpose('age', 'lat', 'lon')
        da_varp = xr.DataArray(varp,
                               coords=[lon, lat, ages],
                               dims=['lon', 'lat',
                                     'age']).transpose('age', 'lat', 'lon')

        def interp_likegpr(ds):
            return ds.rsl.load().transpose().interp_like(da_zp)

        #interpolate all models onto GPR grid
        da_giapriorinterp = interp_likegpr(ds_area)
        ds_giapriorinterp = ds_area.interp(age=ages)
        da_giapriorinterpstd = interp_likegpr(ds_areastd)

        # add total prior RSL back into GPR
        da_priorplusgpr = da_zp + da_giapriorinterp

        return ages, da_zp, da_giapriorinterp, da_priorplusgpr, da_varp, loglikelist

    ages, da_zp, da_giapriorinterp, da_priorplusgpr, da_varp, loglikelist = run_gpr()
    ##################	  	 SAVE NETCDFS 	 	#######################
    ##################  --------------------	 ######################

    path_gen = f'{ages[0]}_{ages[-1]}_modelaverage_{place}'
    da_zp.to_netcdf('output/' + path_gen + '_dazp')
    da_giapriorinterp.to_netcdf('output/' + path_gen + '_giaprior')
    da_priorplusgpr.to_netcdf('output/' + path_gen + '_posterior')
    da_varp.to_netcdf('output/' + path_gen + '_gpvariance')

    #store log likelihood in dataframe
    df_out = pd.DataFrame({'modelrun': 'average_likelihood',
                     'log_marginal_likelihood': loglikelist})

    writepath = f'output/{path_gen}_loglikelihood'

    df_out.to_csv(writepath, index=False)
    df_likes = pd.read_csv(writepath)

if __name__ == '__main__':
    readv()
