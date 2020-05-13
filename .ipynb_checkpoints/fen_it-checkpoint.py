# uses conda environment gpflow6_0

# #!/anaconda3/envs/gpflow6_0/env/bin/python

# from memory_profiler import profile

# generic
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as io
from itertools import product
import glob
import time


# gpflow
import gpflow
import gpflow as gpf
from gpflow.utilities import print_summary
from gpflow.logdensities import multivariate_normal
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from typing import Optional, Tuple
from gpflow.config import default_jitter, default_float


from gpflow.utilities import print_summary, positive
from gpflow.models.model import InputData, RegressionData, MeanAndVariance, GPModel
from gpflow.base import Parameter
from gpflow.models.training_mixins import InternalDataTrainingLossMixin



# tensorflow
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
import argparse

# @profile

def readv():


    parser = argparse.ArgumentParser(description='import vars via c-line')
    parser.add_argument("--mod", default='d6g_h6g_')
    parser.add_argument("--lith", default='l71C')
    parser.add_argument("--um", default="p2")
    parser.add_argument("--lm", default="3")
    parser.add_argument("--tmax", default=1600)
    parser.add_argument("--tmin", default=1)
    parser.add_argument("--place", default="fennoscandia")

    args = parser.parse_args()
    ice_model = args.mod
    lith = args.lith
    um = args.um
    lm = args.lm
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place


    ages_lgm = np.arange(100, 26000, 100)[::-1]
    ages = np.arange(tmin, tmax, 100)[::-1]

    locs = {'europe': [-20, 15, 35, 70],
            'atlantic':[-85,50, 25, 73],
            'fennoscandia': [-15, 50, 45, 75],
            'world': [-175, 175, -85, 85]
           }
    extent = locs[place]

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
    df_slice = dfind[(dfind.age > tmin) & (dfind.age < tmax) &
                     (dfind.lon > extent[0])
                     & (dfind.lon < extent[1])
                     & (dfind.lat > extent[2])
                     & (dfind.lat < extent[3])][[
                         'lat', 'lon', 'rsl', 'rsl_er_max', 'rsl_er_min', 'age', 'age_er_max', 'age_er_min']]
    df_slice['rsl_er'] = (df_slice.rsl_er_max + df_slice.rsl_er_min)/2
    df_slice['age_er'] = (df_slice.age_er_max + df_slice.age_er_min)/2
    df_place = df_slice.copy()

    #prescribe present-day RSL to zero
    preslocs = df_place.groupby(['lat', 'lon'])[['rsl', 'rsl_er_max', 'age']].nunique().reset_index()[::2]
    preslocs['rsl'] = 0.01
    preslocs['rsl_er'] = 0.01
    preslocs['rsl_er_max'] = 0.01
    preslocs['rsl_er_min'] = 0.01
    preslocs['age_er'] = 1
    preslocs['age_er_max'] = 1
    preslocs['age_er_min'] = 1
    preslocs['age'] = 10
    df_place = pd.concat([df_place, preslocs]).reset_index(drop=True)

    ####################  Make xarray template  #######################
    #################### ---------------------- #######################

    filename = 'data/WAISreadvance_VM5_6ka_1step.mat'
    template = io.loadmat(filename, squeeze_me=True)

    template = xr.Dataset({'rsl': (['lat', 'lon', 'age'], np.zeros((256, 512, len(ages))))},
                         coords={'lon': template['lon_out'],
                                 'lat': template['lat_out'],
                                 'age': ages})
    template.coords['lon'] = pd.DataFrame((template.lon[template.lon >= 180] - 360)- 0.12) \
                            .append(pd.DataFrame(template.lon[template.lon < 180]) + 0.58) \
                            .reset_index(drop=True).squeeze()
    ds_template = template.swap_dims({'dim_0': 'lon'}).drop('dim_0').sel(lon=slice(extent[0] + 180 - 2,
                                                                                   extent[1] + 180 + 2),
                                                                         lat=slice(extent[3] + 2,
                                                                                   extent[2] - 2)).rsl
    #add more data at zero locations
    nout = 50
    lat = np.linspace(min(df_place.lat), max(df_place.lat), nout)
    lon = np.linspace(min(df_place.lon), max(df_place.lon), nout)
    xy = np.array(list(product(lon, lat)))[::15]

    morepreslocs = pd.DataFrame(xy, columns=['lon', 'lat'])
    morepreslocs['rsl'] = 0.01 + np.zeros(len(xy))
    morepreslocs['rsl_er_max'] = 0.01 + np.zeros(len(xy))
    morepreslocs['age'] = 100 + np.zeros(len(xy))

    df_place = pd.concat([df_place, preslocs, morepreslocs]).reset_index(drop=True)
    df_place.shape



    ####################    Load GIA datasets   #######################
    #################### ---------------------- #######################
    def make_mod(ice_model, lith):
        """combine model runs from local directory into xarray dataset."""

        path = f'data/{ice_model}/output_{ice_model}{lith}'
        files = f'{path}*.nc'
        basefiles = glob.glob(files)
        modelrun = [key.split('output_', 1)[1][:-3].replace('.', '_') for key in basefiles]
        dss = xr.open_mfdataset(files,
                                chunks=None,
                                concat_dim='modelrun',
                                combine='nested')
        lats, lons, times = dss.LAT.values[0], dss.LON.values[0], dss.TIME.values[0]
        ds = dss.drop(['LAT', 'LON', 'TIME']).assign_coords(lat=lats,
                                                            lon=lons,
                                                            time=times * 1000,
                                                            modelrun=modelrun).rename({
                                                                          'time': 'age', 'RSL': 'rsl'})
        ds = ds.chunk({'lat': 10, 'lon': 10})
        ds = ds.roll(lon=256, roll_coords=True)
        ds.coords['lon'] = pd.DataFrame((ds.lon[ds.lon >= 180] - 360)- 0.12 ) \
                                        .append(pd.DataFrame(ds.lon[ds.lon < 180]) + 0.58) \
                                        .reset_index(drop=True).squeeze()
        ds = ds.swap_dims({'dim_0': 'lon'}).drop('dim_0')

        print(ds)
        print(extent[0] - 2, extent[1] + 2, extent[3] + 2, extent[2] - 2)

        #slice dataset to location
        #ds = ds.rsl.sel(age=slice(ages[0], ages[-1]),
         #       lon=slice(extent[0] - 2, extent[1] + 2),
          #      lat=slice(extent[3] + 2, extent[2] - 2))

        ds = ds.rsl.sel(age=slice(ages[0], ages[-1]))
        # ds = ds.sel(lon=slice(extent[0] - 2, extent[1] + 2))
        ds = ds.sel(lat=slice(extent[3] + 2, extent[2] - 2))

        #add present-day RSL at zero to the GIA model
        ds_zeros = xr.zeros_like(ds)[:,0] + 0.01
        ds_zeros['age'] = 0.1
        ds_zeros = ds_zeros.expand_dims('age').transpose('modelrun','age', 'lon', 'lat')
        ds = xr.concat([ds, ds_zeros], 'age')

        return ds

    ds = make_mod(ice_model, lith)

    #make mean of runs
    ds_giamean = ds.mean(dim='modelrun').load().chunk((-1,-1,-1)).interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()
    ds_giastd = ds.std(dim='modelrun').load().chunk((-1,-1,-1)).interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()

    #sample each model at points where we have RSL data
    def ds_select(ds):
        return ds.rsl.sel(age=[row.age],
                          lon=[row.lon],
                          lat=[row.lat],
                          method='nearest').squeeze().values

    #select points at which RSL data exists
    for i, row in df_place.iterrows():
        df_place.loc[i, 'rsl_realresid'] = df_place.rsl[i] - ds_select(ds_giamean)
        df_place.loc[i, 'rsl_giaprior'] = ds_select(ds_giamean)
        df_place.loc[i, 'rsl_giaprior_std'] = ds_select(ds_giastd)

    print('number of datapoints = ', df_place.shape)

    ##################    RUN GP REGRESSION     #######################
    ##################  --------------------     ######################
    start = time.time()




    class GPR_diag_(gpf.models.GPModel):
        r"""
        Gaussian Process Regression.
        This is a vanilla implementation of GP regression with a pointwise Gaussian
        likelihood.  Multiple columns of Y are treated independently.
        The log likelihood of this model is sometimes referred to as the 'marginal log likelihood',
        and is given by
        .. math::
           \log p(\mathbf y \,|\, \mathbf f) =
                \mathcal N\left(\mathbf y\,|\, 0, \mathbf K + \sigma_n \mathbf I\right)
        """
        def __init__(self,
                     data: Data,
                     kernel: Kernel,
                     mean_function: Optional[MeanFunction] = None,
                     likelihood=noise_variance):
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

            k_diag = tf.linalg.diag_part(kmm)
            s_diag = tf.convert_to_tensor(self.likelihood.variance)
            jitter = tf.cast(tf.fill([num_data], default_jitter()),
                             'float64')  # stabilize K matrix w/jitter
            ks = tf.linalg.set_diag(kmm, k_diag + s_diag + jitter)
            L = tf.linalg.cholesky(ks)

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

        sigmoid = tfb.Sigmoid(low=tf.cast(low, tf.float64),
                              high=tf.cast(high, tf.float64),
                             name='sigmoid')
        parameter = gpf.Parameter(param, transform=sigmoid, dtype=tf.float64)
        return parameter

    class HaversineKernel_Matern32(gpf.kernels.Matern32):
        """
        Isotropic Matern52 Kernel with Haversine distance instead of euclidean distance.
        Assumes n dimensional data, with columns [latitude, longitude] in degrees.
        """
        def __init__(
            self,
            lengthscales=1.0,
            variance=1.0,
            active_dims=None,
        ):
            super().__init__(
                active_dims=active_dims,
                variance=variance,
                lengthscales=lengthscales,
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
            dist = tf.square(self.haversine_dist(X, X2) / self.lengthscales)

            return dist


    ########### Section to Run GPR######################
    ####################################################

    # Input space, rsl normalized to zero mean, unit variance
    X = np.stack((df_place['lon'], df_place['lat'], df_place['age']), 1)

    RSL = normalize(df_place.rsl_realresid)

    #define kernels  with bounds
    k1 = HaversineKernel_Matern32(active_dims=[0, 1])
    k1.lengthscales = bounded_parameter(100, 60000, 300)  #hemispheric space
    k1.variance = bounded_parameter(0.1, 100, 2)

    k2 = HaversineKernel_Matern32(active_dims=[0, 1])
    k2.lengthscales = bounded_parameter(1, 6000, 10)  #GIA space
    k2.variance = bounded_parameter(0.1, 100, 2)

    k3 = gpf.kernels.Matern32(active_dims=[2])  #GIA time
    k3.lengthscales = bounded_parameter(0.1, 20000, 1000)
    k3.variance = bounded_parameter(0.1, 100, 1)

    k4 = gpf.kernels.Matern32(active_dims=[2])  #shorter time
    k4.lengthscales = bounded_parameter(1, 6000, 100)
    k4.variance = bounded_parameter(0.1, 100, 1)

    k5 = gpf.kernels.White(active_dims=[2])
    k5.variance = bounded_parameter(0.01, 100, 1)

    kernel = (k1 * k3)  + k5 # + (k4 * k2)

    #build & train model
    m = GPR_new((X, RSL), kernel=kernel, noise_variance=noise_variance)
    print('model built, time=', time.time() - start)

    @tf.function(autograph=False)
    def objective():
        return - m.log_marginal_likelihood()

    o = gpf.optimizers.Scipy()
    o.minimize(objective, variables=m.trainable_variables, method='trust-constr', options={'maxiter': 2000, 'disp': True, 'verbose': 1})
    print('model minimized, time=', time.time() - start)

    # output space
    nout = 20
    lat = np.linspace(min(ds_giamean.lat), max(ds_giamean.lat), nout)
    lon = np.linspace(min(ds_giamean.lon), max(ds_giamean.lon), nout)
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
          m.log_marginal_likelihood().numpy())

#     loglike = []
#     loglike.append(m.neg_log_marginal_likelihood().numpy())
    loglike = m.log_marginal_likelihood().numpy()


    ##################    INTERPOLATE MODELS    #######################
    ##################  --------------------     ######################

    # turn GPR output into xarray dataarray
    da_zp = xr.DataArray(Zp, coords=[lon, lat, ages],
                         dims=['lon', 'lat','age']).transpose('age', 'lat', 'lon')
    da_varp = xr.DataArray(varp, coords=[lon, lat, ages],
                           dims=['lon', 'lat', 'age']).transpose('age', 'lat', 'lon')

    def interp_likegpr(ds):
        return ds.load().interp_like(da_zp)

    #interpolate all models onto GPR grid
    ds_giapriorinterp = interp_likegpr(ds_giamean)
    ds_giapriorinterpstd = interp_likegpr(ds_giastd)

    # add total prior RSL back into GPR
    ds_priorplusgpr = da_zp + ds_giapriorinterp

    return ages, da_zp, ds_giapriorinterpstd, ds_giapriorinterp, ds_priorplusgpr, da_varp, loglike

    ages, da_zp, ds_giapriorinterpstd, ds_giapriorinterp, ds_priorplusgpr, da_varp, loglike = run_gpr()

    ##################           SAVE NETCDFS           #######################
    ##################  --------------------     ######################

    path_gen = f'output/{place}_{ice_model}{ages[0]}_{ages[-1]}'
    da_zp.to_netcdf(path_gen + '_dazp')
    da_giapriorinterp.to_netcdf(path_gen + '_giaprior')
    da_giapriorinterpstd.to_netcdf(path_gen + '_giapriorstd')
    da_priorplusgpr.to_netcdf(path_gen + '_posterior')
    da_varp.to_netcdf(path_gen + '_gpvariance')

    #store log likelihood in dataframe
    df_out = pd.DataFrame({'modelrun': 'average_likelihood',
                     'log_marginal_likelihood': loglikelist})

    writepath = f'output/{path_gen}_loglikelihood'

    df_out.to_csv(writepath, index=False)
    df_likes = pd.read_csv(writepath)

if __name__ == '__main__':
    readv()





