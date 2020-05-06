import glob
from matplotlib.colors import Normalize
import pandas as pd
import scipy.io as io
import xarray as xr
import numpy as np

from itertools import product

import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pandas.io.json import json_normalize
from df2gspread import df2gspread as d2g


import gpflow
import gpflow as gpf
from gpflow.utilities import print_summary, positive
from gpflow.logdensities import multivariate_normal
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, RegressionData, MeanAndVariance, GPModel
from gpflow.base import Parameter 
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.config import default_jitter, default_float

from typing import Optional, Tuple

from matplotlib import colors, cm



def load_nordata_fromsheet(sheet):
    
    """Connect to google sheet & load norwegian RSL data."""
    
    
    #connect to service account
    scope = ['https://spreadsheets.google.com/feeds'] 
    credentials = ServiceAccountCredentials.from_json_keyfile_name('../fentides/careful-granite-273616-38728a8743ba.json', scope) 
    gc = gspread.authorize(credentials)

    #access data
    spreadsheet_key = '10glCyv79FkDVfIKDBzfalS6vnzOosR8xNfRbXybK7BY'
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet(sheet)
    table = worksheet.get_all_values()

    ##Convert table data into a dataframe
    df = pd.DataFrame(table[2:], columns=table[2]).drop([0, 1, 2]).reset_index()
    df = df[['Column heading',
             'Latitude', 
             'Longitude', 
             'Type', 
             'RSL', 
             'RSL 2σ Uncertainty +',
             'RSL 2σ Uncertainty -',
            'Age',
            'Age 2 sigma uncertainty +',
            'Age 2 sigma uncertainty -',]].replace('', value=np.nan).dropna(how='any')
    df_nor = df[df.Type == '0'].drop(['Type', 'Column heading'], axis=1).replace('%','',regex=True).astype('float')
    df_nor.rename(columns={'Column heading':'lakename',
                           'Latitude':'lat',
                           'Longitude':'lon',
                           'RSL':'rsl',
                           'RSL 2σ Uncertainty +':'rsl_er_max',
                           'RSL 2σ Uncertainty -':'rsl_er_min',
                           'Age':'age',
                           'Age 2 sigma uncertainty +':'age_er_max',
                           'Age 2 sigma uncertainty -':'age_er_min'
                          }, inplace=True
                 )

    df_nor['rsl_er'] = (df_nor.rsl_er_max + df_nor.rsl_er_min)/2
    df_nor['age_er'] = (df_nor.age_er_max + df_nor.age_er_min)/2
    return df_nor



def make_mod(ice_model, lith, ages, extent, zeros=False):
        
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
    
    #slice dataset to location
    ds = ds.rsl.sel(age=slice(ages[0], ages[-1]),
            lon=slice(extent[0] - 2, extent[1] + 2),
            lat=slice(extent[3] + 2, extent[2] - 2))
    
    if zeros:
    #add present-day RSL at zero to the GIA model
        ds_zeros = xr.zeros_like(ds)[:,0] + 0.01
        ds_zeros['age'] = 0.1
        ds_zeros = ds_zeros.expand_dims('age').transpose('modelrun','age', 'lon', 'lat')
        ds = xr.concat([ds, ds_zeros], 'age')
    else:
        pass
    
    return ds


def add_presday_0s(df_place, nout):
    
    """ Prescribe present-day RSL to zero by adding zero points at t=10 yrs."""
    
    #prescribe present-day RSL to zero
    preslocs1 = df_place.groupby(['lat', 'lon'])[['rsl', 
                                                 'rsl_er_max',
                                                 'age']].nunique().reset_index()[['lat',
                                                                                  'lon']][::int(50/nout)]

    # make more present day points at zero on an nout/nout grid
    lat = np.linspace(min(df_place.lat), max(df_place.lat), nout)
    lon = np.linspace(min(df_place.lon), max(df_place.lon), nout)
    xy = np.array(list(product(lon, lat)))[::int(nout/2)]

    preslocs2 = pd.DataFrame(xy, columns=['lon', 'lat'])

    preslocs = pd.concat([preslocs1, preslocs2]).reset_index(drop=True)

    preslocs['rsl'] = 0.1
    preslocs['rsl_er'] = 0.1
    preslocs['rsl_er_max'] = 0.1
    preslocs['rsl_er_min'] = 0.1
    preslocs['age_er'] = 1
    preslocs['age_er_max'] = 1
    preslocs['age_er_min'] = 1
    preslocs['age'] = 10

    df_place = pd.concat([df_place, preslocs]).reset_index(drop=True)
    return df_place


def import_rsls(path, df_nor, tmin, tmax, extent):
    
    """ import khan Holocene RSL database from csv."""
    
    df = pd.read_csv(path, encoding="ISO-8859-15", engine='python')
    df = df.replace('\s+', '_', regex=True).replace('-', '_', regex=True).\
            applymap(lambda s:s.lower() if type(s) == str else s)
    df.columns = df.columns.str.lower()
    df.rename_axis('index', inplace=True)
    df = df.rename({'latitude': 'lat', 'longitude': 'lon'}, axis='columns')
    dfind, dfterr, dfmar = df[(df.type == 0)
                              & (df.age > 0)], df[df.type == 1], df[df.type == -1]

    dfind = pd.concat([dfind, df_nor]).reset_index(drop=True)

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
    return df_place

def xarray_template(filename, ages, extent):
    
    """make template for xarray interpolation"""
    
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
    return ds_template


def cmap_codes(name, number):
    
    " make colormap hexcodes"
    
    cmap = cm.get_cmap(name, number) 
    hexcodes = []
    for i in range(cmap.N): 
        hexcodes.append(colors.rgb2hex(cmap(i)[:3]))
    return hexcodes


class MidpointNormalize(Normalize):
    
    """Normalise the colorbar.  e.g. norm=MidpointNormalize(mymin, mymax, 0.)"""
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
    
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
        lengthscales=None,
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
    
    
class GPR_new(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by
    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
#         noise_variance: float = 1.0,
        noise_variance: list = [],
    ):
        
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        K = self.kernel(X)
        num_data = X.shape[0]
        k_diag = tf.linalg.diag_part(K)
#         s_diag = tf.fill([num_data], self.likelihood.variance)
        s_diag = tf.convert_to_tensor(self.likelihood.variance)

        
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]

#         s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))
        s = tf.linalg.diag(tf.convert_to_tensor(self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

    
def locs_with_enoughsamples(df_place, place, number):
    """make new dataframe, labeled, of sites with [> number] measurements"""
    df_lots = df_place.groupby(['lat',
                                'lon']).filter(lambda x: len(x) > number)

    df_locs = []
    for i, group in enumerate(df_lots.groupby(['lat', 'lon'])):
        singleloc = group[1].copy()
        singleloc['location'] = place
        singleloc['locnum'] = place + '_site' + str(
            i)  # + singleloc.reset_index().index.astype('str')
        df_locs.append(singleloc)
    df_locs = pd.concat(df_locs)

    return df_locs

def ds_select(ds, row):
    
    """ Slice GIA or GPR xarray dataset by rows in RSL pandas dataframe."""
    
    return ds.rsl.sel(age=[row.age],
                      lon=[row.lon],
                      lat=[row.lat],
                      method='nearest').squeeze().values


def interp_likegpr(ds_mean, ds_std, da_zp):
    mean = ds_mean.load().interp_like(da_zp)
    std = ds_std.load().interp_like(da_zp)
    
    return mean, std

def makexyt(ds_giamean, nout, ages):
    
    lat = np.linspace(min(ds_giamean.lat), max(ds_giamean.lat), nout)
    lon = np.linspace(min(ds_giamean.lon), max(ds_giamean.lon), nout)
    
    xyt = np.array(list(product(lon, lat, ages)))
    return(lat, lon, xyt)
    

def gpr_predict_f(lat, lon, xyt, nout, ds_giamean, m, ages, df_place):
    
    """ predict GPR latent function at output locations. """
    
    #query model
    y_pred, var = m.predict_f(xyt)   

    
    # renormalize data
    y_pred_out = denormalize(y_pred, df_place.rsl_realresid)

    # reshape output vectors
    Zp = np.array(y_pred_out).reshape(nout, nout, len(ages))
    varp = np.array(var).reshape(nout, nout, len(ages))
    
    #transform output into xarray dataarrays
    da_zp = xr.DataArray(Zp, coords=[lon, lat, ages],
                     dims=['lon', 'lat','age']).transpose('age', 'lat', 'lon')
    da_varp = xr.DataArray(varp, coords=[lon, lat, ages],
                     dims=['lon', 'lat','age']).transpose('age', 'lat', 'lon')
    
    return y_pred, da_zp, da_varp
