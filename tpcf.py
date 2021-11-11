"""
Tools for computing two-point correlation functions. 
Adopted from astromL two-point correlation module. 

Modified to be run in parallel. 

AUTHOR (modified)
Shyam Harimohan Menon (2020)
"""

import warnings
import numpy as np
from sklearn.neighbors import BallTree
from astroML.utils import check_random_state
# Check if scikit-learn's two-point functionality is available.
# This was added in scikit-learn version 0.14
try:
    from sklearn.neighbors import KDTree
    sklearn_has_two_point = True
except ImportError:
    import warnings
    sklearn_has_two_point = False

from joblib import Parallel, delayed
from itertools import repeat

def compute_tpcf(x,y,nobins=20,Rmin=None,Rmax=None):
    """
    Compute TPCF of star clusters in the galaxy
    Parameters
    ----------
    x : tuple
        x positions of star clusters
    y : tuple
        y positions of star clusters
    nobins: integer
        Number of bins to compute TPCF on
    Rmin: float
        minimum separation for TPCF bin
    Rmax: float
        maximum separation for TPCF bin


    """
    data = np.asarray((x,y),order='F').T
    #by default max radius
    if(Rmax is None):
        Rmax = np.max(np.sqrt(x**2+y**2))
    if(Rmin is None):
        Rmin = Rmax/1000.
    bins = np.logspace(np.log10(Rmin),np.log10(Rmax),nobins)
    corr,dcorr = bootstrap_two_point(data, bins, Nbootstrap=30,
                                return_bootstraps=False,
                                random_state=None,random_type='random')
    return bins,corr,dcorr

def uniform_sphere(RAlim, DEClim, size=1):
    """Draw a uniform sample on a sphere

    Parameters
    ----------
    RAlim : tuple
        select Right Ascension between RAlim[0] and RAlim[1]
        units are degrees
    DEClim : tuple
        select Declination between DEClim[0] and DEClim[1]
    size : int (optional)
        the size of the random arrays to return (default = 1)

    Returns
    -------
    RA, DEC : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to size.
    """
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)

    return RA, DEC

def fill_space(x,y,size=1):
    """
    Fill the cartesian area occupied by the star cluster positions
    Parameters
    ----------
    x : tuple
        x positions of star clusters
    y : tuple
        y positions of star clusters
    
    Returns
    -------
    xr,yr : ndarray
        the random sample

    """
    xlim = [np.min(x),np.max(x)]
    ylim = [np.min(y),np.max(y)]
    xr = xlim[0] + (xlim[1]-xlim[0])*np.random.random(size)
    yr = ylim[0] + (ylim[1]-ylim[0])*np.random.random(size)
    return xr,yr

def ra_dec_to_xyz(ra, dec):
    """Convert ra & dec to Euclidean points

    Parameters
    ----------
    ra, dec : ndarrays

    Returns
    x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)

def two_point(data, bins, method='standard',
              data_R=None, random_state=None):
    """Two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    method : string
        "standard" or "landy-szalay".
    data_R : array_like (optional)
        if specified, use this as the random comparison sample
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    n_samples, n_features = data.shape
    Nbins = len(bins) - 1

    # shuffle all but one axis to get background distribution
    if data_R is None:
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])     
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')

    factor = len(data_R) * 1. / len(data)

    if sklearn_has_two_point:
        # Fast two-point correlation functions added in scikit-learn v. 0.14
        KDT_D = KDTree(data)
        KDT_R = KDTree(data_R)

        counts_DD = KDT_D.two_point_correlation(data, bins)
        counts_RR = KDT_R.two_point_correlation(data_R, bins)

    else:
        warnings.warn("Version 0.3 of astroML will require scikit-learn "
                      "version 0.14 or higher for correlation function "
                      "calculations. Upgrade to sklearn 0.14+ now for much "
                      "faster correlation function calculations.")

        BT_D = BallTree(data)
        BT_R = BallTree(data_R)

        counts_DD = np.zeros(Nbins + 1)
        counts_RR = np.zeros(Nbins + 1)

        for i in range(Nbins + 1):
            counts_DD[i] = np.sum(BT_D.query_radius(data, bins[i],
                                                    count_only=True))
            counts_RR[i] = np.sum(BT_R.query_radius(data_R, bins[i],
                                                    count_only=True))

    DD = np.diff(counts_DD)
    RR = np.diff(counts_RR)

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor ** 2 * DD / RR - 1
    elif method == 'landy-szalay':
        if sklearn_has_two_point:
            counts_DR = KDT_R.two_point_correlation(data, bins)
        else:
            counts_DR = np.zeros(Nbins + 1)
            for i in range(Nbins + 1):
                counts_DR[i] = np.sum(BT_R.query_radius(data, bins[i],
                                                        count_only=True))
        DR = np.diff(counts_DR)

        corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    corr[RR_zero] = np.nan

    return corr



def bootstrap_two_point(data, bins, Nbootstrap=10,
                        method='standard', return_bootstraps=False,
                        random_state=None,random_type='random'):
    """Bootstrapped two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    Nbootstrap : integer
        number of bootstrap resamples to perform (default = 10)
    method : string
        "standard" or "landy-szalay".
    return_bootstraps: bool
        if True, return full bootstrapped samples
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background
    random_type : string
        type of random sample. Can be "uniform", "random" or "default".

    Returns
    -------
    corr, corr_err : ndarrays
        the estimate of the correlation function and the bootstrap
        error within each bin. shape = Nbins
    """

    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    if Nbootstrap < 2:
        raise ValueError("Nbootstrap must be greater than 1")

    n_samples, n_features = data.shape

    # get the baseline estimate
    corr = two_point(data, bins, method=method, random_state=rng)

    bootstraps = np.zeros((Nbootstrap, len(corr)))
    
    data_boot = np.zeros((Nbootstrap,n_samples,n_features))
    data_R = np.zeros((Nbootstrap,n_samples,n_features))
    for i in range(Nbootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        data_boot[i] = data[indices,:]
        if(random_type=='uniform'):
            ra_R, dec_R = uniform_sphere((min(data[:,0]), max(data[:,0])),
                                         (min(data[:,1]), max(data[:,1])),
                                         len(data[:,0]))
            random_arr = np.asarray((ra_R, dec_R), order='F').T
            data_R[i] = random_arr

        elif(random_type == 'random'):
            ra_R,dec_R = fill_space(data[:,0],data[:,1],size=len(data[:,0]))
            random_arr = np.asarray((ra_R, dec_R), order='F').T
            data_R[i] = random_arr

        elif(random_type == 'default'):
            data_Rdefault = data.copy()
            for i in range(n_features - 1):
                rng.shuffle(data_Rdefault[:, i])
            data_R[i] = data_Rdefault
                   
        else:
            raise ValueError("Random type is undefined.")
    results = Parallel(n_jobs=-1,prefer='processes',verbose=0)(map(delayed(two_point),
        data_boot,repeat(bins),repeat(method),data_R,repeat(rng)))
    bootstraps = results
    bootstraps = np.asarray(bootstraps)
    corr = np.mean(bootstraps, 0)
    corr_err = np.std(bootstraps, 0, ddof=1)
    

    if return_bootstraps:
        return corr, corr_err, bootstraps
    else:
        return corr, corr_err