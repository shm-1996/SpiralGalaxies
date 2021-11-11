"""
Routines to plot stuff.

AUTHOR
Shyam Harimohan Menon (2020)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
mpl.rc_file('./matplotlibrc',use_default_template=False)

def plot_clusters(x,y,save=False,filename=None,axs=None,**kwargs):
    """
    Plot spatial distribution of clusters in the galaxy

   Parameters
    ----------
    x : ndarray
        x Positions of star clusters
    y : ndarray
        y Positions of star clusters
    save : boolean
        Flag to save the plot, else just show.
    axs : Matplotlib axes instance
        Plot in this axis. Creates a new axis if not provided. 
    filename : string
        File to save to, else default filename 
    **kwargs: 
        Other optional parameters for plotting that go directly into the call for plt.plot.
    
    Returns
    -------
    None
    
    """ 

    #Create fig and axs if not provided
    if(axs is None):
        axsProvided = False
        fig,axs = plt.subplots(ncols=1)
    else:
        axsProvided = True

    #Use some plot defaults if not explicitly provided by user
    lw = kwargs.pop('lw',0.2)
    fmt = kwargs.pop('fmt','o')
    alpha = kwargs.pop('alpha',0.8)
    ms = kwargs.pop('ms',3)
    ms = kwargs.pop('markersize',3) #alias name for parameter
    mec = kwargs.pop('mec','#696DFF')
    mec = kwargs.pop('markeredgecolor','#696DFF') #alias name for parameter
        

    axs.plot(x,y,fmt,alpha=alpha,lw=lw,ms=ms,mfc='none',mec=mec,**kwargs)
    axs.set_xlabel(r'$x \, (\mathrm{kpc})$') 
    axs.set_ylabel(r'$y \, (\mathrm{kpc})$') 

    if(axsProvided is False):
        axs.set_aspect('equal')

    if(save):
        if(filename == None):
            filename = './ClusterPostions.pdf'
        plt.savefig(filename,bbox_inches='tight')
        plt.close(fig)
    else :
        #Check if axs provided, which could mean plotting part of bigger plot, so don't show just return
        if(axsProvided is False):
            plt.show(block=True)
        return

def plot_tpcf(bins,corr,dcorr,axs=None,save=False,filename=None,**kwargs):
    """
    Plot computed Two-Point Correlation Function (TPCF) as a function of separation
   Parameters
    ----------
    bins : ndarray
        bins where TPCF has been computed
    corr : ndarray
        Correlation value
    dcorr : ndarray
        Error in Correlation value
    axs : Matplotlib axes instance
        Plot in this axis. Creates a new axis if not provided. 
    save : boolean
        Flag to save the plot, else just show.
    filename : string
        File to save to, else default filename 
    **kwargs: 
        Other optional parameters for plotting that go directly into the call for plt.plot.
    
    Returns
    -------
    None
    
    """ 
    #Create fig and axs if not provided
    if(axs is None):
        axsProvided = False
        fig,axs = plt.subplots(ncols=1)
    else:
        axsProvided = True

    #Filter nans in TPCF
    separation_bins,corr,dcorr = filter_stuff(bins,corr,dcorr)
    
    #Use some plot defaults if not explicitly provided by user
    lw = kwargs.pop('lw',2.0)
    fmt = kwargs.pop('fmt','o-')
    color=kwargs.pop('color','#4591F5')
    capsize=kwargs.pop('capsize',5)
    ms = kwargs.pop('ms',4)
    ms = kwargs.pop('markersize',4) #alias name for parameter


    axs.errorbar(separation_bins,1+corr,yerr=dcorr,
        fmt=fmt,lw=lw,ms=ms,color=color,capsize=capsize,**kwargs)
    axs.set_xlabel(r"$\Delta x \, (\mathrm{kpc})$")
    axs.set_ylabel(r"$1+ \omega_{\mathrm{LS}}\left(\theta \right)$")
    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.legend()

    if(save):
        if(filename == None):
            filename = './ClusterTPCF.pdf'
        plt.savefig(filename,bbox_inches='tight')
        plt.close(fig)
    else :
        #Check if axs provided, which could mean plotting part of bigger plot, so don't show just return
        if(axsProvided is False):
            plt.show(block=True)
        return

def filter_stuff(bins,corr,dcorr):
    """
    Filters bins where there are nan values in the TPCF

    Parameters
    ----------
    bins : ndarray
        Bins where TPCF computed
    corr : ndarray
        Values of the TPCF
    dcorr : Matplotlib axes instance
        Errors in calculated TPCF
    
    Returns
    -------
    separation_bins: ndarray
        Filtered bins
    corr_lz: ndarray
        Filtered corr
    dcorr_lz: ndarray
        Filtered dcorr

    """
    separation_bins = (bins[1:]+bins[:-1])/2.
    indices = np.where(np.logical_and(1+corr>0.0,np.abs(corr)>np.abs(dcorr)))
    dcorr_lz = dcorr[indices]
    separation_bins = separation_bins[indices]
    corr_lz = corr[indices]
    return separation_bins,corr_lz,dcorr_lz
