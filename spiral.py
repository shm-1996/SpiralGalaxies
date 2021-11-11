"""
Routines to distribute star clusters in a spiral pattern to emulate a spiral galaxy.

AUTHOR
Shyam Harimohan Menon (2020)
"""

import numpy as np

def SpiralNoise(nsamples=10000,noiseConst=10):
    """
    Introduce some noise in the azimuthal angle for smoother spiral arms
    Parameters
    ----------
    nsamples : integer
        Number of samples to produce
    noiseConst: float
        Constant denoting the degree of smoothness in the spiral. Higher is smoother.

    Returns
    ----------
    noise : ndarray
        Azimuthal angular noise for smoother spiral arms
    """
    import scipy
    angle = 2*np.pi * np.random.uniform(0, 1,size=nsamples)
    angle = np.sort(angle)
    Ptheta = 0.5*noiseConst**(1.0-np.cos(2.0*angle)) #PDF to emulate
    cumsum = np.trapz(Ptheta,angle)
    Ptheta = Ptheta/cumsum # this is the normalised PDF
    Ctheta = np.cumsum(Ptheta[1:]*np.diff(angle)) #This is the CDF
    interp = scipy.interpolate.interp1d(Ctheta,angle[1:],fill_value='extrapolate')
    x = np.random.uniform(0,1,size=nsamples)
    noise = interp(x)
    return noise



def Create_Spiral(pitchAngle=15.0,m=1,r0=2.0,rmax=20.0,nsamples=1000,noiseConst=10):
    """
    Create a galaxy with star clusters distributed in a spiral pattern.
    Parameters
    ----------
    pitchAngle  : float
        Pitch angle of the spiral arm in degrees.
    m  : integer
        Number of spiral arms
    ro  : float
        Reference radius for setting the phase
    rmax : float
        Maximum radius of the galaxy
    nsamples : integer
        Number of samples to produce
    noiseConst  : float
        This value sets how smooth the spirals are, higher is smoother

    Returns
    ----------
    x : ndarray
        x positions of star clusters
    y : ndarray
        y positions of star clusters
    """

    #Parameters
    PA = np.deg2rad(pitchAngle)  # Pitch angle of spiral arms

    #Generate sinusoidal noise
    if(noiseConst>0):
        dtheta = SpiralNoise(nsamples=nsamples,noiseConst=noiseConst)
    else:
        dtheta = 0
    
    #r = np.sqrt(np.random.uniform(0,rmax,size=nsamples))
    r = np.random.uniform(0,rmax,size=nsamples)
    theta = m/(np.tan(PA)) * np.log((r+1.e-4)/r0) #1e-4 added here to prevent nans near log(0)
    xl, yl = r*np.cos(theta+dtheta), r*np.sin(theta+dtheta)
    xr, yr = r*np.cos(theta+dtheta+np.pi), r*np.sin(theta+dtheta+np.pi)
    x,y = np.append(xl,xr),np.append(yl,yr)

    return x,y
