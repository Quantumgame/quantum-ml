# Module to build a potential landscape
import numpy as np

def gauss(x,mean=0.0,stddev=0.02,peak=1.0):
    '''
    Input:
    x : x-coordintes
    Output:
    f(x) where f is a Gaussian with the given mean, stddev and peak value
    '''
    stddev = 5*(x[1] - x[0])
    return peak*np.exp(-(x-mean)**2/(2*stddev**2))

def init_ndot(x,n_dot):
    '''
    Input:
    x : 1d grid for the dots
    ndot :  number of dots
    Output:
    y : cordinates of the potential grid with ndots

    The potential barriers are modelled as gaussians
    '''
    # n dots imply n+1 barriers
    bar_centers = x[0] + (x[-1] - x[0])*np.random.rand(n_dot+1) 
    bar_heights = np.random.rand(n_dot+1)
    #bar_heights = 0.5*np.ones(n_dot+1)

    N = len(x)
    y = np.zeros(N)

    # no need to optimize here really since the dot number is generally small, the calculation of the gauss function is already done in a vectorised manner
    for j in range(n_dot+1):
        y += gauss(x-bar_centers[j],peak=bar_heights[j])
    return y

