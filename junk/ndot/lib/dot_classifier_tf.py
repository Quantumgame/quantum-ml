# Module to classify a given potential profile using Thomas-Fermi

import numpy as np

def create_K_matrix(x, E_scale=1.0, sigma=1.0,x_0 = 1.0 ):
    '''
    Input: 
        x : discrete 1D grid
        E_scale : energy scale for the K matrix, default units is eV
        sigma : impact paramter to prevent blow up at the same point
    Output:
        K : matrix of size x.size times x.size

    K(x1,x2) = E_scale / sqrt((x1 - x2)^2 + sigma^2)
    '''
    r = np.abs(x[:,np.newaxis] - x)
    K = E_scale/np.sqrt(r**2 + sigma**2)*np.exp(-r/x_0)
    return K

def solve_thomas_fermi_fixed_mu(x,V,K,mu):
    '''
    Input:
        x    : discrete 1D grid
        V    : potential
        K    : Coulomb interaction matrix between two points
        mu   : fixed chemical potential equated to the lead potential
    Output:
        n : electron density as a function of x

    Solves the equation (V - mu)+ K * n = 0
    '''
    
    b = mu - V
    A = K
    n = np.linalg.solve(A,b)
    return n

def classify_n(n,tol=1e-1):
    '''
    Input:
        n : value of charge density
    Output:
        'b','d' : barrier or dot, very crude classification scheme
    '''
    if n < 0:
        return 'b'
    else:
        return 'd'
def add_leads(mask):
    '''
    Input:
    mask :  prelim mask with only 'b' or 'd'
    Output:
    mask : new mask in which the first and last islands are labelled as 'l1' and 'l2'
    '''
    # lead1
    i = 0
    while(i < len(mask) and mask[i] != 'b'):
        mask[i] = 'l1' 
        i += 1
    # lead2
    i = len(mask) - 1
    while(i > 0 and mask[i] != 'b'):
        mask[i] = 'l2' 
        i -= 1

    return mask
def get_mask(x,V,K,mu):
    '''
    Input:
    x    : discrete 1D grid
    V    : potential
    K    : Coulomb interaction matrix between two points
    mu   : fixed chemical potential equated to the lead potential
    Output:
    mask : array size of len(x), where each point is either 'l1','d','b','l2' standing for lead1, dot, barrier and lead2 respectively. 
    Based on a crude classification scheme, where a region is classified as an island if the electron density calculated for a fixed mu(= lead potential) under Thomas-Fermi is greater than 0 or a barrier otherwise.
    '''
    n = solve_thomas_fermi_fixed_mu(x,V,K,mu)
    mask = map(classify_n,n)

    # the mask needs to be brought into a standard form, where the points to left of the first barrier and to the right barrier aree the leads
    mask = add_leads(mask)

    return mask






