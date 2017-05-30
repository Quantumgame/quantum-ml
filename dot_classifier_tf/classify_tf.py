# Module to classify a given potential profile using Thomas-Fermi

import numpy as np

def create_K_matrix(x, E_scale=1.0, sigma=1.0):
    '''
    Input: 
        x : discrete 1D grid
        E_scale : energy scale for the K matrix, default units is eV
        sigma : impact paramter to prevent blow up at the same point
    Output:
        K : matrix of size x.size times x.size

    K(x1,x2) = E_scale / sqrt((x1 - x2)^2 + sigma^2)
    '''
    K = E_scale/np.sqrt((x[:, np.newaxis] - x)**2 + sigma**2)
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


