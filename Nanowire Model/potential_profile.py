# potential_profile.py
# This script contains the definition of the gate model used to create the
# potential profile. The script should not be changed when being used for 
# development since the data generation scripts might be dependent on this.
# Use the "Test Notebooks" folder instead.

# Last Updated : 12th October 2017
# Sandesh Kalantre

import numpy as np

def calc_V_gate(x,param):
    '''
    x : linspace of x values on which the potential is to be calculated

    param = [peak,mean,h,rho,screen]
        peak : value of V at x = mean
        mean : central position of the gate
        h : height of the gate above the 2DEG
        rho : radius of the cylindrical gate
        screen : screening length in the 2DEG
        
    V(x-mean) = peak/log(h/rho) * log(sqrt(h^2 + x^2)/rho) * exp(-|x-mean|/screen)

    This model is assuming the gate behaves like a a cylindrical conductor of
    radius r placed at a height h from the 2DEG.
    '''
    (peak,mean,h,rho,screen) = param
    dx = np.abs(x-mean)
    return (peak/np.log(h/rho))*np.log(np.sqrt(dx**2 + h**2)/rho)*np.exp(-dx/screen)

def calc_V(x,param_list):
    '''
    This function is used to add potential profiles of different gates.
    '''
    return np.sum([calc_V_gate(x,y) for y in param_list],axis=0)
