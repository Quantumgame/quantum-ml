# potential_profile.py
# This script contains the definition of the gate model used to create the
# potential profile. The script should not be changed when being used for 
# development since the data generation scripts might be dependent on this.
# Use the "Test Notebooks" folder instead.

# Last Updated : 14th November 2017
# Sandesh Kalantre

import numpy as np

def calc_V_gate(x,gate_param):
    '''
    Calculates the gate potential profile for each gate

    Input: x,gate_param
    x : linear array of x values, units are predefined 
    gate_param : dictionary of parameters which define each gate
        ‘peak’ : potential at the mean at the location of the electrons
        ‘mean’ : position of the gate along x
        ‘rho’ : radius of the cylindrical gate
        ‘h’ : distance of the gate from the electron density 
    ‘alpha’ : lever arm 
    ‘screen’ : screening length 

    Output: V
    The potential profile calculated as
    V(x-mean) = peak/log(h/rho) * log(sqrt(h^2 + x^2)/rho) * exp(-|x-mean|/screen)
    This model is assumes the gate behaves like a cylindrical conductor of
    radius rho placed at a height h from the 2DEG.
    '''
    peak = gate_param['peak']
    mean = gate_param['mean']
    rho = gate_param['rho']
    h = gate_param['h']
    screen = gate_param['screen']
    
    dx = np.abs(x-mean)
    return (peak/np.log(h/rho))*np.log(np.sqrt(dx**2 + h**2)/rho)*np.exp(-dx/screen)

def calc_V(x,gate_param_list):
    '''
    Calculates the gate potential profile for a list of gates. Takes in a list of gate parameters where each element 
    is a dictionary as required by calc_V_gate

    Input: x,gate_param_list
    x : linear array of x values, units are predefined
    V : list of gate parameters as required by calc_V_gate

    Output: V
    linear sum of potentials calculated for each gate
    '''
    return np.sum([calc_V_gate(x,y) for key,y in gate_param_list.items()],axis=0)
