# random_single_dot.py
# This script is used to generate I vs V_gate characteristics for a single dot. The physical paramters across different simulations are randomized.
# Data for machine learning is generated using this script.

# Last Modified: 5th November
# Sandesh Kalantre

import numpy as np
import time
import datetime
import itertools

import potential_profile
import thomas_fermi

data_path = "/Users/sandesh/data/quantum-ml/single_dot/"

def random_sample(mean,sigma_mu = 0.0001):
    return np.random.normal(mean,sigma_mu*np.abs(mean))

def randomize_dict(dictionary,keys = []):
    '''
    Randomizes the elements in the dictionary for the given list of keys
    If no keys are specified, then randomizes over all keys
    '''
    if keys == []:
        for key,item in dictionary.items():
            dictionary[key] = random_sample(item)
    else:
        for key in keys:
            dictionary[key] = random_sample(dictionary[key])
    return dictionary

def calc_input_physics():
    N_grid = 100
    system_size = 60
    x = np.linspace(-system_size/2,system_size/2,N_grid,endpoint=True)

    physics = {'x' : x,
               'K_0' : random_sample(1e-2), 
               'sigma' : 1.0,
               'mu' : 0.1,
               'D' : 2,
               'g_0' : 5e-2,
               'c_k' : random_sample(1e-3),
               'beta' : 50,
               'kT' : 1e-5,
               'WKB_coeff' : 1,
               'barrier_tunnel_rate' : 10.0,
               'V_L' : 5e-5,
               'V_R' : -5e-5,
               'short_circuit_current' : 1.0,
               'attempt_rate_coef' : 1,
               'sensors' : [(0,50)],
               'barrier_current' : 1.0,
               }
    K_mat = thomas_fermi.calc_K_mat(x,physics['K_0'],physics['sigma'])
    physics['K_mat'] = K_mat
    return physics


def calc_plunger_trace(N_v = 100):
    physics = calc_input_physics()
    
    gate1 = {'peak' : 200e-3,'mean' : -15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate2 = {'peak' : -150e-3,'mean' : 0,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate3 = {'peak' : 200e-3,'mean' : 15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gates = [gate1,gate2,gate3]
    # randomize over the elements of the gates
    for i in range(len(gates)):
        gates[i] = randomize_dict(gates[i])
    
    V_P_vec = np.linspace(-100e-3,-200e-3,N_v)

    def wrapper(V_p):
        '''
        Set the voltage for the plunger
        '''
        gate2['peak'] = V_p
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output

    output_vec = [wrapper(y) for y in V_P_vec]

    result = {'physics' : physics, 'type' : 'V_P_trace', 'V_P_vec' : V_P_vec, 'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

def calc_barrier_map(N_v = 100,V_p = -150e-3):
    physics = calc_input_physics()
    
    gate1 = {'peak' : 200e-3,'mean' : -15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate2 = {'peak' : -150e-3,'mean' : 0,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate3 = {'peak' : 200e-3,'mean' : 15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    
    gates = [gate1,gate2,gate3]
    # randomize over the elements of the gates
    for i in range(len(gates)):
        gates[i] = randomize_dict(gates[i])
        
    V_B1_vec = np.linspace(150e-3,300e-3,N_v)
    V_B2_vec = np.linspace(150e-3,300e-3,N_v)
    V_B_map = list(itertools.product(V_B1_vec,V_B2_vec))

    def wrapper(V_gate):
        # potential profile
        gate1['peak'] = V_gate[0]
        gate3['peak'] = V_gate[1]
        
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output
    
    output_vec = [wrapper(y) for y in V_P_vec]

    result = {'physics' : physics, 'type' : 'V_B_map', 'V_B1_vec' : V_B1_vec,'V_B2_vec' : V_B2_vec, 'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

def calc_full_map(N_v = 100):
    physics = calc_input_physics()
    
    gate1 = {'peak' : 150e-3,'mean' : -15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate2 = {'peak' : -150e-3,'mean' : 0,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    gate3 = {'peak' : 150e-3,'mean' : 15,'rho' : 50, 'h' : 25,'screen' : 50,'alpha' : 1.0}
    
    gates = [gate1,gate2,gate3]
    # randomize over the elements of the gates
    for i in range(len(gates)):
        gates[i] = randomize_dict(gates[i])
        
    V_B1_vec = np.linspace(150e-3,300e-3,N_v) 
    V_P_vec = np.linspace(-100e-3,-200e-3,N_v) 
    V_B2_vec = np.linspace(150e-3,300e-3,N_v) 
    V_map = list(itertools.product(V_B1_vec,V_P_vec,V_B2_vec))

    def wrapper(V_gate):
        # potential profile
        gate1['peak'] = V_gate[0]
        gate2['peak'] = V_gate[1]
        gate3['peak'] = V_gate[2]
        
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output
    
    output_vec = [wrapper(y) for y in V_P_vec]

    result = {'physics' : physics, 'type' : 'V_full_map', 'V_B1_vec' : V_B1_vec,'V_P_vec' : V_P_vec,'V_B2_vec' : V_B2_vec,\
              'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

