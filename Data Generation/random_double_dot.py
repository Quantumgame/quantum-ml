# random_double_dot.py
# This script is used to generate I vs V_gate characteristics for a double dot. The physical paramters across different simulations are randomized.
# Data for machine learning is generated using this script.

# Last Modified: 15th November
# Sandesh Kalantre

import numpy as np
import time
import datetime
import itertools

import potential_profile
import thomas_fermi

data_path = "/Users/sandesh/data/quantum-ml/double_dot_mac/"

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
    system_size = 120
    x = np.linspace(-system_size/2,system_size/2,N_grid,endpoint=True)

    physics = {'x' : x,
               'K_0' : random_sample(1e-2), 
               'sigma' : 3.0,
               'mu' : 0.1,
               'D' : 2,
               'g_0' : np.random.uniform(1.0),
               'c_k' : random_sample(1e-3),
               'beta' : 1000,
               'kT' : 5e-5,
               'WKB_coeff' : 0.5,
               'barrier_tunnel_rate' : 10.0,
               'V_L' : 5e-5,
               'V_R' : -5e-5,
               'short_circuit_current' : 1e-4,
               'attempt_rate_coef' : 1,
               'sensors' : [(-20,50),(20,50)],
               'barrier_current' : 1e-3,
               'sensor_gate_coeff' : 1e-1,
               }
    
    K_mat = thomas_fermi.calc_K_mat(x,physics['K_0'],physics['sigma'])
    physics['K_mat'] = K_mat
    physics['bias'] = physics['V_L'] - physics['V_R']
    return physics

def calc_gates():
    gate1 = {'peak' : 200e-3,'mean' : -40,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gate2 = {'peak' : -100e-3,'mean' : -20,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gate3 = {'peak' : 200e-3,'mean' : 0,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gate4 = {'peak' : -100e-3,'mean' : 20,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gate5 = {'peak' : 200e-3,'mean' : 40,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gates = {'gate1' : gate1,'gate2' : gate2, 'gate3' : gate3, 'gate4' : gate4, 'gate5' : gate5}
    return gates


def calc_plunger_map(N_v = 100,data_path=data_path):
    physics = calc_input_physics()
    gates = calc_gates() 
    # randomize over the elements of the gates
    for key,item in gates.items():
        gates[key] = randomize_dict(gates[key])
        
    V_P1_vec = np.linspace(-80e-3,-210e-3,N_v)
    V_P2_vec = np.linspace(-80e-3,-210e-3,N_v)
    V_P_map = list(itertools.product(V_P1_vec,V_P2_vec))

    def wrapper(V_gate):
        # potential profile
        gates['gate2']['peak'] = V_gate[0]
        gates['gate4']['peak'] = V_gate[1]
        
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        physics['gates'] = gates
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output
    
    output_vec = [wrapper(y) for y in V_P_map]

    result = {'physics' : physics, 'type' : 'V_P_map', 'V_P1_vec' : V_P1_vec,'V_P2_vec' : V_P2_vec, 'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f"),result)
    return result 


