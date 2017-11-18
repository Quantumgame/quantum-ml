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

# either modify this path or change while calling the function
data_path = "/Users/sandesh/data/quantum-ml/single_dot/"

def random_sample(mean,sigma_mu = 0.05):
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

def calc_gates():
    gate1 = {'peak' : 200e-3,'mean' : -20,'rho' : 5, 'h' :50,'screen' : 20,'alpha' : 1.0}
    gate2 = {'peak' : 0e-3,'mean' : 0,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gate3 = {'peak' : 200e-3,'mean' : 20,'rho' : 5, 'h' : 50,'screen' : 20,'alpha' : 1.0}
    gates = {'gate1' : gate1,'gate2' : gate2, 'gate3' : gate3}
    return gates

def calc_input_physics():
    N_grid = 100
    system_size = 80
    x = np.linspace(-system_size/2,system_size/2,N_grid,endpoint=True)

    physics = {'x' : x,
               'K_0' : random_sample(1e-2), 
               'sigma' : 3.0,
               'mu' : 0.1,
               'D' : 2,
               'g_0' : np.random.uniform(0.1,1),
               'c_k' : random_sample(1e-3),
               'beta' : 1000,
               'kT' : 5e-5,
               'WKB_coeff' : 0.5,
               'barrier_tunnel_rate' : 10.0,
               'V_L' : 5e-5,
               'V_R' : -5e-5,
               'short_circuit_current' : 1e-4,
               'attempt_rate_coef' : 1,
               'sensors' : [(0,50)],
               'barrier_current' : 1e-3,
               'sensor_gate_coeff' : 1e-1,
               }
    K_mat = thomas_fermi.calc_K_mat(x,physics['K_0'],physics['sigma'])
    physics['K_mat'] = K_mat
    physics['bias'] = physics['V_L'] - physics['V_R']
    return physics

def calc_plunger_trace(N_v = 100,data_path=data_path):
    physics = calc_input_physics()
    gates = calc_gates()
    # randomize over the elements of the gates
    for key,item in gates.items():
        gates[key] = randomize_dict(gates[key])
    
    V_P_vec = np.linspace(-400e-3,0,N_v)

    def wrapper(V_p):
        '''
        Set the voltage for the plunger and calculate the outputs
        '''
        gates['gate2']['peak'] = V_p
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        physics['gates'] = gates
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output

    output_vec = [wrapper(y) for y in V_P_vec]

    result = {'physics' : physics, 'type' : 'V_P_trace', 'V_P_vec' : V_P_vec, 'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

def calc_barrier_map(N_v = 100,V_p = -100e-3,data_path=data_path):
    physics = calc_input_physics()
    gates = calc_gates() 
    # randomize over the elements of the gates
    for key,item in gates.items():
        gates[key] = randomize_dict(gates[key])
        
    V_B1_vec = np.linspace(100e-3,150e-3,N_v)
    V_B2_vec = np.linspace(100e-3,150e-3,N_v)
    V_B_map = list(itertools.product(V_B1_vec,V_B2_vec))
    gates['gate2']['peak'] = V_p

    def wrapper(V_gate):
        # potential profile
        gates['gate1']['peak'] = V_gate[0]
        gates['gate3']['peak'] = V_gate[1]
        
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output
    
    output_vec = [wrapper(y) for y in V_B_map]

    result = {'physics' : physics, 'type' : 'V_B_map', 'V_B1_vec' : V_B1_vec,'V_B2_vec' : V_B2_vec, 'output' : output_vec}
    np.save(data_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

def calc_full_map(N_v = 100,data_path = data_path):
    physics = calc_input_physics()
    gates = calc_gates() 
    
    # randomize over the elements of the gates
    for key,item in gates.items():
        gates[key] = randomize_dict(gates[key])
        
    V_B1_vec = np.linspace(100e-3,150e-3,N_v) 
    V_P_vec = np.linspace(50e-3,150e-3,N_v) 
    V_B2_vec = np.linspace(100e-3,150e-3,N_v) 
    V_map = list(itertools.product(V_B1_vec,V_P_vec,V_B2_vec))

    def wrapper(V_gate):
        # potential profile
        gates['gate1']['peak'] = V_gate[0]
        gates['gate2']['peak'] = V_gate[1]
        gates['gate3']['peak'] = V_gate[2]
        
        V = potential_profile.calc_V(physics['x'],gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        output = tf.output_wrapper()
        return output
    
    output_vec = [wrapper(y) for y in V_map]

    result = {'physics' : physics, 'type' : 'V_full_map', 'V_B1_vec' : V_B1_vec,'V_P_vec' : V_P_vec,'V_B2_vec' : V_B2_vec,\
              'output' : output_vec}
    np.save(data_path + "full_map" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),result)
    return result 

