# random_single_dot.py
# This script is used to generate I vs V_gate characteristics for a single dot. The physical paramters across different simulations are randomized.
# Data for machine learning is generated using this script.

# Last Modified: 5th November
# Sandesh Kalantre

import numpy as np
import time
import imp
import datetime

import potential_profile
import thomas_fermi
imp.reload(thomas_fermi)

def random_sample(mean,sigma_mu = 0.1):
    return np.random.normal(mean,sigma_mu*np.abs(mean))

def calc_plunger_trace(N_v = 1000):
    st = time.time()
    N_grid = 100
    system_size = 60
    x = np.linspace(-system_size/2,system_size/2,N_grid,endpoint=True)


    physics = {'x' : x,
               'K_0' : random_sample(1.0e-2), 
               'sigma' : 1,
               'mu' : 0.10,
               'D' : 2,
               'g_0' : 5e-2,
               'c_k' : 1e-3,
               'beta' : 50,
               'kT' : 1e-5,
               'WKB_coeff' : 1,
               'barrier_tunnel_rate' : 10.0,
               'bias' : 1e-4,
               'ShortCircuitCurrent' : 0.0,
               'attempt_rate_coef' : 1
                }

    V1_val = random_sample(200e-3)
    V3_val = random_sample(200e-3)

    gate1_pos = random_sample(-15)
    gate2_pos = random_sample(0)
    gate3_pos = random_sample(15)
    param1 = [V1_val,gate1_pos,25,50,50]
    param2 = [-200e-3,gate2_pos,25,50,50]
    param3 = [V3_val,gate3_pos,25,50,50]
    gates = [param1,param2,param3] 
    
    V_gate = np.linspace(-100e-3,-220e-3,N_v)

    def wrapper(V_gate):
        # potential profile
        gates[1][0] = V_gate
        V = potential_profile.calc_V(x,gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        n = tf.calc_n()
        islands = tf.calc_islands()
        barriers = tf.calc_barriers()
        p_WKB = tf.calc_WKB_prob()
        charges = tf.calc_charges()
        cap_model = tf.calc_cap_model()
        stable_config = tf.calc_stable_config()
        current = tf.calc_current()
        charge = tf.calc_graph_charge()
        
        return current,charge,cap_model

    res = [wrapper(y) for y in V_gate]
    I = np.array([y[0] for y in res])
    charge = np.array([np.sum(y[1]) for y in res])
    res = {'physics' : physics, 'gates' : gates, 'V_gate' : V_gate,'I' : I,'charge' : charge}

    print("Physics\n",physics)
    print("Gates\n",gates)
    print("Completed calculation in",time.time()-st,"seconds.")
    np.save("/Users/sandesh/data/single_dot/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),res)
    return V_gate,I,charge

    
def calc_barrier_map(N_v = 100):
    st = time.time()
    N_grid = 100
    system_size = 60
    x = np.linspace(-system_size/2,system_size/2,N_grid,endpoint=True)


    physics = {'x' : x,
               'K_0' : random_sample(1.0e-2), 
               'sigma' : 1,
               'mu' : 0.10,
               'D' : 2,
               'g_0' : 5e-2,
               'c_k' : 1e-3,
               'beta' : 50,
               'kT' : 1e-5,
               'WKB_coeff' : 1,
               'barrier_tunnel_rate' : 10.0,
               'bias' : 1e-4,
               'ShortCircuitCurrent' : 0.0,
               'attempt_rate_coef' : 1,
               'sensors' : [(0,50)] 
                }

    V1_val = random_sample(200e-3)
    V3_val = random_sample(200e-3)

    gate1_pos = random_sample(-15)
    gate2_pos = random_sample(0)
    gate3_pos = random_sample(15)
    param1 = [V1_val,gate1_pos,25,50,50]
    param2 = [-150e-3,gate2_pos,25,50,50]
    param3 = [V3_val,gate3_pos,25,50,50]
    gates = [param1,param2,param3] 
    
    import itertools
    V_gate = list(itertools.product(np.linspace(150e-3,300e-3,N_v),np.linspace(150e-3,300e-3,N_v)))

    def wrapper(V_gate):
        # potential profile
        gates[0][0] = V_gate[0]
        gates[2][0] = V_gate[1]
        
        V = potential_profile.calc_V(x,gates) 
        physics['V'] = V
        
        tf = thomas_fermi.ThomasFermi(physics)
        n = tf.calc_n()
        islands = tf.calc_islands()
        barriers = tf.calc_barriers()
        p_WKB = tf.calc_WKB_prob()
        charges = tf.calc_charges()
        charge_centres = tf.calc_charge_centers()
        cap_model = tf.calc_cap_model()
        stable_config = tf.calc_stable_config()
        current = tf.calc_current()
        charge = tf.calc_graph_charge()
        sensor = tf.calc_sensor()
        state = tf.calc_state()
        
        return current,charge,sensor,state,cap_model,p_WKB

    res = [wrapper(y) for y in V_gate]
    I = np.array([y[0] for y in res])
    charge = np.array([y[1] for y in res])
    sensor = np.array([y[2] for y in res])
    state = [y[3] for y in res]
    cap_model = [y[4] for y in res]
    p_WKB = [y[5] for y in res]
    res = {'physics' : physics, 'gates' : gates, 'V_gate' : V_gate,'current' : I,'charge' : charge, 'sensor' : sensor,'state' : state, 'cap_model' : cap_model,'tunnel_vec' : p_WKB}

    print("Physics\n",physics)
    print("Gates\n",gates)
    print("Completed calculation in",time.time()-st,"seconds.")
    np.save("/Users/sandesh/data/single_dot/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),res)
    return res
