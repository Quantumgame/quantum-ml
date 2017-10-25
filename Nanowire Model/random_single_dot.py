# random_single_dot.py
# This script is used to generate I vs V_gate characteristics for a single dot. The physical paramters across different simulations are randomized.
# Data for machine learning is generated using this script.

# Last Modified: 14th October
# Sandesh Kalantre

import numpy as np
import time
import imp
import datetime

import potential_profile
import thomas_fermi

def random_sample(mean,sigma_mu = 0.1):
    return np.random.normal(mean,sigma_mu*np.abs(mean))

def calc_1D_trace(N_v = 1000):
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

    
