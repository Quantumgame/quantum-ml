# double_dot.py
# This script is used to perform a two dimensional sweep

# Last Updated: 12th October 2017
# Sandesh Kalantre
import numpy as np
import potential_profile
import thomas_fermi 
import time
import datetime

def calc_2D_map():
    N_v = 100
    V_d1_vec = np.linspace(-140e-3,-180e-3,N_v)
    V_d2_vec = np.linspace(-140e-3,-180e-3,N_v)

    x = np.linspace(-200,200,100)

    # trial potential profile
    param1 = [200e-3,-150,50,5,50]
    param3 = [250e-3,0,50,5,50]
    param5 = [200e-3,150,50,5,50]

    physics = {'x' : x,
               'K_0' : 5e-3, 
               'sigma' : 10.0,
               'mu' : 0.1,
               'D' : 2,
               'g_0' : 1e-1,
               'c_k' : 0e-4,
               'beta' : 100,
               'kT' : 1e-5,
               'WKB_coeff' : 1,
               'barrier_tunnel_rate' : 1.0,
               'bias' : 1e-5,
               'ShortCircuitCurrent' : 1.0,
               'attempt_rate_coef' : 1
               }

    # list_list_b is a list of the possible set of gate voltages which is later converted into a potential profile
    list_list_b = [[param1,param3,param5] + [[a,-50,50,5,50],[b,50,50,5,50]] for a in V_d1_vec for b in V_d2_vec]

    st = time.time()
    tf = thomas_fermi.ThomasFermi(physics) 

    def wrapper(V):
        tf.physics['V'] = V 
        n = tf.calc_n()
        islands = tf.calc_islands()
        barriers = tf.calc_barriers()
        p_WKB = tf.calc_WKB_prob()
        charges = tf.calc_charges()
        cap_model = tf.calc_cap_model()
        stable_config = tf.calc_stable_config()
        current = tf.calc_current()
        charge = tf.calc_graph_charge()
        
        return current,charge 

    res = [wrapper(potential_profile.calc_V(x,y)) for y in list_list_b]
    np.save("~/data/double_dot" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),res)
    print("Time",time.time()-st)
