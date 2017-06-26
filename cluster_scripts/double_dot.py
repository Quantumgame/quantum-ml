# This is a script to generate a single 2D map of double dot system 

import numpy as np
import imp
import sys
import os
import time
import copy
import multiprocessing as mp

sys.path.append(os.path.expanduser('~/quantum-ml/nanowire_model'))
import potential_profile
import markov
import exceptions

# wrapper around the base markov class functions
# this function also handles the NoBarrier and InvalidChargeState exceptions
def calculate_current(param):
    graph = param[0]
    physics_model = param[1]
    try:
        graph.physics = physics_model
        graph.tf.__init__(physics_model)
        graph.find_n_dot_estimate()
        graph.find_start_node()
        graph.generate_graph()
        return graph.get_output()
    except exceptions.NoBarrierState:
        output = {}
        output['current'] = graph.tf.short_circuit_current
        output['charge_state'] = (0,)
        output['prob_dist'] = (0,)
        output['num_dot'] = 0
        output['state'] = 'ShortCircuit'
        return output
    except exceptions.InvalidChargeState:
        output = {}
        output['current'] = 0
        output['charge_state'] = (0,)
        output['prob_dist'] = (0,)
        output['num_dot'] = 0
        output['state'] = 'NoDot'
        return output

def get_random(mean,sigma_mean = 0.05):
        return np.random.normal(mean,sigma_mean*np.abs(mean))
    
def calculate_2d_map(ind):
    physics_model = {}
    # multiple of eV
    physics_model['E_scale'] = 1
    # multiple of nm
    physics_model['dx_scale'] = 1
    physics_model['kT'] = 1000e-6

    # just initial param to generate the graph object
    b1 = [get_random(-200e-3,sigma_mean=0.02),get_random(-0.6),get_random(0.05),1]
    d1 = [200e-3,get_random(-0.4),get_random(0.05),1]
    b2 = [get_random(-250e-3,sigma_mean=0.02),get_random(0.0),get_random(0.05),1]
    d2 = [200e-3,get_random(0.4),get_random(0.05),1]
    b3 = [get_random(-200e-3,sigma_mean=0.02),get_random(0.6),get_random(0.05),1]

    x = np.linspace(-1,1,100)
    physics_model['x'] = x
    physics_model['list_b'] = [b1,d1,b2,d2,b3]
    physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])


    physics_model['K_onsite'] = get_random(45e-3)
    physics_model['sigma'] = x[1] - x[0]
    physics_model['x_0'] = 0.1*(x[1] - x[0])
    physics_model['mu_l'] = (300.0e-3,300.1e-3)
    physics_model['battery_weight'] = 10
    physics_model['short_circuit_current'] = 1

    graph_model = (2,1)
    tf_strategy = 'simple'

    graph = markov.Markov(graph_model,physics_model,tf_strategy)
    graph.find_n_dot_estimate()


    N_v = 100
    V_d_vec = np.linspace(50e-3,400e-3,N_v)
    output_vec = []
    input_vec = []
    st = time.time()
    for i in range(N_v):
        for j in range(N_v):
            d1[0] = V_d_vec[i]
            d2[0] = V_d_vec[j]
            physics_model['list_b'] = [b1,d1,b2,d2,b3]
            V = potential_profile.V_x_wire(x,physics_model['list_b'])
            physics_model['V'] = potential_profile.V_x_wire(x,[b1,b2,b3,d1,d2])
            input_vec += [physics_model.copy()]
            output_vec += [calculate_current((graph,physics_model))]
    print("time",time.time() - st)

    # store the data
    # data is a list of dictonaries, with two keys : 'input' , 'output'
    # input is a dict with three keys '
    data = []
    for i in range(len(output_vec)):
        inp = {}
        inp['graph_model'] = graph_model
        inp['tf_strategy'] = tf_strategy
        inp['physics_model'] = input_vec[i]
        data += [{'input' : inp,'output' : output_vec[i]}]

    import datetime
    dt = str(datetime.datetime.now()) 
    np.save(os.path.expanduser('~/datadump/double_dot_' + str(N_v) + '_grid_' + dt + '.npy'),data)