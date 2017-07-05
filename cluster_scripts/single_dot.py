# This script is used to generate data for a physical setup with 3 gates, implying a QPC, single dot and a SC region.

import numpy as np
import sys
import time
import os
import imp

sys.path.append('/Users/ssk4/quantum-ml/nanowire_model')
import physics
import potential_profile
import markov
import exceptions
imp.reload(physics)
imp.reload(potential_profile)
imp.reload(markov)
imp.reload(exceptions)

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
    
def calculate_1d_map(ind=0):
    #ind is just a param passed when multicore processing is used.
    st = time.time()
    
    physics_model = {}
    # multiple of eV
    physics_model['E_scale'] = 1
    # multiple of nm
    physics_model['dx_scale'] = 1
    physics_model['kT'] = 1000e-6

    # just initial param to generate the graph object
    b1 = [get_random(-200e-3,sigma_mean=0.02),get_random(-0.5),get_random(0.05),1]
    d = [200e-3,get_random(0.0),get_random(0.05),1]
    b2 = [get_random(-200e-3,sigma_mean=0.02),get_random(0.5),get_random(0.05),1]

    x = np.linspace(-1,1,100)
    physics_model['x'] = x
    physics_model['list_b'] = [b1,d,b2]
    physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])

        
    # K_onsite decides the charging energy
    physics_model['K_onsite'] = np.random.uniform(10e-3,50e-3)
    physics_model['sigma'] = x[1] - x[0]
    physics_model['x_0'] = 0.1*(x[1] - x[0])
    physics_model['mu_l'] = (100.0e-3,100.1e-3)
    physics_model['battery_weight'] = 10
    physics_model['short_circuit_current'] = 1
    physics_model['QPC_current_scale'] = 1e-4

    graph_model = (2,1)
    tf_strategy = 'simple'

    graph = markov.Markov(graph_model,physics_model,tf_strategy)
    graph.find_n_dot_estimate()

    N_v = 1024
    V_d_vec = np.linspace(0e-3,200e-3,N_v)
    
    output_vec = []
    for i in range(N_v):
        d[0] = V_d_vec[i]
        physics_model['list_b'] = [b1,d,b2]
        V = potential_profile.V_x_wire(x,physics_model['list_b'])
        physics_model['V'] = potential_profile.V_x_wire(x,[b1,d,b2])
        output_vec += [calculate_current((graph,physics_model))]

    # data is a dictionary with two keys, 'input' and 'output'
    # data['input'] = {physics_model, graph_model, tf_strategy}
    # data['output'] : list with output from calculate current
    data = {}
    data['input'] = {'physics_model' : physics_model,
                     'graph_model' : graph_model,
                     'tf_strategy' : tf_strategy,
                     'V_d_vec' : V_d_vec}
    
    data['output'] = output_vec
    
    import datetime
    dt = str(datetime.datetime.now()) 
    #np.save('/Users/ssk4/data/single_dot_' + str(N_v) + '_grid_' + dt + '.npy',data)
    # during testing
    np.save('/Users/ssk4/data/single_dot_test.npy',data)
    
    return (time.time()-st)
