# This is a script to generate a single 2D map of double dot system 

import numpy as np
import imp
import sys
import os
import time

sys.path.append('/Users/sandesh/repos/quantum-ml/nanowire_model')
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
    
def calculate_2d_map(ind=0):
    st = time.time()
    physics_model = {}
    # multiple of eV
    physics_model['E_scale'] = 1
    # multiple of nm
    physics_model['dx_scale'] = 1
    physics_model['kT'] = 350e-6

    # just initial param to generate the graph object
    #b1 = [get_random(-200e-3,sigma_mean=0.02),get_random(-0.6),get_random(0.05),1]
    #d1 = [200e-3,get_random(-0.1),get_random(0.05),1]
    #b2 = [get_random(-250e-3,sigma_mean=0.02),get_random(0.0),get_random(0.05),1]
    #d2 = [200e-3,get_random(0.1),get_random(0.05),1]
    #b3 = [get_random(-200e-3,sigma_mean=0.02),get_random(0.6),get_random(0.05),1]
    
    b1 = [-200e-3,-0.6,0.05,1]
    d1 = [200e-3,-0.2,0.05,1]
    b2 = [-350e-3,0.0,0.05,1]
    d2 = [200e-3,0.2,0.05,1]
    b3 = [-200e-3,0.6,0.05,1]

    x = np.linspace(-1,1,100)
    physics_model['x'] = x
    physics_model['list_b'] = [b1,d1,b2,d2,b3]
    physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])


    physics_model['K_onsite'] = np.random.uniform(5e-3,5e-3)
    physics_model['sigma'] = x[1] - x[0]
    physics_model['x_0'] = 50*(x[1] - x[0])
    physics_model['mu_l'] = (300.01e-3,300.0e-3)
    physics_model['battery_weight'] = 100
    physics_model['short_circuit_current'] = 1
    physics_model['QPC_current_scale'] = 1e-4
    physics_model['t'] = 10e-3
    
    graph_model = (1,1)
    tf_strategy = 'simple'

    graph = markov.Markov(graph_model,physics_model,tf_strategy)
    graph.find_n_dot_estimate()

    N_v = ind

    V_d1_vec = np.linspace(200e-3,300e-3,N_v)
    V_d2_vec = np.linspace(200e-3,300e-3,N_v)
    output_vec = []
    
    for i in range(N_v):
        print(i)
        for j in range(N_v):
            d1[0] = V_d1_vec[i]
            d2[0] = V_d2_vec[j]
            physics_model['list_b'] = [b1,d1,b2,d2,b3]
            physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])
            output_vec += [calculate_current((graph,physics_model))]

    # data is a dictionary with two keys, 'input' and 'output'
    # data['input'] = {physics_model, graph_model, tf_strategy}
    # data['output'] : list with output from calculate current
    data = {}
    data['input'] = {'physics_model' : physics_model,
                     'graph_model' : graph_model,
                     'tf_strategy' : tf_strategy,
                     'V_d1_vec' : V_d1_vec,
                     'V_d2_vec' : V_d2_vec}
    
    data['output'] = output_vec
    
    import datetime
    dt = str(datetime.datetime.now()) 
    #np.save('/Users/sandesh/data/double_dot_' + str(N_v) + '_grid_' + dt + '.npy',data)
    # during testing
    np.save('/Users/sandesh/data/double_dot_test1.npy',data)
    
    return (time.time()-st)

def calculate_3d_map(ind=0):
    st = time.time()
    physics_model = {}
    # multiple of eV
    physics_model['E_scale'] = 1
    # multiple of nm
    physics_model['dx_scale'] = 2
    physics_model['kT'] = 350e-6

    # just initial param to generate the graph object
    b1 = [get_random(-200e-3,sigma_mean=0.05),get_random(-0.5),get_random(0.05),1]
    d1 = [get_random(200e-3),get_random(-0.1),get_random(0.05),1]
    b2 = [get_random(-250e-3,sigma_mean=0.05),get_random(0.0),get_random(0.05),1]
    d2 = [get_random(200e-3),get_random(0.1),get_random(0.05),1]
    b3 = [get_random(-200e-3,sigma_mean=0.05),get_random(0.5),get_random(0.05),1]
    
    x = np.linspace(-1,1,100)
    physics_model['x'] = x
    physics_model['list_b'] = [b1,d1,b2,d2,b3]
    physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])


    physics_model['K_onsite'] = np.random.uniform(5e-3,25e-3)
    physics_model['sigma'] = x[1] - x[0]
    physics_model['x_0'] = 1*(x[1] - x[0])
    physics_model['mu_l'] = (300.1e-3,300.0e-3)
    physics_model['battery_weight'] = 10
    physics_model['short_circuit_current'] = 1
    physics_model['QPC_current_scale'] = 1e-4
    
    graph_model = (1,1)
    tf_strategy = 'simple'

    graph = markov.Markov(graph_model,physics_model,tf_strategy)
    graph.find_n_dot_estimate()

    N_v = 50
    V_b1_vec = np.linspace(-300e-3,-150e-3,N_v)
    V_b2_vec = np.linspace(-500e-3,-300e-3,N_v)
    V_b3_vec = np.linspace(-300e-3,-150e-3,N_v)
    output_vec = []
    
    for i in range(len(V_b2_vec)):
        print(i)
        for j in range(len(V_b1_vec)):
            for k in range(len(V_b3_vec)):
                b1[0] = V_b1_vec[j]
                b3[0] = V_b3_vec[k]
                b2[0] = V_b2_vec[i]
                physics_model['list_b'] = [b1,d1,b2,d2,b3]
                V = potential_profile.V_x_wire(x,physics_model['list_b'])
                physics_model['V'] = potential_profile.V_x_wire(x,physics_model['list_b'])
                output_vec += [calculate_current((graph,physics_model))]

    # data is a dictionary with two keys, 'input' and 'output'
    # data['input'] = {physics_model, graph_model, tf_strategy}
    # data['output'] : list with output from calculate current
    data = {}
    data['input'] = {'physics_model' : physics_model,
                     'graph_model' : graph_model,
                     'tf_strategy' : tf_strategy,
                     'V_b1_vec' : V_b1_vec,
                     'V_b2_vec' : V_b2_vec,
                     'V_b3_vec' : V_b3_vec}
    
    data['output'] = output_vec
    import datetime
    dt = str(datetime.datetime.now()) 
    np.save('/Users/ssk4/data/double_dot_3d' + str(N_v) + '_grid_' + dt + '.npy',data)
    # during testing
    #np.save('/Users/ssk4/data/double_dot_3d_test.npy',data)
    
    return (time.time()-st)