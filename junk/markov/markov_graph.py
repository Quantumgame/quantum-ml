from thomas_fermi import *

import numpy as np
import matplotlib.pyplot as plt
import Queue
import networkx as nx


def check_validity(v,model):
    '''
       Check if a vertex v=(N_L,N_D,N_R) under model=(p,q) 
        Constraints:
        1. 0 <= N_D <= p
        2. abs(N_L-N_R) <= q
        3. N_L + N_D + N_R = 0
    '''
    (N_L,N_D,N_R) = v
    (p,q) = model

    cond1 = (0 <= N_D <= p)
    cond2 = (np.abs(N_L-N_R) <= q)
    cond3 = (N_L + N_D + N_R == 0)
    cond = cond1 and cond2 and cond3
    return cond

def generate_neighbours(v,model):
    '''
        Takes in a 3-tuple (N_L,N_D,N_R) vertex and generates neighbours in the validity of model=(p,q)
        Constraints:
        1. 0 <= N_D <= p
        2. abs(N_L-N_R) <= q
        3. N_L + N_D + N_R = 0
        There are 4 possible neighbours for each vertex. Return the list of neighbours which are valid.
    '''

    (N_L,N_D,N_R) = v
    (p,q) = model

    L_dagger = (N_L+1,N_D-1,N_R)
    L = (N_L-1,N_D+1,N_R)
    R_dagger = (N_L,N_D-1,N_R+1)
    R = (N_L,N_D+1,N_R-1)

    valid = filter(lambda x : check_validity(x,model),[L_dagger,L,R_dagger,R])    
    return valid

def fermi(E,kT):
    return 1.0/(1 + np.exp(E/kT))

def find_weight(u,v,physics,method='energy'):
    '''
        Find the weight of edge from u to v
        Note that find_weight cannot find weight between any two vertices, it assumes that u,v are neighbours. DO NOT USE FOR FINDING WEIGHT BETWEEN ANY TWO NODES.

        Methods to assign the edge weights, 'mu' or 'energy' depending on which picture, chemical potential or Thomas-Fermi energy is used.
        Default is 'energy'

        'hybrid' uses both pictures.
    '''
    if method == 'mu':
        (mu_L1,mu_L2,V,K,kT) = physics
        # number of electons in u state on the dot 
        N = u[1]
        mu_D,n = solve_TF(mu_L1,mu_L2,N,V,K)
        
        u = np.array(u)
        v = np.array(v)
        diff = u - v
        # transport through right contact
        if diff[0] == 0:
            # contact to dot
            if v[1] > u[1]:
                weight = fermi(mu_D-mu_L2,kT)
            # dot to contact
            else:
                weight = 1 - fermi(mu_D-mu_L2,kT)
        # transport through left contact
        elif diff[2] == 0:
            # contact to dot
            if v[1] > u[1]:
                weight = fermi(mu_D-mu_L1,kT)
            # dot to contact
            else:
                weight = 1 - fermi(mu_D-mu_L1,kT)
    elif method == 'energy':
        (mu_L1,mu_L2,V,K,kT) = physics
        # number of electons in u state on the dot 
        N1 = u[1]
        mu_D1,n1 = solve_TF(mu_L1,mu_L2,N1,V,K)
        E_TF1 = calculate_E_TF(mu_L1,mu_L2,mu_D1,n1,V,K)

        N2 = v[1]
        mu_D2,n2 = solve_TF(mu_L1,mu_L2,N2,V,K)
        E_TF2 = calculate_E_TF(mu_L1,mu_L2,mu_D2,n2,V,K)

        weight = fermi(E_TF2 - E_TF1,kT)

    elif method == 'hybrid':
        (mu_L1,mu_L2,V,K,kT) = physics
        # number of electons in u state on the dot 
        N1 = u[1]
        mu_D1,n1 = solve_TF(mu_L1,mu_L2,N1,V,K)
        E_TF1 = calculate_E_TF(mu_L1,mu_L2,mu_D1,n1,V,K)

        N2 = v[1]
        mu_D2,n2 = solve_TF(mu_L1,mu_L2,N2,V,K)
        E_TF2 = calculate_E_TF(mu_L1,mu_L2,mu_D2,n2,V,K)

        energy_weight = fermi(E_TF2 - E_TF1,kT)
        
        u = np.array(u)
        v = np.array(v)
        diff = u - v
        # transport through right contact
        if diff[0] == 0:
            # contact to dot
            if v[1] > u[1]:
                weight = fermi(mu_D1-mu_L2,kT)
            # dot to contact
            else:
                weight = 1 - fermi(mu_D1-mu_L2,kT)
        # transport through left contact
        elif diff[2] == 0:
            # contact to dot
            if v[1] > u[1]:
                weight = fermi(mu_D1-mu_L1,kT)
            # dot to contact
            else:
                weight = 1 - fermi(mu_D1-mu_L1,kT)
        weight = weight*energy_weight
    return weight 

def add_battery_edges(G,physics,weight):
    (mu_L1,mu_L2,V,K,kT) = physics

    cond1 = (mu_L1 < mu_L2)
    for u in list(G.nodes()):
        for v in list(G.nodes()):
            # not a battery edge since number on dot changes
            if u[1] != v[1]:
                pass
            # electron passes from left to right
            elif cond1:
                if u[0] > v[0]:
                    G.add_edge(u,v,weight=weight)
                    nx.set_node_attributes(G,'battery_node',{u : 'True'})
            # electron passes from right to left
            else:
                if u[0] < v[0]:
                    G.add_edge(u,v,weight=weight)
                    nx.set_node_attributes(G,'battery_node',{u : 'True'})
    return G


def generate_graph(model,physics,method):
    G = nx.DiGraph()
    # set all nodes as non-battery nodes
    nx.set_node_attributes(G,'battery_node',False)
    V = Queue.Queue()
    V.put((0,0,0))

    while not V.empty():
        v = V.get()
        G.add_node(v)
        N = generate_neighbours(v, model)
        for n in N:
            # non-optimal. TODO: Find a better strategy
            if n not in list(G.nodes()):
                V.put(n)
                G.add_node(n) 
            # Catch here : Put in the weight even if node exists, because weights might not be added
            # put in weight information
            # finally, Physics, Yay!
            G.add_edge(v,n,weight=find_weight(v,n,physics,method))
            G.add_edge(n,v,weight=find_weight(n,v,physics,method))
   
    battery_weight = 10000
    G = add_battery_edges(G,physics,battery_weight)
    return G 

