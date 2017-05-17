# code to model a single point quantum dot, with the leads modelled as single points under the Thomas-Fermi approximation
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

def find_weight(u,v,physics):
    '''
        Find the weight of edge from u to v
        Note that find_weight cannot find weight between any two vertices, it assumes that u,v are neighbours. DO NOT USE FOR FINDING WEIGHT BETWEEN ANY TWO NODES.
    '''
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


def generate_graph(model,physics):
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
            G.add_edge(v,n,weight=find_weight(v,n,physics))
            G.add_edge(n,v,weight=find_weight(n,v,physics))
   
    battery_weight = 100
    G = add_battery_edges(G,physics,battery_weight)
    return G 


# physical parameters
# Temperature kT (eV)
kT = 10e-6 # 4K

# Model parameters
# potential profile
V_L1 = 5e-3 
V_L2 = 5e-3 

# lead voltages
mu_L1 = 10e-3
mu_L2 = 11e-3


V_D_vec = np.linspace(3e-3,5e-3,1000)
K = calculate_K(1e-3,3)
dist_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    V = np.array([V_L1, V_D, V_L2])
    K = calculate_K(1e-3,3)

    model = (2,1)
    physics = (mu_L1,mu_L2,V,K,kT)
    G = generate_graph(model,physics)

    # Adjacency matrix, caution not the Markov matrix
    A = nx.to_numpy_matrix(G)
    # look at this carefully
    M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))

    w,v = np.linalg.eig(M)
    ind = np.argwhere(np.abs(w) < 1e-1).flatten()[0]
    dist = v[:,ind]/v[:,ind].sum(axis=0)

    # battery
    # TODO: Find a better way to find the indices for the battery edges
    battery_nodes = nx.get_node_attributes(G,'battery_node')
    nodes = list(G.nodes())
    battery_ind = []
    for key in battery_nodes:
        battery_ind += [nodes.index(key)]

    for ind in battery_ind:
        dist_vec[i] += dist[ind,0]

print list(G.nodes(data=True))
print list(G.edges(data=True))
plt.figure(1)
plt.plot(V_D_vec,np.gradient(dist_vec))
plt.figure(2)
nx.draw_networkx(G,with_labels=True)
plt.show()

 
