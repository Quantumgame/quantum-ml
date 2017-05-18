# code to model a single point quantum dot, with the leads modelled as single points under the Thomas-Fermi approximation

from thomas_fermi import *
from markov_graph import *

import numpy as np
import matplotlib.pyplot as plt

# physical parameters
# Temperature kT (eV)
kT = 10e-6 

method = 'mu'
# Model parameters
# potential profile
V_L1 = 5e-3 
V_L2 = 5e-3 

N_points = 200
V_D_vec = np.linspace(3e-3,5e-3,N_points)

bias = 0.01e-3
# lead voltages
mu_L1 = 10e-3
mu_L2 = 10e-3+bias

K = calculate_K(1e-3,3)
dist_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    V = np.array([V_L1, V_D, V_L2])
    K = calculate_K(1e-3,3)

    model = (9,1)
    physics = (mu_L1,mu_L2,V,K,kT)
    G = generate_graph(model,physics,method)

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
plt.plot(V_D_vec,dist_vec)
plt.xlabel('Gate voltage')
plt.ylabel('Current or Conductance')
plt.figure(2)
nx.draw_networkx(G,with_labels=True)
plt.show()
 
