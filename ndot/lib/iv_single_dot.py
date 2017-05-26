# basic IV test routine for single point single dot

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import dot_classifier
reload(dot_classifier)
import thomas_fermi
reload(thomas_fermi)
import markov_graph
reload(markov_graph)

N_v_points = 500
V_d_vec = np.linspace(3e-3,5e-3,N_v_points)
I_vec = np.zeros(N_v_points)

x = np.arange(3)
K = thomas_fermi.create_K_matrix(x,E_scale=1e-3)

for i in range(N_v_points):
    V = np.array([5e-3,V_d_vec[i],5e-3])
    mu_l = (10e-3,10e-3)
    
    graph_model = (5,1)
    battery_weight = 1000
    kT = 10e-6
    physics = (x,V,K,mu_l,battery_weight,kT)
    G = markov_graph.generate_graph(graph_model, physics)

    I_vec[i] = markov_graph.get_current(G)

print G.nodes(data=True)
print G.edges(data=True)

plt.figure(1)
nx.draw_shell(G,with_labels=True)

plt.figure(2)
plt.plot(V_d_vec,I_vec)
