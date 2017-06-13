# playing to see the relation between the thomas-fermi energy and the dot potential

from thomas_fermi import *

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
# potential profile
V_L1 = 5e-3 
V_L2 = 5e-3 

# lead voltages
mu_L1 = 10e-3
mu_L2 = 10.1e-3

# mu_D vs V_D
V_D_vec = np.linspace(0,5e-3,100)

N = 0
K = calculate_K(1e-3,3)
mu_D0_vec = np.zeros(V_D_vec.size)
E_TF0_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    mu_D, n = solve_TF(mu_L1,mu_L2,N,V,K)
    mu_D0_vec[i] = mu_D
    E_TF0_vec[i] = calculate_E_TF(mu_L1,mu_L2,mu_D,n,V,K) 

N = 1
K = calculate_K(1e-3,3)
mu_D1_vec = np.zeros(V_D_vec.size)
E_TF1_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    mu_D, n = solve_TF(mu_L1,mu_L2,N,V,K)
    mu_D1_vec[i] = mu_D
    E_TF1_vec[i] = calculate_E_TF(mu_L1,mu_L2,mu_D,n,V,K) 

N = 2
K = calculate_K(1e-3,3)
mu_D2_vec = np.zeros(V_D_vec.size)
E_TF2_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    mu_D, n = solve_TF(mu_L1,mu_L2,N,V,K)
    mu_D2_vec[i] = mu_D
    E_TF2_vec[i] = calculate_E_TF(mu_L1,mu_L2,mu_D,n,V,K) 

f, ax = plt.subplots(1)
ax.plot(V_D_vec,E_TF0_vec)
ax.plot(V_D_vec,E_TF1_vec)
ax.plot(V_D_vec,E_TF2_vec)
ax.set_xlabel('Gate voltage')
ax.set_ylabel('Thomas-Fermi energy')
plt.show()
