from thomas_fermi import *

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
# potential profile
V_L1 = -0.5e-3 
V_L2 = -0.5e-3 

# lead voltages
mu_L1 = 1e-3
mu_L2 = 1.1e-3

# mu_D vs V_D
V_D_vec = np.linspace(0,-0.5e-3,50)
N = 1
K = calculate_K(1e-3,3)
mu_D_vec = np.zeros(V_D_vec.size)
for i in range(V_D_vec.size):
    V_D = V_D_vec[i]
    V = np.array([V_L1, V_D, V_L2])
    
    mu_D, n = solve_TF(mu_L1,mu_L2,N,V,K)
    mu_D_vec[i] = mu_D

plt.plot(V_D_vec,mu_D_vec)
plt.show()

