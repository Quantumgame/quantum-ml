import numpy as np
import matplotlib.pyplot as plt

# model-parameters
V_L1 = 3.0
V_L2 = 3.0

mu_L1 = 10.0
mu_L2 = 10.0

V_L = np.array([V_L1,V_L2]).T
mu_L = np.array([mu_L1,mu_L2]).T

V_11 = 5.0
V_12 = 2.0
V_13 = 1.0
K = np.array([[V_11,  V_12,  V_13, 0],
              [V_12,  V_11,  V_12,  V_13],
              [V_13, V_12,  V_11,  V_12], 
              [0.0,  V_13, V_12, V_11]])

# matrices defined for ease of notation
A = np.array([[K[0,1]/K[0,0],K[0,2]/K[0,0]],[K[3,1]/K[3,3],K[3,2]/K[3,3]]])

B = np.array([[K[1,0],K[1,3]],[K[2,0],K[2,3]]])
C = np.array([[K[1,1],K[1,2]],[K[2,1],K[2,2]]])

# this array will store the list of points which correspond to the boundary in (V_D1, V_D2) space
boundary = []
# number of electron on the two dots
# equivalent equation for mu_D = (vec(V_D) - mat(B)*vec(V_L)) + mat(B)*vec(mu_L) + mat(BA-C)*vec(n_D)
# define vec(a) = (vec(V_D) - mat(B)*vec(V_L))
#        vec(b) = mat(BA-C)*vec(n_D)
# these are constants for fixed n_D

N_D_lim = 4
for n_D1 in np.arange(N_D_lim):
    for n_D2 in np.arange(N_D_lim):
        n_D = np.array([n_D1,n_D2]).T


        # create the grid in V_D space
        N = 10
        x = np.linspace(-3,3,N)
        y = np.linspace(-3,3,N)
        coords = np.array(np.meshgrid(x,y)).transpose([1,2,0]).reshape(-1,2,1)

        # the idea is go over the list of mu_L, determine mu_D, see if the point lies on the boundary 
        for V_D in coords:
            a = V_D - np.dot(B,V_L).reshape((2,1))
            b = np.dot((np.dot(B,A) - C),n_D)
            mu_D = a.reshape((2,1)) + b.reshape((2,1)) + np.dot(B, mu_L).reshape((2,1))
            print mu_D
            # boradening to capture the transitions points
            # mu_L1 = mu_L2 set, so we only need to check 2 sets of conditions
            delta = 1
            cond1 = np.abs(mu_D[0,0] - mu_L[0]) < delta  
            cond2 = np.abs(mu_D[1,0] - mu_L[0]) < delta  
            
            if (cond1 or cond2):
                boundary += [[V_D[0,0],V_D[1,0],n_D1 + n_D2]]

boundary = np.array(boundary)
print boundary
plt.scatter(boundary[:,0],boundary[:,1],c=boundary[:,2],s=15)
plt.grid(True)
plt.show()
