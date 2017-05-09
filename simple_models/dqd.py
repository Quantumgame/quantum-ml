import numpy as np
import matplotlib.pyplot as plt

# model-parameters
V_L1 = 1.0
V_L2 = 1.0
V_D1 = 0.0
V_D2 = 0.0

V_L = np.array([V_L1,V_L2]).T
V_D = np.array([V_D1,V_D2]).T

K = np.array([[1.0,  0.5,  0.25, 0],
              [0.5,  1.0,  0.5,  0.25],
              [0.25, 0.5,  1.0,  0.5], 
              [0.0,  0.25, 0.5,  1.0]])

# matrices defined for ease of notation
A = np.array([[K[0,1]/K[0,0],K[0,2]/K[0,0]],[K[3,1]/K[3,3],K[3,2]/K[3,3]]])

B = np.array([[K[1,0],K[1,3]],[K[2,0],K[2,3]]])
C = np.array([[K[1,1],K[1,2]],[K[2,1],K[2,2]]])

boundary = []
# number of electron on the two dots
for n_D1 in np.arange(2):
    for n_D2 in np.arange(2):
        n_D = np.array([n_D1,n_D2]).T

        # equivalent equation for mu_D = (vec(V_D) - mat(B)*vec(V_L)) + mat(B)*vec(mu_L) + mat(BA-C)*vec(n_D)
        # define vec(a) = (vec(V_D) - mat(B)*vec(V_L))
        #        vec(b) = mat(BA-C)*vec(n_D)
        # these are constants for fixed n_D

        a = V_D - np.dot(B,V_L)
        b = np.dot((np.dot(B,A) - C),n_D)

        # create the grid in mu_L space
        x = np.linspace(-9,-2,100)
        y = np.linspace(-9,-2,100)
        coords = np.array(np.meshgrid(x,y)).transpose([1,2,0]).reshape(-1,2,1)

        # the idea is go over the list of mu_L, determine mu_D, see if the point lies on the boundary 
        for mu_L in coords:
            mu_D = a.reshape((2,1)) + b.reshape((2,1)) + np.dot(B, mu_L)
            delta = 1e-2
            cond1 = np.abs(mu_D[0,0] - mu_L[0,0]) < delta  
            cond2 = np.abs(mu_D[0,0] - mu_L[1,0]) < delta  
            cond3 = np.abs(mu_D[1,0] - mu_L[0,0]) < delta  
            cond4 = np.abs(mu_D[1,0] - mu_L[1,0]) < delta  
            
            if (cond1 or cond2 or cond3 or cond4):
                boundary += [[mu_L[0,0],mu_L[1,0],n_D1 + n_D2]]

boundary = np.array(boundary)
plt.scatter(boundary[:,0],boundary[:,1],c=boundary[:,2],s=15,cmap=plt.get_cmap('inferno'))
plt.grid(True)
plt.show()
