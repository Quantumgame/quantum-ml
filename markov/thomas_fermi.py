# Module for Thomas-Fermi 

import numpy as np

# Coulomb interaction matrix
def calculate_K(E,N_dim,sigma = 1):
	''' 
		Calculates the K matrix based on a power law with sigma added to prevent blowup
		E : energy scale for the self interaction, K_ij = E/(sqrt(sigma^2 + (i-j)^2)
		N_dim : size of the interaction matrix
		sigma : parameter to prevent blowup for self-interaction, default = 1
	'''
	x = np.arange(N_dim)
	K = E/np.sqrt((x[:,np.newaxis] - x)**2 + sigma**2)
	return K

def solve_TF(mu_L1,mu_L2,N,V,K):
	'''
		Solves the TF equation V - mu + K n = 0 for mu_D and n along the N_D = N constraint
                Linear system for V.size unknowns : vec(n) and mu_D 

		returns mu_D,vec(n)
	'''
	N_dim = V.size

	# build up the LHS
	A = K
	a1 = -np.ones(N_dim)
	a1[0] = 0
	a1[N_dim-1] = 0

	a2 = np.ones(N_dim+1)
	a2[0] = 0
	a2[N_dim-1] = 0
	a2[N_dim] = 0

	A = np.concatenate((A,a1[:,np.newaxis]),axis=1)
	A = np.concatenate((A,[a2]))

        # build up the RHS
        b = -V
        b[0] = b[0] + mu_L1
        b[N_dim-1] = b[N_dim-1] + mu_L2
        b = np.concatenate((b,[N]))

        x = np.linalg.solve(A,b)
        return x[N_dim],x[:N_dim]
