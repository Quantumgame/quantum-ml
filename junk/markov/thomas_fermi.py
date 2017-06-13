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

def calculate_E_TF(mu_L1,mu_L2,mu_D,n,V,K):
    '''
        Calculates the Thomas-Fermi energy
        E_TF = V n + 1/2 n K n
        Note that it does not include mu, so this is actual energy and not the free energy 
    '''
    N_dim = V.size

    # constructing the chemical potential vector, = mu_L at the leads, and mu_D inside the dot
    mu_vec = mu_D*np.ones(N_dim)
    mu_vec[0] = mu_L1
    mu_vec[-1] = mu_L2

    E_TF = np.sum(V*n) + 0.5*np.sum(n*np.dot(K,n))
    return E_TF 
