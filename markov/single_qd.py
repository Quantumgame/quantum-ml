# code to model a single point quantum dot, with the leads modelled as single points under the Thomas-Fermi approximation

import numpy as np
import matplotlib.pyplot as plt
import Queue
import networkx as nx

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
		Linear system for K.size unknowns, vec(n) and mu_D 

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


# physical parameters
# Temperature kT (eV)
kT = 1e-3 # 4K

# Model parameters
# potential profile
V_L1 = 1.0 
V_D = 0
V_L2 = 1.0 

# lead voltages
mu_L1 = 5
mu_L2 = 5

V = np.array([V_L1, V_D, V_L2])
K = calculate_K(1,3)
mu_D,n = solve_TF(mu_L1, mu_L2,1,V,K)
print mu_D
print n


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

def fermi(E,mu,kT):
    return 1.0/(1 + np.exp((E - mu)/kT))

def find_weight(u,v,physics):
    '''
        Find the weight of edge from u to v
    '''
    (mu_L1,mu_L2,V,K,kT) = physics
    # number of electons in u state on the dot 
    N = u[1]
    mu_D,n = solve_TF(mu_L1,mu_L2,N,V,K)
   
    u = np.array(u)
    v = np.array(v)
    diff = u - v
    # transport through right contact
    if u[0] == 0:
        # contact to dot
        if v[1] > u[1]:
            weight = fermi(mu_D,mu_L2,kT)
        # dot to contact
        else:
            weight = 1 - fermi(mu_D,mu_L2,kT)
    # transport through left contact
    else:
        # contact to dot
        if v[1] > u[1]:
            weight = fermi(mu_D,mu_L1,kT)
        # dot to contact
        else:
            weight = 1 - fermi(mu_D,mu_L1,kT)

    return weight

def generate_graph(model,physics):
    G = nx.DiGraph()
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
                # put in weight information
                # finally, Physics, Yay!
                G.add_edge(v,n,weight=find_weight(v,n,physics))
                G.add_edge(n,v,weight=find_weight(n,v,physics))
    return G 

model = (1,1)
physics = (mu_L1,mu_L2,V,K,kT)
G = generate_graph(model,physics)
print list(G.nodes())
print list(G.edges(data=True))

M = nx.to_numpy_matrix(G)
M = np.nan_to_num(M/M.sum(axis=1))
print M

w,v = np.linalg.eig(M)
print w
print v
	

 
