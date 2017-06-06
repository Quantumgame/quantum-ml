# Module to solve the Thomas-Fermi problem in 1D

import numpy as np
import dot_classifier

def create_K_matrix(x, E_scale=1.0, sigma=1.0):
    '''
    Input: 
        x : discrete 1D grid
        E_scale : energy scale for the K matrix, default units is eV
        sigma : impact paramter to prevent blow up at the same point
    Output:
        K : matrix of size x.size times x.size

    K(x1,x2) = E_scale / sqrt((x1 - x2)^2 + sigma^2)
    '''
    K = E_scale/np.sqrt((x[:, np.newaxis] - x)**2 + sigma**2)
    return K
    
def create_A_matrix(x,V,K,mu_l,N_dot,mask,dot_info):
    '''
    Convinience function
    Input:
        x    : discrete 1D grid
        V    : potential
        K    : Coulomb interaction matrix between two points
        mask : array specifying the nature of each point 'l1','d','b' or 'l2'
        dot_info : information about number of dots and their start and end points
    Output:
        A : matrix A used in solution of TF, A z = b 
    '''

    #set up the A matrix
    N_points = x.size
    A = np.zeros([2*N_points,2*N_points])

    # top left half for n
    for i in range(N_points):
        for j in range(N_points):
            A[i,j]  = K[i,j]
    # top right half for the chemical potentials
    for i in range(N_points):
        A[i,i + N_points] = -1

    # for leads and barrier
    # add the N_points number of constraints, index_constraint keeps track of number of constraint added
    index_constraint = 0
    index_dot = -1
    while (index_constraint < N_points):
        if (mask[index_constraint] == 'l1' or mask[index_constraint] == 'l2'):
            A[index_constraint + N_points,index_constraint + N_points] = 1
            index_constraint += 1
        elif(mask[index_constraint] == 'b'):
            A[index_constraint+N_points,index_constraint] = 1
            index_constraint += 1
        else:
            # mask is a dot
            index_dot += 1
            value = dot_info[index_dot]
            dot_begin = value[0]
            dot_end = value[1]
            dot_size = dot_end - dot_begin + 1
            for j in range(dot_size):
                A[index_constraint + N_points, dot_begin + j] = 1   
            # now the number of electrons in a dot is fixed constraint has been added
            index_constraint += 1

            # add the mu is same over the dot constraint
            # note that there are only dot_size - 1 constraints of this type
            for j in range(dot_size-1): 
                A[index_constraint + N_points, index_constraint + N_points - 1] = 1 
                A[index_constraint + N_points, index_constraint + N_points] = -1 
                index_constraint += 1

    return A

def create_b_matrix(x,V,K,mu_l,N_dot,mask,dot_info):
    '''
    Convinience function
    Input:
        x    : discrete 1D grid
        V    : potential
        K    : Coulomb interaction matrix between two points
        mask : array specifying the nature of each point 'l1','d','b' or 'l2'
        dot_info : information about number of dots and their start and end points
    Output:
        b : vector b used in solution of TF, A z = b 
        mu_l : (mu_L1, mu_L2) tuple with the lead potentials
        N_dot: vector with number of electrons in each dot, can be of size 0 i.e no dot
    '''


    N_points = x.size


    # set up the RHS
    b = np.zeros(2*N_points)
    for i in range(N_points):
        b[i] = -V[i]

    # lead constraints
    index_constraint = 0
    index_dot = -1

    while (index_constraint < N_points):
        if (mask[index_constraint] == 'l1'): 
            b[index_constraint + N_points] = mu_l[0]
            index_constraint += 1
        elif (mask[index_constraint] == 'l2'):
            b[index_constraint + N_points] =  mu_l[1]
            index_constraint += 1
        elif (mask[index_constraint] == 'b'):
            # actually do nothing since b is already 0
            b[index_constraint + N_points] = 0
            index_constraint += 1
        else:
            index_dot += 1
            value = dot_info[index_dot]
            dot_begin = value[0]
            dot_end = value[1]
            dot_size = dot_end - dot_begin + 1
            for j in range(dot_size):
                b[index_constraint + N_points] = N_dot[index_dot]   
            # now the number of electrons in a dot is fixed constraint has been added
            index_constraint += 1

            # add the 'mu is same over the dot' constraint
            # note that there are only dot_size - 1 constraints of this type
            for j in range(dot_size-1): 
                # again, do nothing since b is already 0
                b[index_constraint + N_points] = 0
                index_constraint += 1
    return b

def solve_thomas_fermi(x,V,K,mu_l,N_dot):
    '''
    Input:
        x    : discrete 1D grid
        V    : potential
        K    : Coulomb interaction matrix between two points
        mu_l : (mu_L1, mu_L2) tuple with the lead potentials
        N_dot: vector with number of electrons in each dot, can be of size 0 i.e no dot
    Output:
        (n, mu) where
        n    : electronic charge density as a function of x
        mu   : chemical potential as a function of x
               mu(x) = mu_L when x in leads
    
    Solves the Thomas-Fermi equation V - mu + K n = 0 along with the constraint that integral of electron density in a dot is a constant and electron density in the barrier region is zero.
    '''
    #solve the equation A z = b
    # z = (n mu)^T
    N_points = x.size
    
    mask = dot_classifier.get_mask(x,V,K,mu_l[0])
    # dictionary index by dot number, gives [dot_begin_index,dot_end_index]
    dot_info = dot_classifier.get_dot_info(mask)
    A = create_A_matrix(x,V,K,mu_l,N_dot,mask,dot_info)
    b = create_b_matrix(x,V,K,mu_l,N_dot,mask,dot_info)
    z = np.linalg.solve(A,b)

    n,mu = z[:N_points],z[N_points:]

    # trying out one iteration to improve the island definition
    # mu_new as average over the dot potentials
    #mu_new = mu_l[0]
    #for i in range(len(dot_info)):
    #    mu_new += mu[dot_info[i][0]] 
    #mu_new = (1.0/(len(dot_info) + 1))*mu_new

    #mask = dot_classifier.get_mask(x,V,K,mu_new)
    ## dictionary index by dot number, gives [dot_begin_index,dot_end_index]
    #dot_info = dot_classifier.get_dot_info(mask)
    # 
    #
    #A = create_A_matrix(x,V,K,mu_l,N_dot,mask,dot_info)
    #b = create_b_matrix(x,V,K,mu_l,N_dot,mask,dot_info)
    #z = np.linalg.solve(A,b)

    #n,mu = z[:N_points],z[N_points:]

    # return n,mu
    return n,mu
     
def calculate_thomas_fermi_energy(V,K,n,mu):
    '''
    Input: 
        V : potential profile
        K : Coulomb interaction matrix 
        n : electorn density
        mu : chemical potenial profile, includes leads and barriers as well
    Output:
        E : Thomas-Fermi energy

    E = V n + 1/2 n K n
    '''
    E = np.sum(V*n) + 0.5 * np.sum(n*np.dot(K,n))
    return E


    
