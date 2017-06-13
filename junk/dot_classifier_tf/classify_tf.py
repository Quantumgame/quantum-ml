# Module to classify a given potential profile using Thomas-Fermi

import numpy as np

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

def add_leads(mask):
    '''
    Input:
    mask :  prelim mask with only 'b' or 'd'
    Output:
    mask : new mask in which the first and last islands are labelled as 'l'
    '''
    # lead1
    i = 0
    while(i < len(mask) and mask[i] != 'b'):
        mask[i] = 'l' 
        i += 1
    # lead2
    i = len(mask) - 1
    while(i > 0 and mask[i] != 'b'):
        mask[i] = 'l' 
        i -= 1

    return mask

def get_dot_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        dot_info : (Dictonary) key = dot_number, value = [dot_begin, dot_end]
    '''
    dot_info = {}
    n_dot = 0
    index = 0
    while(index < len(mask)):
        try:
            index = index + mask[index:].index('d')
            dot_begin = index
            index = index + mask[index:].index('b')
            dot_end = index - 1
            dot_info[n_dot] = [dot_begin,dot_end]
            n_dot += 1
        # an axception is raised when no 'd' exists
        except ValueError:
            break

    if len(dot_info) == 0:
        raise ValueError('No dot!')

    return dot_info
    

def get_lead_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        lead_info : (Dictonary) key = lead_number (0,1), value = [lead_begin, lead_end]
    '''

    lead_info = {}
    l1_start = mask.index('l')
    # clever way to find the end
    l1_end = len(mask) - mask[::-1].index('l') - 1
    lead_info[0] = [l1_start,l1_end]

    l2_start = mask.index('l')
    # clever way to find the end
    l2_end = len(mask) - mask[::-1].index('l') - 1
    lead_info[1] = [l2_start,l2_end]

    return lead_info

def get_mu_x(mask,mu_v,mu_d):
    '''
    Input:
        mask : vector of length V, each point is either 'd','l' or 'b'
        mu_v : potential on the leads, assumed to be constant
        mu_dot : chemical potential of the dots, vector of size n_dot
    Output:
        mu_x : chemical potential as a function of x, according to the mask
    '''
    lead_info = get_lead_info(mask)
    dot_info  = get_dot_info(mask)

    mu_x = np.zeros(len(mask))
    for i in range(len(lead_info)):
        mu_x[lead_info[i][0] : lead_info[i][1] + 1] = mu_v 

    for i in range(len(dot_info)):
        mu_x[dot_info[i][0] : dot_info[i][1] + 1] = mu_d[i] 

    return mu_x

def get_dot_charges(n,dot_info):
    '''
    Input:
    n : charge density
    dot_info : dictionary with dot starting and end points
    Output:
    N_dot : vector of size n_dot with n(x) summed over each dot
    '''
    n_dot = len(dot_info)
    N_dot = np.zeros(n_dot)
    for i in range(n_dot):
       N_dot[i] = np.sum(n[dot_info[i][0]:dot_info[i][1]+1]) 

    return N_dot

def solve_tf(V,K,C_k,N_dot,mu_v):
    '''
    Input:
        V : Potential profile
        K : interaction matrix
        N_dot : number of electrons on the dots, vector of size n_dot
        mu_v : potential on the leads, assumed to be constant
    Output:
        n : chrage density
        mu_dot : chemical potential of the dots, vector of size n_dot
        mask : vector of length V, each point is either 'd','l1','l2' or 'b'
    '''
    # tolerance in electron density
    eps_n = 1e-2
    # tolerance in dot_number
    eps_N = 1e-1
    # delta_mu is the change in mu_d per iteration
    delta_mu = mu_v/100


    # number of dots
    n_dot = len(N_dot)

    # at the first step, set all dot potentials equal to the lead potential
    mu_d = mu_v * np.ones(n_dot)

    # assume no electron density everywhere
    n = np.zeros(len(V),dtype=np.complex64)
    # assume everything is an island
    mask = ['d']*len(V)
    mask[0] = 'l'
    mask[1] = 'b'
    mask[-1] = 'l'
    mask[-2] = 'b'
 
    # iter variable to keep count on number of iterations 
    iter_count = 0 
    iter_max = 100

    mu_old = np.zeros(n_dot)
    N_old = np.zeros(n_dot)

    while(iter_count < iter_max):
        mu_x = get_mu_x(mask,mu_v,mu_d)
      
        err = 1.0 
        n_iter_count = 0
        while(err > eps_n and n_iter_count < 1000):
            n = np.asarray(n,dtype=np.complex64)
            n_old = n
            n = np.power(np.real(np.sqrt(3.0/(5*C_k) * (mu_x - V - np.dot(K,n)))),3)
            err = np.linalg.norm((n-n_old))
            n_iter_count += 1

        n = np.real(n)

        import pdb;pdb.set_trace()
        # prelim mask with only 'd' or 'b' 
        prelim_mask = map(lambda x : 'd' if x > 0.0 else 'b',n)
        # add leads to the prelim mask
        mask = add_leads(prelim_mask)

        dot_info = get_dot_info(mask)
        # calculated charges on the dots
        N_c = get_dot_charges(n,dot_info)

        if (iter_count > 0): 
            dmu_dN = (mu_d - mu_old)/(N_c - N_old)
            delta_mu = (N_dot - N_c)*dmu_dN 

        N_old = np.copy(N_c) 
        mu_old = np.copy(mu_d)
        
        if (np.linalg.norm(N_c - N_dot) < eps_N):
            break;
        else:
            for i in range(n_dot):
                # adjust the dot potential to increase or decrese the electron number on each dot
                if(iter_count == 0):  
                    mu_d += np.sign(N_dot - N_c)*delta_mu
                else:
                    mu_d += delta_mu
        iter_count += 1

    return n,mu_d,mask
                
            
