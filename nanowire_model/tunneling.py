# Module to calculate tunnel rates between barries as well as attempt rates
# Uses WKB method for the tunnel probability calculation 
# attempt rate is calculated in a semi-classical way

import scipy.integrate

import numpy as np
import thomas_fermi
import dot_classifier

def calculate_tunnel_prob(v,u,physics,n,mu,mask):
    '''
    Input:
        v : start node
        u : end node
        physics : (x,V,K,mu_l,battery_weight,kT)
        n : electron density as a function of x
        mu : chemical potential as a function of x
        (n,mu) are from the starting node

    Output:
        tunnel_prob : (under WKB approximation)
    '''
    # find where the transition is occuring
    u = np.array(u)
    v = np.array(v)
    diff = u - v

    index_electron_to = np.argwhere(diff == 1.0)[0,0]
    index_electron_from = np.argwhere(diff == -1.0)[0,0]

    (x,V,K,mu_l,battery_weight,kT) = physics

    # clever way to find the barrier index
    bar_index = np.floor(0.5*(index_electron_to + index_electron_from))
    bar_key = 'b' + str(bar_index)
   
    # chemical_potential = energy of the electron 
    mu_e = mu[index_electron_from]

    # integral
    bar_begin = mask.mask_info[bar_key][0]
    bar_end = mask.mask_info[bar_key][1]

    V_eff = V + np.dot(K,n)
    factor = scipy.integrate.simps(np.sqrt(np.abs(V_eff[bar_begin:bar_end+1] - mu_e)),x[bar_begin:bar_end+1])
    # check this, units of E(eV) and dx(nm)
    scale = 10
    tunnel_prob = np.exp(-scale*factor)
    return tunnel_prob

def calculate_attempt_rate(v,u,physics,n,mu,mask):
    '''
    Input:
        v : start node
        u : end node
        physics : (x,V,K,mu_l,battery_weight,kT)
        n : electron density as a function of x
        mu : chemical potential as a function of x
        (n,mu) are from the starting node

    Output:
        attempt_rate : calculate simply as the semi-classical travel time
    '''
    u = np.array(u)
    v = np.array(v)
    diff = u - v

    index_electron_to = np.argwhere(diff == 1.0)[0,0]
    index_electron_from = np.argwhere(diff == -1.0)[0,0]

    (x,V,K,mu_l,battery_weight,kT) = physics

    if index_electron_from == 0 or index_electron_from == (len(u) - 1): 
        #transport from leads
        attempt_rate = 1e-2
    else:
        u[index_electron_from] -= 1
        N_dot_1 = u[1:-1] 
        n1,mu1 = thomas_fermi.solve_thomas_fermi(x,V,K,mu_l,N_dot_1)

        dot_index = index_electron_from
        dot_key = 'd' + str(dot_index)
        dot_begin = mask.mask_info[dot_key][0]
        dot_end = mask.mask_info[dot_key][1]
        attempt_rate = np.sqrt(mu[dot_index] - mu1[dot_index])/(dot_end - dot_begin + 1) 
    return attempt_rate
    
     








