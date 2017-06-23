# Module to calculate tunnel rates between barries as well as attempt rates
# Uses WKB method for the tunnel probability calculation 
# attempt rate is calculated in a semi-classical way

import scipy.integrate

import numpy as np
import thomas_fermi

def calculate_tunnel_prob(v,u,tf,tf_strategy,tf_solutions):
    '''
    Input:
        v : start node
        u : end node
        tf : ThomasFermi object (n,mu) are calculated again

    Output:
        tunnel_prob : (under WKB approximation)
    '''
    # find where the transition is occuring
    u = np.array(u)
    v = np.array(v)
    diff = u - v

    index_electron_to = np.argwhere(diff == 1.0)[0,0]
    index_electron_from = np.argwhere(diff == -1.0)[0,0]

    # clever way to find the barrier index
    bar_index = np.floor(0.5*(index_electron_to + index_electron_from))
    bar_key = 'b' + str(int(bar_index))
  
    if (v[1:-1] not in tf_solutions):
        # solve tf for start node 
        n,mu = tf.tf_iterative_solver_fixed_N(v[1:-1],strategy=tf_strategy)
        E = self.tf.calculate_thomas_fermi_energy(n,mu)
        tf_solutions[v[1:-1]] = {'n':n,'mu':mu,'E':E}
    else:
        n = self.tf_solutions[v[1:-1]]['n']
        mu = self.tf_solutions[v[1:-1]]['mu']
        E = self.tf_solutions[v[1:-1]]['E']
        
    # chemical_potential = energy of the electron 
    # add in the lead potentials to mu to simplify notation
    mu = np.concatenate((np.array([tf.mu_l[0]]),mu,np.array([tf.mu_l[1]]))) 
    mu_e = mu[index_electron_from]

    # integral
    bar_begin = tf.mask.mask_info[bar_key][0]
    bar_end = tf.mask.mask_info[bar_key][1]

    # in the barrier region, since n = 0, the effective potential is almost just V
    V_eff = tf.V + np.dot(tf.K,n)
    
    factor = scipy.integrate.simps(np.sqrt(np.abs(V_eff[bar_begin:bar_end+1] - mu_e)),tf.x[bar_begin:bar_end+1])

    # calcualte the scale based on physics in tf
    scale = tf.WKB_scale   
    
    tunnel_prob = np.exp(-scale*factor)
    return tunnel_prob

def calculate_attempt_rate(v,u,tf,tf_strategy):
    '''
    Input:
        v : start node
        u : end node
        tf : Thomas Fermi object

    Output:
        attempt_rate : calculate simply as the semi-classical travel time
    '''
    u = np.array(u)
    v = np.array(v)
    diff = u - v
    
    # solve for start node state
    n,mu = tf.tf_iterative_solver_fixed_N(v[1:-1],strategy=tf_strategy)

    index_electron_to = np.argwhere(diff == 1.0)[0,0]
    index_electron_from = np.argwhere(diff == -1.0)[0,0]

    if index_electron_from == 0:  
        #transport from leads
        # calculate the attempt rate from the dot
        index_electron_from = 1
    if index_electron_from == len(v) - 1:  
        #transport from leads
        # calculate the attempt rate from the dot
        index_electron_from = len(v) - 2
    
    v[index_electron_from] -= 1
    N_dot_1 = u[1:-1] 
    n1,mu1 = tf.tf_iterative_solver_fixed_N(N_dot_1,strategy=tf_strategy)

    # correct for the fact that leads are included in v
    # so 1 = index_electron_from corresponds to 'd0'
    dot_index = index_electron_from - 1
    dot_key = 'd' + str(int(dot_index))
    dot_begin = tf.mask.mask_info[dot_key][0]
    dot_end = tf.mask.mask_info[dot_key][1]
    attempt_rate = np.sqrt(np.abs(mu[dot_index] - mu1[dot_index]))/(dot_end - dot_begin + 1) 

    return attempt_rate
    
     








