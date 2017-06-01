# Module to classify a grid x given a potential profile into dots, barriers and leads
# This module only has convience function around the actual dot classifer. This is done to ensure independence and choice between different dot classification schemes
# The module doing dot classification has to be imported
import numpy as np
import dot_classifier_tf 

def get_mask(x,V,K,mu):
    '''
    Input:
        x : 1D grid
        V : potential profile
        K : Coulomb matrix
        mu : lead potential
    Output:
        mask : array of x.size with each element one of 'l1','b','d','l2' standing for lead1, barrier,dot and lead2 respectively
    '''
    # CHEAP TESTING SOLUTION
    #mask = np.array(['l1','d','b','l2'])
    #mask = np.array(['l1','d','l2'])

    # the real deal
    mask = dot_classifier_tf.get_mask(x,V,K,mu)
    return mask

def get_dot_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        dot_info : (Dictonary) key = dot_number, value = [dot_begin, dot_end]
    '''
    # CHEAP TESTING SOLUTION
    #dot_info = {0 : [1,1]}
    #dot_info = {0: [1,1]}

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

    return dot_info
    


