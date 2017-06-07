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
    

def get_bar_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        bar_info : (Dictonary) key = barrier_number, value = [barrier_begin, barrier_end]
    '''

    bar_info = {}
    n_bar = 0
    index = 0
    # treat the leads as dots to simplify the code structure
    mask = map(lambda x: 'd' if (x == 'l1' or x == 'l2') else x, mask)
    while(index < len(mask)):
        try:
            # the basic idea is find a 'b' point, then look for the next 'd' points, the points in between correspond to a barrier
            index = index + mask[index:].index('b')
            bar_begin = index
            index = index + mask[index:].index('d')
            bar_end = index - 1
            bar_info[n_bar] = [bar_begin,bar_end]
            n_bar += 1
        # an axception is raised when no 'b' exists
        except ValueError:
            break

    return bar_info

def get_lead_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        lead_info : (Dictonary) key = lead_number (0,1), value = [lead_begin, lead_end]
    '''

    lead_info = {}
    l1_start = mask.index('l1')
    # clever way to find the end
    l1_end = len(mask) - mask[::-1].index('l1') - 1
    lead_info[0] = [l1_start,l1_end]

    l2_start = mask.index('l2')
    # clever way to find the end
    l2_end = len(mask) - mask[::-1].index('l2') - 1
    lead_info[1] = [l2_start,l2_end]

    return lead_info
    

