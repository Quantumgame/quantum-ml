# Module to classify a grid x given a potential profile into dots, barriers and leads
import numpy as np

def get_mask(x,V):
    '''
    Input:
        x : 1D grid
        V : potential profile
    Output:
        mask : array of x.size with each element one of 'l1','b','d','l2' standing for lead1, barrier,dot and lead2 respectively
    '''
    # CHEAP TESTING SOLUTION
    #mask = np.array(['l1','d','d','l2'])
    mask = np.array(['l1','d','l2'])
    return mask

def get_dot_info(mask):
    ''' 
    Input:
        mask : mask as described in the get_mask function
    Output:
        dot_info : (Dictonary) key = dot_number, value = [dot_begin, dot_end]
    '''
    # CHEAP TESTING SOLUTION
    #dot_info = {0 : [1,1],1:[2,2]}
    dot_info = {0: [1,1]}
    return dot_info
    


