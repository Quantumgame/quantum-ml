# Mask class to handle classfication of the potential landscape into leads, barriers and dots.
# Physics is not done here! This is a just a higher level wrapper to work with masks.

import exceptions

class Mask:
    '''
    Mask class is used by ThomasFermi class for handling dot classfication. There are two attributes : mask  and mask_info.

    mask is a list of the size of the x-grid which labels each points as 'l' (lead), 'b' (barrier) and 'd' (dot). This is a lower level data structure that is NOT used for interfacing with other modules.

    mask_info is a dictionary which stores the same info as mask but in processed form. It has keys as 'l0','l1' (leads), 'bi' (barriers)(i = 0,1,2,...) and 'di' (dots)( i = 0,1,2,...). The key values are 2-tuples (start_index,end_index) which starting for the index in x-grid at which the feature begins and ends. (Note that the end is included in end)

    '''

    def __init__(self,n,eps = 1e-8):
        '''
        It takes in electron density and finds the mask and mask_info by itself without requiring explicit method calls.
        '''
        self.mask = []
        self.mask_info = {}
        self.calculate_mask_from_n(n,eps)
        self.calculate_mask_info_from_mask()

    def calculate_mask_from_n(self,n,eps=1e-8):
        '''
        This function uses a preliminary idea that n > eps is a lead or a dot region, while the rest is a barrier.
        '''
        n = list(n) 
        # All regions are classified as a dot or a barrier. 
        tmp_mask = ['d' if x > eps else 'b' for x in n]
        # Then the first and the last regions are labelled as leads.
        try:
            # Lead 0
            l0_start = 0
            # The end of the lead 0 is found by finding the beginning of the first barrier.
            l0_end = tmp_mask.index('b') - 1
           
            # + 1 has to be included since the end of the range is non-inclusive
            tmp_mask[l0_start:(l0_end + 1)] = 'l' * (l0_end + 1 - l0_start)

            # Lead 1
            l1_end = len(tmp_mask) - 1
            # The starting of the lead 1 is found by finding the end of the last barrier. 
            # This is done by looking for the first 'b' while trasnversing from the end of the tmp_mask
            # Note that when searching in the reversed list, the index returned is counted from the end of the list. It has to be converted to be an index from the start of the list.
            l1_start = len(tmp_mask) - tmp_mask[::-1].index('b')  
            
            # + 1 has to be included since the end of the range is non-inclusive
            tmp_mask[l1_start:(l1_end + 1)] = 'l' * (l1_end + 1 - l1_start)
        except ValueError:
           raise exceptions.NoBarrierState

        self.mask = tmp_mask 
   
    def calculate_mask_info_from_mask(self):
        '''
        This function processes the information from mask into a dictionary for use by other modules.
        '''
        # important to set this to empty, else values from the last mask_info will not be removed
        tmp_mask_info = {}
        # Load the information about the leads. 
        # The strategy is similar as in calculate_mask_from_n.
        try: 
            # Lead 0
            l0_start = 0
            # The end of the lead 0 is found by finding the beginning of the first barrier.
            l0_end = self.mask.index('b') - 1
         
            tmp_mask_info['l0'] = (l0_start,l0_end)
            
            # Lead 1
            l1_end = len(self.mask) - 1
            # The starting of the lead 1 is found by finding the end of the last barrier. 
            # This is done by looking for the first 'b' while trasnversing from the end of the self.mask
            # Note that when searching in the reversed list, the index returned is counted from the end of the list. It has to be converted to be an index from the start of the list.
            l1_start = len(self.mask) - self.mask[::-1].index('b')  
            
            tmp_mask_info['l1'] = (l1_start,l1_end)

        except ValueError:
            raise NoBarrierState

        # Now find the information about the dots and barriers.
        dot_index = 0
        barrier_index = 0
        # index keeps track of the position in the mask until which the mask has been processed.
        # It is initially set to l0_end + 1, the beginning of the first barrier.
        index = l0_end + 1
        try:
            # find all the barriers and dots sequentially
            # The basic idea is each barrier is followed by a dot or a lead. Each dot is followed by a barrier
            # The lead followed by a barrier case is handled in the except clause.
            while(True):
                # find a barrier
                # index has to be added since the find function return index relative to the sliced array and not the original array
                new_index = index + self.mask[index:].index('d')
                barrier_key = 'b' + str(barrier_index)
                # -1 because new_index points to the beginnig of the dot
                tmp_mask_info[barrier_key] = (index,new_index - 1)
                barrier_index += 1
                index = new_index
                
                # find a dot 
                new_index = index + self.mask[index:].index('b')
                # index has to be added since the find function return index relative to the sliced array and not the original array
                dot_key = 'd' + str(dot_index)
                tmp_mask_info[dot_key] = (index,new_index - 1)
                dot_index += 1
                index = new_index

                # and then repeat

        except ValueError:        
            # either there was something wrong with the mask or the barrier followed by a lead case has arose i.e no dots exists and we are at the last barrier before the last lead.
            new_index = index + self.mask[index:].index('l')
            barrier_key = 'b' + str(barrier_index)
            # -1 because new_index points to the beginnig of the dot
            tmp_mask_info[barrier_key] = (index,new_index - 1)

        # add the num_dot value
        tmp_mask_info['num_dot'] = dot_index

        #if(tmp_mask_info['num_dot'] == 0):
        #    raise exceptions.InvalidChargeState

        self.mask_info = tmp_mask_info
   
    def calculate_new_mask_turning_points(self,V,mu_l,mu_d):
        '''
        This function calculates a new mask from scratch using the turning points where V = mu. Note that no preliminary mask is necessary.

        The mask_info from the mask has to be calculated again and explicitly.

        It is not necessary to have run the calculate_mask_from_n, since the mask size is determined from len(V).
        '''
        # treat all points as barriers, then selectively we change the ones which are leads and which are dots
        tmp_mask = ['b'] * len(V)
      
        try:   
            # find the turning point for the left lead
            i_left = 0
            # in the edge case that mu_l[0] > V(x) for all x, all the landscape is only a lead
            while(V[i_left] <= mu_l[0]):
                i_left += 1
            # this loop will exit for the first i such that V[i] > mu_l[0]
            # lead1 will end at i-1
             
            tmp_mask[:i_left] = 'l' * (i_left)
            
            # find the turning point for the right lead
            i_right = len(V)-1
            while(V[i_right] <= mu_l[1]):
                i_right -= 1
            # this loop exits for the first i from the right such that V[i] > mu_l[1]
            # lead2 starts at i + 1
            tmp_mask[i_right + 1:] = 'l' * (len(tmp_mask) - i_right - 1) 
            
            n_index_dot = 0
            i_dot = i_left
            while(n_index_dot < len(mu_d)):
                if V[i_dot] > mu_d[n_index_dot]:
                    while(V[i_dot] > mu_d[n_index_dot]):
                        i_dot += 1
                    # now the left turning point of the dot has been found
                    i_dot_start = i_dot
                    while(V[i_dot] <= mu_d[n_index_dot]):
                        i_dot += 1
                    i_dot_end = i_dot - 1
                    tmp_mask[i_dot_start:i_dot_end + 1] = 'd' * (i_dot_end - i_dot_start + 1)
                    n_index_dot += 1
                else:
                    while(V[i_dot] < mu_d[n_index_dot]):
                        i_dot += 1
                    while(V[i_dot] > mu_d[n_index_dot]):
                        i_dot += 1
                    i_dot_start = i_dot
                    while(V[i_dot] <= mu_d[n_index_dot]):
                        i_dot += 1
                    i_dot_end = i_dot - 1
                    tmp_mask[i_dot_start:i_dot_end + 1] = 'd' * (i_dot_end - i_dot_start + 1)
                    n_index_dot += 1
        
        except IndexError:
           raise exceptions.InvalidChargeState 

        self.mask = tmp_mask
