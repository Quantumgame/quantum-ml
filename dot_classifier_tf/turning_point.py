# find turning point routine
# classifies a given potential grid, into leads and dots given the lead potentials and the dot potentials

def find_turning_points(V,mu_l,mu_d):
    '''
    Input
    V : Potential Profile 
    mu_l : lead potentials
    mu_d : dot potentials
    
    Output:
    lead_info : dictionary indexed by lead no (1/2) with value [starting_point,end_point]
    dot_info : dictionary of size len(mu_d) with value [starting_point,end_point]
    
    Algorithm:
    The dots are identified by the classical turning points, mu = V(x)
    '''
    lead_info = {}
    dot_info = {}
    
    # find the turning point for the left lead
    i_left = 0
    while(V[i_left] <= mu_l[0] and i_left < len(V)):
    # in the edge case that mu_l[0] > V(x) for all x, all the landscape is only a lead
        i_left += 1
    # this loop will exit for the first i such that V[i] > mu_l[0]
    # lead1 will end at i-1
    lead_info[0] = [0,i_left-1]
    
    # find the turning point for the right lead
    i_right = len(V)-1
    while(V[i_right] <= mu_l[1] and i_right > -1):
        i_right -= 1
    # this loop exits for the first i from the right such that V[i] > mu_l[1]
    # lead2 starts at i + 1
    lead_info[1] = [i_right+1,len(V)-1]
    
    n_index_dot = 0
    i_dot = i_left
    while(n_index_dot < len(mu_d)):
        if V[i_dot] > mu_d[n_index_dot]:
            while(V[i_dot] > mu_d[n_index_dot]):
                i_dot += 1
            # now the left turning point of the dot has been found
            i_dot_start = i_dot
            while(V[i_dot] < mu_d[n_index_dot]):
                i_dot += 1
            i_dot_end = i_dot - 1
            dot_info[n_index_dot] = [i_dot_start,i_dot_end]
            n_index_dot += 1
        else:
            while(V[i_dot] < mu_d[n_index_dot]):
                i_dot += 1
            while(V[i_dot] > mu_d[n_index_dot]):
                i_dot += 1
            i_dot_start = i_dot
            while(V[i_dot] < mu_d[n_index_dot]):
                i_dot += 1
            i_dot_end = i_dot - 1
            dot_info[n_index_dot] = [i_dot_start,i_dot_end]
            n_index_dot += 1
        
    return lead_info,dot_info