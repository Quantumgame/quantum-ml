# Module to generate a markov graph based on charge states of quantum dots

import numpy as np
import Queue
import networkx as nx
import pdb

#local modules
import dot_classifier
import thomas_fermi
import tunneling
import rank_nullspace

def check_validity(v, graph_model):
    '''
    Input:
        v : node to check validity of
        graph_mode : (p,q)

    Output:
        True/False

    Constraints:
    1. 0 <= N_D <= p
    2. abs(N_L-N_R) <= q
    3. N_L + N_D + N_R = 0
    '''
    (p,q) = graph_model
    
    cond1 = True
    num_dots = len(v) - 2
    for i in range(1,num_dots+1):
        cond1 =  cond1 and (0 <= v[i] <= p)

    cond2 = (abs(v[0] - v[-1]) <= q)

    cond3 = (np.sum(v) == 0)

    return (cond1 and cond2 and cond3)

def generate_neighbours(v, graph_model):
    '''
    Input:
        v : node to find neighbours of
        graph_model : (p,q)
    Output:
        valid : valid set of neighbours v in the graph

        Takes in a num_dot + 2 charge state (N_L,vec N_D,N_R) model and generates neighbours in the validity of model=(p,q)
        Constraints:
        1. 0 <= N_D <= p
        2. abs(N_L-N_R) <= q
        3. N_L + N_D + N_R = 0
    '''

    (p,q) = graph_model
    neigh = []
    # TODO: zero dot case NOT handled
    num_dots = len(v) - 2 
    for i in range(1,num_dots+1):
        # 4 possible neighbours of each change in dot charge state, ld,l,rd,r 
        # the nomenclature stems from d : dagger, so ld denotes create an electron in the left and so on

        # typecasting between arrays and tuples involved here, since the nodes are stored as tuples, whereas tuples do not support item assignment
        ld = np.array(v)
        l = np.array(v)
        rd = np.array(v)
        r = np.array(v)

        ld[i - 1] += 1
        ld[i] += -1 
        neigh.append(tuple(ld))
        
        l[i - 1] += -1
        l[i] += 1 
        neigh.append(tuple(l))
        
        rd[i + 1] += 1
        rd[i] += -1 
        neigh.append(tuple(rd))
        
        r[i + 1] += -1
        r[i] += 1 
        neigh.append(tuple(r))
    valid = filter(lambda x : check_validity(x,graph_model),neigh)    
    return valid

def fermi(E,kT):
    '''
    Input:
        E : energy (eV)
        kT : temp in eV

    Output:
        fermi_function
    '''
    return 1.0/(1 + np.exp(E/kT))

def find_weight(v,u,physics):
    '''
    Input:
        v : start node
        u : end node
        physics : (x,V,K,mu_l,battery_weight,kT)

    Output:
        weight : weight of edge from v to u
    '''
    (x,V,K,mu_l,battery_weight,kT) = physics

    # number of electons in v state on the dot 
    N_dot_1 = v[1:-1] 
    n1,mu1 = thomas_fermi.solve_thomas_fermi(x,V,K,mu_l,N_dot_1)
    E_1 = thomas_fermi.calculate_thomas_fermi_energy(V,K,n1,mu1)

    N_dot_2 = u[1:-1] 
    n2,mu2 = thomas_fermi.solve_thomas_fermi(x,V,K,mu_l,N_dot_2)
    E_2 = thomas_fermi.calculate_thomas_fermi_energy(V,K,n2,mu2)

    simple_prob = fermi(E_2 - E_1,kT)
    tunnel_prob = 1.0
    #tunnel_prob = tunneling.calculate_tunnel_prob(v,u,physics,n1,mu1)
   
    attempt_rate = 1.0
    #attempt_rate = tunneling.calculate_attempt_rate(v,u,physics,n1,mu1)
    
    weight = attempt_rate*tunnel_prob*simple_prob
    return weight
    
def add_battery_edges(G,physics):
    '''
    Input: 
        G : graph
        physics : (x,V,K,mu_l,battery_weight,kT)
    Output:
        G

    Return graph G with battery edges added
    '''
    (x,V,K,mu_l,battery_weight,kT) = physics

    (mu_l1,mu_l2) = mu_l
    cond1 = (mu_l1 < mu_l2)
    for u in list(G.nodes()):
        for v in list(G.nodes()):
            # not a battery edge since number on dot changes
            if u[1:-1] != v[1:-1]:
                pass
            # electron passes from left to right
            elif cond1:
                if u[0] > v[0]:
                    G.add_edge(u,v,weight=battery_weight)
                    nx.set_edge_attributes(G,'battery_edge',{(u,v) : True})
                    nx.set_node_attributes(G,'battery_node',{u : True})
            # electron passes from right to left
            else:
                if u[0] < v[0]:
                    G.add_edge(u,v,weight=battery_weight)
                    nx.set_edge_attributes(G,'battery_edge',{(u,v) : True})
                    nx.set_node_attributes(G,'battery_node',{u : True})
    return G

def generate_graph(graph_model, physics):
    '''
    Input:
        graph_model : (p,q) : describes the constraints on the graph nodes
        physics : (x,V,K,mu_l,battery_weight,kT)
    Output:
        G : Markov graph of the charge states, weights assigned to edges using the energy method at zero bias, battery edges are added according to the battery weight paramter in physics input

    '''
    G = nx.DiGraph()
    (x,V,K,mu_l,battery_weight,kT) = physics

    # queue used for BFS generation of the graph
    Q = Queue.Queue()

    # dot classification done using the left lead potential
    mu = mu_l[0]
    mask = dot_classifier.get_mask(x,V,K,mu)
    # dictionary index by dot number, gives [dot_begin_index,dot_end_index]
    dot_info = dot_classifier.get_dot_info(mask)
    num_dots = len(dot_info)
    
    # dots + leads
    start_node =(0,) * (num_dots + 2)
    #start_node = np.zeros(num_dots + 2)
    Q.put(start_node)

    while not Q.empty():
        v = Q.get()
        G.add_node(v)
        neigh = generate_neighbours(v, graph_model)
        for n in neigh:
            # non-optimal: TODO: find a better strategy
            if n not in list(G.nodes()):
                Q.put(n)
                G.add_node(n)

            # Catch here : Put in the weight even if node exists, because weights might not be added
            # put in weight information
            # finally, Physics, Yay!
            G.add_edge(v,n,weight=find_weight(v,n,physics))
            nx.set_edge_attributes(G,'battery_edge',{(v,n) : False})
            G.add_edge(n,v,weight=find_weight(n,v,physics))
            nx.set_edge_attributes(G,'battery_edge',{(n,v) : False})
   
    G = add_battery_edges(G,physics)
    return G 
def recalculate_weights(G,physics):
    '''
    Input:
        G: Graph
        physics : (x,V,K,mu_l,battery_weight,kT)
    Output:
        G : Graph G with edge weight recalculated according to new physics
    
    TODO: BATTERY EDGES ARE NOT RECALCULATED
    '''
    edges = nx.get_edge_attributes(G,'battery_edge') 
    for key in edges:
        # not a battery edge
        if(edges[key] == False):
            nx.set_edge_attributes(G,'weight',{key:find_weight(key[0],key[1],physics)})
        # battery edge
        else:
            pass
    
    return G        

def get_battery_nodes(G):
    '''
    Input:
        G :Graph
    Output:
        battery_ind : list of battery nodes
    '''
    # battery
    # TODO: Find a better way to find the indices for the battery edges
    battery_nodes = nx.get_node_attributes(G,'battery_node')
    nodes = list(G.nodes())
    battery_ind = []
    # find the keys of the battery nodes
    for key in battery_nodes:
        battery_ind += [nodes.index(key)]

    return battery_ind

def get_prob_dist(M):
    '''
    Input:
        M : matrix
    Output:
        dist : prob normalised nullspace vector of M
    '''
    nullspace = rank_nullspace.nullspace(M,rtol=1e-5)  
    if (nullspace.shape[1] > 0):
        #non-trivial nullspace exists for M
        # dist is prob distribution
        dist = nullspace[:,0]/nullspace[:,0].sum(axis=0)
    else:
        #nullspace is trivial, in this case there is no stable prob. distribution,
        #In case raised, try changing the rtol parameter
        raise ValueError('Nullspace of Markov matrix is trivial. No probability distribution exists')
    return dist
        
def get_current(G,battery_ind):
    '''
    Input:
        G : graph with nodes as charge states and weights assigned, battery edges should also be present in G
    Output:
        current : current 

    The basic idea is to create a Markov evolution matrix from the weights. The stable probability distribution is given as the nullspace of this matrix.

    The current is calculated by summing over the probabilities at the beginning of the battery edges.
    '''

    # Adjacency matrix, caution not the Markov matrix
    A = nx.to_numpy_matrix(G)
    # look at this carefully
    M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))

    dist = get_prob_dist(M)

    # calculate the current by summing over the probabities over the battery nodes 
    current = 0
    for b_ind in battery_ind:
        current += dist[b_ind,0]

    return current

def get_max_prob_node(G):
    '''
    Input:
        G : graph with nodes as charge states and weights assigned, battery edges should also be present in G
    Output:
        Node with highest occupation probability
    '''
    # Adjacency matrix, caution not the Markov matrix
    A = nx.to_numpy_matrix(G)
    # look at this carefully
    M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))

    #w,v = np.linalg.eig(M)
    #ind = np.argwhere(np.abs(w) < 1e-1).flatten()[0]
    
    dist = get_prob_dist(M)
   
    max_prob_index = np.argmax(dist)
    nodes = list(G.nodes())
    return nodes[max_prob_index],dist
    
    
     
