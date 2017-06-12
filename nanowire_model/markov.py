# Markov graph class to calculate transport properties.

import numpy as np
import queue
import networkx as nx

import thomas_fermi
import rank_nullspace

class Markov():
    '''
    Class Markov is used for graph creation and calculation of currents.
    '''

    def __init__(self,graph_model,physics):
        self.graph_model = graph_model
        self.physics = physics
        self.G = nx.DiGraph() 
    
    def check_validity(self, u):
        '''
        Input:
            u : node to check validity of

        Output:
            True/False

        Constraints:
        1. 0 <= abs(N_D - start_node) <= p
        2. abs(N_L-N_R) <= q
        3. N_L + N_D + N_R = sum(start_node)
        4. N_D >= 0
        '''
        (p,q) = self.graph_model
        
        cond1 = True
        cond4 = True 
        num_dots = len(u) - 2
        for i in range(1,num_dots+1):
            cond1 = cond1 and (np.abs(u[i] - self.start_node[i]) <= p)
            cond4 = cond4 and (u[i] >= 0)

        cond2 = (abs(u[0] - u[-1]) <= q)

        cond3 = (np.sum(u) == np.sum(np.array(self.start_node)))

        return (cond1 and cond2 and cond3 and cond4)

    def generate_neighbours(self,v):
        '''
        Input:
            v : node to find neighbours of
        Output:
            valid : valid set of neighbours v in the graph

            Takes in a num_dot + 2 charge state (N_L,vec N_D,N_R) model and generates neighbours in the validity of model=(p,q)
            Constraints:
            1. 0 <= abs(N_D - start_node) <= p
            2. abs(N_L-N_R) <= q
            3. N_L + N_D + N_R = sum(start_node)
        '''

        (p,q) = self.graph_model
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

        # note the python3 syntax
        valid = [*filter(self.check_validity,neigh)]    
        return valid

    def fermi(self,E,kT):
        '''
        Input:
            E : energy (eV)
            kT : temp in eV

        Output:
            fermi_function
        '''
        return 1.0/(1 + np.exp(E/kT))

    def find_weight(self,u,v):
        '''
        Input:
            u : start node
            v : end node

        Output:
            weight : weight of edge from u to v
        '''
       
        N_dot_1 = u[1:-1] 
        n1,mu1 = self.tf.tf_iterative_solver_fixed_N(self.prelim_mask,N_dot_1)
        E_1 = self.tf.calculate_thomas_fermi_energy(n1)

        N_dot_2 = v[1:-1] 
        n2,mu2 = self.tf.tf_iterative_solver_fixed_N(self.prelim_mask,N_dot_2)
        E_2 = self.tf.calculate_thomas_fermi_energy(n2)

        simple_prob = self.fermi(E_2 - E_1,self.tf.kT)
        
        #tunnel_prob = tunneling.calculate_tunnel_prob(u,v,physics,n1,mu1)
        tunnel_prob = 1

        #attempt_rate = tunneling.calculate_attempt_rate(u,v,physics,n1,mu1)
        #attempt_rate *= 1000
        attempt_rate = 1
        
        weight = attempt_rate*tunnel_prob*simple_prob
        return weight
        
    def add_battery_edges(self):
        (mu_l1,mu_l2) = self.tf.mu_l
        cond1 = (mu_l1 < mu_l2)
        for u in list(self.G.nodes()):
            for v in list(self.G.nodes()):
                # not a battery edge since number on dot changes
                if u[1:-1] != v[1:-1]:
                    pass
                # electron passes from left to right
                elif cond1:
                    if u[0] > v[0]:
                        self.G.add_edge(u,v,weight=self.tf.battery_weight)
                        nx.set_edge_attributes(self.G,'battery_edge',{(u,v) : True})
                        nx.set_node_attributes(self.G,'battery_node',{u : True})
                # electron passes from right to left
                else:
                    if u[0] < v[0]:
                        self.G.add_edge(u,v,weight=self.tf.battery_weight)
                        nx.set_edge_attributes(self.G,'battery_edge',{(u,v) : True})
                        nx.set_node_attributes(self.G,'battery_node',{u : True})

    def generate_graph(self):
        '''
        Input:
        Output:
            G : Markov graph of the charge states, weights assigned to edges using the energy method at zero bias, battery edges are added according to the battery weight paramter in physics input

        '''
        self.G = nx.DiGraph() 
        
        # queue used for BFS generation of the graph
        Q = queue.Queue()

        # get start_node and num_dot
        self.tf = thomas_fermi.ThomasFermi(self.physics)
        self.num_dot,self.prelim_mask = self.tf.find_n_dot_estimate() 

        mu_d = [self.tf.mu_l[0]]*self.num_dot
        n,N_d = self.tf.tf_iterative_solver_fixed_mu(self.prelim_mask,mu_d)
        N_est = [int(x) for x in N_d]
         
        # dots + leads
        # create the start node tuple
        N_est = [0] + N_est + [0]
        self.start_node = tuple(N_est)

        Q.put(self.start_node)
        while not Q.empty():
            v = Q.get()
            self.G.add_node(v)
            neigh = self.generate_neighbours(v)
            for n in neigh:
                # non-optimal: TODO: find a better strategy
                if n not in list(self.G.nodes()):
                    Q.put(n)
                    self.G.add_node(n)

                # Catch here : Put in the weight even if node exists, because weights might not be added
                # put in weight information
                # finally, Physics, Yay!
                self.G.add_edge(v,n,weight=self.find_weight(v,n))
                nx.set_edge_attributes(self.G,'battery_edge',{(v,n) : False})
                self.G.add_edge(n,v,weight=self.find_weight(n,v))
                nx.set_edge_attributes(self.G,'battery_edge',{(n,v) : False})
       
        self.add_battery_edges()
        self.get_battery_nodes()

    def get_battery_nodes(self):
        '''
        Input:
        Output:
            battery_ind : list of battery nodes
        '''
        # battery
        # TODO: Find a better way to find the indices for the battery edges
        battery_nodes = nx.get_node_attributes(self.G,'battery_node')
        nodes = list(self.G.nodes())
        battery_ind = []
        # find the keys of the battery nodes
        for key in battery_nodes:
            battery_ind += [nodes.index(key)]

        self.battery_ind = battery_ind
    
    def get_prob_dist(self,M):
        '''
        Input:
            M : matrix
        Output:
            dist : prob normalised nullspace vector of M
        '''
        nullspace = rank_nullspace.nullspace(M,rtol=1e-9)  
        if (nullspace.shape[1] > 0):
            #non-trivial nullspace exists for M
            # dist is prob distribution
            dist = nullspace[:,0]/nullspace[:,0].sum(axis=0)
        else:
            #nullspace is trivial, in this case there is no stable prob. distribution,
            #In case raised, try changing the rtol parameter
            raise ValueError('Nullspace of Markov matrix is trivial. No probability distribution exists')
        return dist
            
    def get_current(self):
        '''
        Input:
        Output:
            current : current 

        The basic idea is to create a Markov evolution matrix from the weights. The stable probability distribution is given as the nullspace of this matrix.

        The current is calculated by summing over the probabilities at the beginning of the battery edges.
        '''

        # Adjacency matrix, caution not the Markov matrix
        A = nx.to_numpy_matrix(self.G)
        # look at this carefully
        M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))

        dist = self.get_prob_dist(M)

        # calculate the current by summing over the probabities over the battery nodes 
        current = 0
        for b_ind in self.battery_ind:
            current += dist[b_ind,0]

        return current

    def get_max_prob_node(self):
        '''
        Output:
            Node with highest occupation probability
        '''
        # Adjacency matrix, caution not the Markov matrix
        A = nx.to_numpy_matrix(self.G)
        # look at this carefully
        M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))

        dist = self.get_prob_dist(M)
       
        max_prob_index = np.argmax(dist)
        nodes = list(self.G.nodes())
        return nodes[max_prob_index]
