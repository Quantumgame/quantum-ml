# Markov graph class to calculate transport properties.

import numpy as np
import queue
import networkx as nx
from scipy.special import expit
import scipy.integrate

import thomas_fermi
import rank_nullspace
import exceptions

class Markov():
    '''
    Class Markov is used for graph creation and calculation of currents.
    '''

    def __init__(self,graph_model,physics,tf_strategy='opt_iter'):
        self.graph_model = graph_model
        # note that physics is a dict of the physical arguments and is passed while creating
        # ThomasFermi class
        self.physics = physics
        self.tf = thomas_fermi.ThomasFermi(self.physics)
        self.start_node = None
        self.tf_strategy = tf_strategy
        self.tf_solutions = {}
        self.recalculate_graph = True

    def find_n_dot_estimate(self,fix_mask = False):
        '''
        Finds the estimated number of dots using V - mu_l criterion
        '''
        # this is necessary to init the prelim mask
        self.num_dot = self.tf.find_n_dot_estimate(fix_mask)

        return self.num_dot
    
    def find_start_node(self): 
        '''
        # get start_node and num_dot
        '''
        if self.num_dot != 0:
            mu_d = [self.tf.mu_l[0]]*self.num_dot
            n,N_d = self.tf.tf_solver_fixed_mu(mu_d)
            N_est = [int(np.rint(x)) for x in N_d]
            # dots + leads
            # create the start node tuple
            N_est = [0] + N_est + [0]
        else:
            # only the leads
            N_est = [0,1]
         
        new_start_node = tuple(N_est)
        if(new_start_node == self.start_node):
            self.recalculate_graph = False
        else:
            self.start_node = new_start_node
            self.recalculate_graph = True

        return self.start_node

    #def find_tf_solutions(self):
    #    '''
    #    This function uses the self.start_node and the graph_model to precalculate the TF solutions.
    #    The solutions are stored as a dict indexed by the N_dot values. Each value is itself a dict with n and mu stored.

    #    The data is stored in self.tf_solutions
    #    '''
    #    if self.num_dot != 0:
    #        self.tf_solutions = {}
    #        N_est = self.start_node[1:-1]
    #        p = self.graph_model[0]
    #         
        
    
    def check_validity(self, u):
        '''
        Input:
            u : node to check validity of

        Output:
            True/False

        0 : Whether the physics can support such a charge state
        False if InvalidChargeException is raised.
        Constraints:
        1. 0 <= abs(N_D - start_node) <= p
        2. abs(N_L-N_R) <= q
        3. N_L + N_D + N_R = sum(start_node)
        4. N_D >= 0
        '''
        (p,q) = self.graph_model

        cond0 = True
        N_d = u[1:-1]
        try:
            self.tf.tf_iterative_solver_fixed_N(N_d,strategy=self.tf_strategy)
        except exceptions.InvalidChargeState:
            cond0 = False
            
        cond1 = True
        cond4 = True 
        num_dots = len(u) - 2
        for i in range(1,num_dots+1):
            cond1 = cond1 and (np.abs(u[i] - self.start_node[i]) <= p)
            cond4 = cond4 and (u[i] >= 0)

        cond2 = (abs(u[0] - u[-1]) <= q)

        cond3 = (np.sum(u) == np.sum(np.array(self.start_node)))

        return (cond0 and cond1 and cond2 and cond3 and cond4)

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
        num_dots = len(v) - 2 
        
        # handling the single barrier case
        if num_dots == 0:
            # in this case, v = (0,1) or (1,0)
            # so the only other case is given by reverse of v
            neigh.append(v[::-1])
            return neigh
        
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
        return expit(-E/kT)


    def find_weight(self,u,v):
        '''
        Input:
            u : start node
            v : end node

        Output:
            weight : weight of edge from u to v
        '''
        if self.num_dot != 0: 
            N_dot_1 = u[1:-1] 
            if(N_dot_1 not in self.tf_solutions):
                n1,mu_d_1 = self.tf.tf_iterative_solver_fixed_N(N_dot_1,strategy=self.tf_strategy)
                E_1 = self.tf.calculate_thomas_fermi_energy(n1,mu_d_1)
                self.tf_solutions[N_dot_1] = {'n':n1,'mu':mu_d_1,'E':E_1}
            else:
                n1 = self.tf_solutions[N_dot_1]['n']
                mu_d_1 = self.tf_solutions[N_dot_1]['mu']
                E_1 = self.tf_solutions[N_dot_1]['E']
            
            N_dot_2 = v[1:-1] 
            if(N_dot_2 not in self.tf_solutions):
                n2,mu_d_2 = self.tf.tf_iterative_solver_fixed_N(N_dot_2,strategy=self.tf_strategy)
                E_2 = self.tf.calculate_thomas_fermi_energy(n2,mu_d_2)
                self.tf_solutions[N_dot_2] = {'n':n2,'mu':mu_d_2,'E':E_2}
            else:
                n2 = self.tf_solutions[N_dot_2]['n']
                mu_d_2 = self.tf_solutions[N_dot_2]['mu']
                E_2 = self.tf_solutions[N_dot_2]['E']
            # change in number of electrons on the lead
            diff_lead = v[0] - u[0] + v[-1] - u[-1]
            simple_prob = self.fermi(E_2 - E_1 + diff_lead*self.tf.mu_l[0],self.tf.kT)
        else:
            simple_prob = 1.0
       
        tunnel_prob = self.calculate_tunnel_prob(u,v)
        #tunnel_prob = 1

        #attempt_rate = tunneling.calculate_attempt_rate(u,v,self.tf,self.tf_strategy)
        #attempt_rate *= 100
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

    def recalculate_weights(self):
        '''
        This function is called when the graph is unchanged and only the weights need to be recalculated.
        '''
        self.tf_solutions = {}
        for key in set(x[1:-1] for x in self.G.nodes()):
            n,mu = self.tf.tf_iterative_solver_fixed_N(key,strategy=self.tf_strategy)
            E = self.tf.calculate_thomas_fermi_energy(n,mu)
            self.tf_solutions[key] = {'n':n,'mu':mu,'E':E}

        for edge in self.G.edges():        
            # recalculate only the non-battery edges
            if (not nx.get_edge_attributes(self.G,"battery_edge")[edge]):
                u = edge[0]
                v = edge[1]
                self.G.add_edge(u,v,weight=self.find_weight(u,v))
            
        return 

    def generate_graph(self):
        '''
        Input:
        Output:
            G : Markov graph of the charge states, weights assigned to edges using the energy method at zero bias, battery edges are added according to the battery weight paramter in physics input

        '''
        if (not self.recalculate_graph):
            try:
                self.recalculate_weights()
                # get the stable prob distribution
                self.get_prob_dist()
                return
            except exceptions.InvalidChargeState:
                # this is necesary to init the prelim_mask
                self.num_dot = self.tf.find_n_dot_estimate(False)
                self.recalculate_graph = True 

        # queue used for BFS generation of the graph
        Q = queue.Queue()
        self.tf_solutions = {}
        self.G = nx.DiGraph() 

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

        # get the stable prob distribution
        self.get_prob_dist()

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
    
    def get_prob_dist(self):
        '''
        Output:
            dist : prob normalised nullspace vector of M
        '''
        # Adjacency matrix, caution not the Markov matrix
        A = nx.to_numpy_matrix(self.G)
        # look at this carefully
        M =  A.T - np.diag(np.array(A.sum(axis=1)).reshape((A.shape[0])))
    
        try:
            nullspace = rank_nullspace.nullspace(M,rtol=1e-9)  
            if (nullspace.shape[1] > 0):
                #non-trivial nullspace exists for M
                # dist is prob distribution
                self.dist = nullspace[:,0]/nullspace[:,0].sum(axis=0)
            else:
                #nullspace is trivial, in this case there is no stable prob. distribution,
                #In case raised, try changing the rtol parameter
                raise ValueError('Nullspace of Markov matrix is trivial. No probability distribution exists')
        except (ValueError,np.linalg.LinAlgError) as e:
            raise exceptions.InvalidChargeState
        return self.dist

    def get_current(self):
        '''
        Input:
        Output:
            current : current 

        The basic idea is to create a Markov evolution matrix from the weights. The stable probability distribution is given as the nullspace of this matrix.

        The current is calculated by summing over the probabilities at the beginning of the battery edges.
        '''


        # calculate the current by summing over the probabities over the battery nodes 
        current = 0
        for b_ind in self.battery_ind:
            current += self.dist[b_ind,0]

        return current

    def get_charge_state(self):
        '''
        Output:
            Node with highest occupation probability
        '''
        max_prob_index = np.argmax(self.dist)
        nodes = list(self.G.nodes())
        # remove the leads
        return nodes[max_prob_index][1:-1]

    def get_output(self):
        '''
        Output : returns a dict of current and max_prob_state
        '''
        output = {}
        output['current'] = self.get_current()
        output['charge_state'] = self.get_charge_state()
        output['prob_dist'] = self.dist
        output['num_dot'] = self.num_dot

        return output

    def calculate_tunnel_prob(self,v,u):
        '''
        Input:
            v : start node
            u : end node

        Output:
            tunnel_prob : (under WKB approximation)
        '''
        # Uses WKB method for the tunnel probability calculation 

        # find where the transition is occuring
        u = np.array(u)
        v = np.array(v)
        diff = u - v

        index_electron_to = np.argwhere(diff == 1.0)[0,0]
        index_electron_from = np.argwhere(diff == -1.0)[0,0]

        # clever way to find the barrier index
        bar_index = np.floor(0.5*(index_electron_to + index_electron_from))
        bar_key = 'b' + str(int(bar_index))
       
        N_dot = tuple(v[1:-1]) 
        if (N_dot not in self.tf_solutions):
            # solve tf for start node 
            n,mu = self.tf.tf_iterative_solver_fixed_N(N_dot,strategy=self.tf_strategy)
            E = self.tf.calculate_thomas_fermi_energy(n,mu)
            self.tf_solutions[N_dot] = {'n':n,'mu':mu,'E':E}
        else:
            n = self.tf_solutions[N_dot]['n']
            mu = self.tf_solutions[N_dot]['mu']
            E = self.tf_solutions[N_dot]['E']
            
        # chemical_potential = energy of the electron 
        # add in the lead potentials to mu to simplify notation
        mu = np.concatenate((np.array([self.tf.mu_l[0]]),mu,np.array([self.tf.mu_l[1]]))) 
        mu_e = mu[index_electron_from]

        # integral
        bar_begin = self.tf.mask.mask_info[bar_key][0]
        bar_end = self.tf.mask.mask_info[bar_key][1]

        # in the barrier region, since n = 0, the effective potential is almost just V
        V_eff = self.tf.V + np.dot(self.tf.K,n)
        
        factor = scipy.integrate.simps(np.sqrt(np.abs(V_eff[bar_begin:(bar_end+1)] - mu_e)),self.tf.x[bar_begin:(bar_end+1)])

        # calcualte the scale based on physics in self.tf
        scale = self.tf.WKB_scale   
        
        tunnel_prob = np.exp(-scale*factor)
        return tunnel_prob
