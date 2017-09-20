import numpy as np

class ThomasFermi():
    '''
    Thomas-Fermi routines based on the polylog function approach.
    '''
    def __init__(self,physics):
        '''
        physics is a dict with the revelant physics
        
        x : linspace
        V_x : potential profile
        K_0 : strength of the Coulomb interaction
        mu : chemical potential (assumed to be equal for both leads)
        '''
       
        self.physics = physics
        self.K_mat = self.calc_K_mat()

    def calc_K_mat(self):
        x = self.physics['x']
        K_0 = self.physics['K_0']
        
        dx = np.sqrt((x - x[:,np.newaxis])**2 + 1)
        return K_0/dx

    def calc_n(self):
        V = self.physics['V_x']
        K_mat = self.K_mat
        
        def f(x):
            np_polylog = np.frompyfunc(mpmath.polylog, 2, 1)
            output = -1e-2*np_polylog(1,-np.exp(50*x))
            # cast from mpc array to numpy array
            return np.array(list(map(lambda x : complex(x),output)))
        
        n = np.zeros(len(V))
        n_prev = np.zeros(len(V))
        phi = np.zeros(len(V))
        for i in range(500):
            # turn on the Coulomb over 10 steps
            if i < 10:
                phi = (i/10) * np.dot(K_mat,n)
            n_prev = n
            n = f(mu - V - phi)

            if (i > 10) and (np.linalg.norm(n - n_prev)**2 < (1e-12) * np.linalg.norm(n) * np.linalg.norm(n_prev)):
                #print("Done in",i)
                break;
        self.n = np.real(n)
        return self.n

    def calc_islands(self):
      
        n = self.n
        # adaptive eps makes the calculation more robust 
        n_eps = 3e-1*np.max(n)
        
        n_chop = np.array([x if x > n_eps else 0.0 for x in n])
        islands = {}
        start = False
        islands_index = 0
        left = 0
        for i in range(len(n_chop)):
            if(n_chop[i] > 0.0 and not start):
                start = True
                left = i
            if(n_chop[i] == 0.0 and start):
                start = False
                islands[islands_index] = [left,i - 1]
                islands_index += 1
            if((i == (len(n_chop) - 1)) and start):
                start = False
                islands[islands_index] = [left,i]
                islands_index += 1
        
        # ignore the leads
        try:
            lead1 = 0
            lead2 = len(islands) - 1
            islands.pop(lead1)
            islands.pop(lead2)
            self.islands = islands 
        except KeyError as e:
            self.islands = islands
            #raise e
        
        self.islands = islands
        return self.islands

    def calc_charges(self):
        n = self.n
        islands = self.islands
        
        charges = []
        for key,item in islands.items():
            charges.append(np.sum(n[item[0]:item[1]]))

        return np.array(charges)

    def calc_cap_model(self):
        islands = self.islands
        n = self.n

        # list of charge densities for islands
        n_list = []
        for key,item in islands.items():
            n_island = np.zeros(len(n))
            n_island[item[0]:item[1]] = n[item[0]:item[1]] 
            n_list.append(n_island)
            
        V = self.physics['V_x']
        mu = self.physics['mu']
        def cap_func(i,j):
            energy = 0.0
            if i == j:
                energy += 5*np.sum(n_list[i]*n_list[i]) + np.sum((V - mu)*n_list[i])

            energy += 2*np.dot(np.dot(n_list[i].T,self.K_mat),n_list[j])
            return energy

        # Seriously? I did not even normalise by Z_i^2 and Z_i*Z_j
        energy_matrix = np.array([cap_func(i,j) for i in range(len(n_list)) for j in range(len(n_list))])\
        .reshape((len(n_list),len(n_list)))

        # A problem here is that Z could be zero
        Z = self.calc_charges() + 1e-2
        # newaxis trick is used to calculate the outer product
        inverse_cap_matrix = 2*energy_matrix/np.dot(Z[:,np.newaxis],Z[np.newaxis])
        
        cap_model = (Z,inverse_cap_matrix)
        self.cap_model = cap_model
        
        return self.cap_model 

    def calc_cap_energy(self,N_vec):
        N_vec = np.array(N_vec)
        cap_model = self.cap_model
        return 0.5*np.dot(np.dot((N_vec-cap_model[0]),cap_model[1]),(N_vec-cap_model[0]).T)

    def calc_stable_charge_config(self):
        '''
        Full routine
        
        '''
        self.calc_n()
        self.calc_islands()
        self.calc_cap_model()
        
        (Z,inverse_cap_matrix) = self.cap_model 

        N_int = [int(x) for x in Z]
        dN_list = [range(max(0,x-2),x+2,1) for x in N_int] 
        import itertools
        N_list = list(itertools.product(*dN_list))

        energy_table = [self.calc_cap_energy(np.array(x)) for x in N_list]
        min_energy = min(energy_table)
        charge_configuration = N_list[energy_table.index(min_energy)]
        self.charge_configuration = np.array(charge_configuration)

        return self.charge_configuration

    def calc_state_current(self):
        '''
        Current and Charge calculation using a Master equation approach
        '''
        self.calc_n()
        self.calc_islands()
        self.calc_cap_model()
        
        (Z,inverse_cap_matrix) = self.cap_model 

        N_int = [int(x) for x in Z]
        dN_list = [range(max(x-1,0),x+1,1) for x in N_int] 
        import itertools
        states = list(itertools.product(*dN_list))
       
        def fermi(E,kT):
            from scipy.special import expit
            return expit(-E/kT)
        
        def calc_weight(a,b,kT):
            N_dot = len(a)
            if list(abs(np.array(a) - np.array(b))) in [list(x) for x in list(np.eye(N_dot,dtype=np.int))]:
                U_a = self.calc_cap_energy(np.array(a))
                U_b = self.calc_cap_energy(np.array(b))
                # notice the order, calc_weight calculates the weight to go from a to b
                return fermi(U_b - U_a,kT)
            else:
                return 0.0
            
        # A : adjacency matrix between the possible states, two states are connected only by a single electron tunneling event
        kT = self.physics['kT']
        A = np.array([calc_weight(a,b,kT) for a in states for b in states])\
            .reshape((len(states),len(states)))

        M = A.T - np.diag(np.sum(A,axis=1))

        # append the normalisation condition, and drop ones of the rows of M
        M_solver = np.append(M[:-1,:],[np.ones(M.shape[0])]).reshape(M.shape)

        # RHS in the master equation solution, the last element is the prob. normalisation condition
        b = np.zeros(M.shape[0])
        b[-1] = 1
        P = np.linalg.solve(M_solver,b)
        #state = states[np.argmax(P)]

        state = np.sum(np.array([x*np.array(y) for (x,y) in zip(P,states)]),axis=0)

        # poor's man current model
        # smart trick, current is finite over kT range
        if np.any([abs(x-0.5) < 25*kT for x in P]):
            current = 1
        else:
            current = 0
        return state,current