# Module for Thomas-Fermi calculation of electron density along a 1D potential

import numpy as np
import mask
from physics import Physics
import scipy.optimize
    
class ThomasFermi(Physics):
    '''
    Subclass of the Physics class. Used for all Thomas-Fermi calculations 
    '''

    def __init__(self,physics):
        Physics.__init__(self,physics)
    
        # There are two broad TF solvers. One of them is a fixed mu solver, which given the dot potentials, finds the expected charge density.
        # The other is a fixed N solver, which given number of electron on each dot, finds the expected charge density.
        # Both solvers take in an initial mask and require the number of dots to be known in advance.
    
    def find_n_dot_estimate(self,fix_mask):
        '''
        Estimate the number of dots and a mask assuming the chemical potential is mu_l everywhere

        The mask is defined as a list of size len(x) where x is the x-grid. Each element is labelled as 'l0','l1' corresponding to the leads, 'di' corresponding 
        to the ith dot and 'bi' corresponding to the ith barrier.

        Counting begins from 0.
        ''' 

        mu = self.mu_l[0] 
        mu_x = np.repeat(mu,len(self.V))
       
        # Use mu - V ~ n to make a preliminary classfication 
        if(not fix_mask):
            prelim_mask = mask.Mask(mu_x - self.V) 
            self.mask = prelim_mask
            return prelim_mask.mask_info['num_dot'] 
        else:
            # return value from the old calculation
            return self.mask.mask_info['num_dot']  
    
    def calculate_mu_x_from_mask(self,mu_d):
        '''
        Takes in a mask object and uses it to form an array of mu_x. The vector is zero in barrier regions by design since the chemical potential is an unknown in the barrier region.

        The lead potentials are taken from the underlying physics object and the mu_d is taken as an argument to this function.
        '''
        
        mu_x = np.zeros(len(self.mask.mask))
        for key,val in self.mask.mask_info.items():
            if (key == 'l0'):
                mu_x[val[0]:val[1] + 1] = self.mu_l[0] 
            elif (key == 'l1'):
                mu_x[val[0]:val[1] + 1] = self.mu_l[1] 
            elif (key[0] == 'd'):
                mu_x[val[0]:val[1] + 1] = mu_d[int(key[1:])]
            else:
                # region is a barrier
                # leave untounced i.e = 0 since the potential over a barrier is an unknown
                pass
        return mu_x 
    
    def create_fixed_mu_A(self,mu_d):
        '''
        Creates the LHS matrix for fixed mu Thomas Fermi calculation.
        A z = b
        where z = (n N_d mu_bar)
        A =
        (K                 0              mu_bar_constraint
         sum_n             N_d_constraint 0
         mu_bar_constraint 0              0)
        '''
        N_grid = len(self.mask.mask)
        num_dot = self.mask.mask_info['num_dot']
        num_barrier_points = self.mask.mask.count('b') 
        N_A = N_grid + num_dot + num_barrier_points
        A = np.zeros((N_A,N_A))
        
        A[:N_grid,:N_grid] = self.K

        barrier_col_index = N_grid + num_dot               
        for key,val in self.mask.mask_info.items():
            # implement the n = 0 barrier constraint thorugh the Lagrange multipliers mu_bar
            if key[0] == 'b':
                barrier_size = val[1] - val[0] + 1 
                A[val[0]:(val[0] + barrier_size),barrier_col_index:(barrier_col_index + barrier_size) ] = -1.0*np.identity(barrier_size)
                A[barrier_col_index:(barrier_col_index + barrier_size),val[0]:(val[0] + barrier_size) ] = np.identity(barrier_size)
                barrier_col_index += barrier_size
            
            # implement the sum_n over dot - N_d constraint
            elif key[0] == 'd':
                dot_col_index = N_grid + int(key[1:])
                dot_size = val[1] - val[0] + 1
                A[dot_col_index,val[0]:(val[0] + dot_size)] = 1        
                A[dot_col_index,dot_col_index] = -1
        return A

    def create_fixed_mu_b(self,mu_d):
        '''
        Creates the RHS column vector for fixed mu Thomas Fermi Calculation.
        A z = b
        b =
        ( mu_x - V
          0
          0)
        '''

        N_grid = len(self.mask.mask)
        num_dot = self.mask.mask_info['num_dot']
        num_barrier_points = self.mask.mask.count('b') 
        N_b = N_grid + num_dot + num_barrier_points
        b = np.zeros(N_b)
        
        mu_x = self.calculate_mu_x_from_mask(mu_d)
        b[:N_grid] = mu_x - self.V
        
        return b 

    def create_fixed_N_A(self,N_d):
        '''
        Creates the LHS matrix for fixed N Thomas Fermi calculation.
        A z = b
        A =
        (K                 mu_d_constraint mu_bar_constraint
         sum_n             0               0
         mu_bar_constraint 0               0)
        '''
        N_grid = len(self.mask.mask)
        num_dot = self.mask.mask_info['num_dot']
        num_barrier_points = self.mask.mask.count('b') 
        N_A = N_grid + num_dot + num_barrier_points
        A = np.zeros((N_A,N_A))
        
        
        A[:N_grid,:N_grid] = self.K

        barrier_col_index = N_grid + num_dot                
        for key,val in self.mask.mask_info.items():
            # implement the n = 0 barrier constraint thorugh the Lagrange multipliers mu_bar
            if key[0] == 'b':
                barrier_size = val[1] - val[0] + 1 
                A[val[0]:val[0] + barrier_size,barrier_col_index:barrier_col_index + barrier_size ] = -1.0*np.identity(barrier_size)
                A[barrier_col_index:barrier_col_index + barrier_size,val[0]:val[0] + barrier_size ] = np.identity(barrier_size)
                barrier_col_index += barrier_size
            
            # implement the sum_n over dot - N_d constraint
            elif key[0] == 'd':
                dot_col_index = N_grid + int(key[1:])
                dot_size = val[1] - val[0] + 1
                A[dot_col_index,val[0]:val[0] + dot_size] = 1        
                A[val[0]:val[0] + dot_size,dot_col_index] = -1
        return A

    def create_fixed_N_b(self,N_d):
        '''
        Creates the RHS matrix for fixed N Thomas Fermi calculation
        A z = b

        b = 
        (-V+mu_x
          N_d
          0)
        where mu_x is calculated with dot_potentials = 0
        '''
        N_grid = len(self.mask.mask)
        num_dot = self.mask.mask_info['num_dot']
        num_barrier_points = self.mask.mask.count('b') 
        N_b = N_grid + num_dot + num_barrier_points
        b = np.zeros(N_b)
       
        # mu_l - V, this is an easy way to use mu_x to calculate 
        # set the mu_d to zero
        mu_d = [0.0]*num_dot
        mu_x = self.calculate_mu_x_from_mask(mu_d)
        b[:N_grid] = mu_x - self.V
        b[N_grid:N_grid + num_dot] = N_d
        
        return b
        
    def tf_solver_fixed_mu(self,mu_d):
        '''
        Assumes a mask object and a array of mu_d dot chemical potentials
        The lead potentials are taken from self.mu_l and not explicitly given.

        This solves the equation V - mu_mask + K n = 0. The constraint of n = 0 in the barriers is implemented according to the mask.
        Return a n(x) : electron density, N_d : estimate of electrons on each dot
    
        N_dot elements are not integers!
        '''
        if(len(mu_d) != self.mask.mask_info['num_dot']):
            # number of dots in mu_d does not match in the mask
            # This means that mu_d provided is wrong or there is a problem with the mask.
            raise ValueError('Calculation of mu_x failed. Check number of dots in mu_d and the mask') 
        
        # Formulate the problem as A z = b
        # z = (n N_d mu_bar)       
        
        A = self.create_fixed_mu_A(mu_d) 
        b = self.create_fixed_mu_b(mu_d)

        z = np.linalg.solve(A,b)

        num_dot = len(mu_d)
        N_grid = len(self.mask.mask)
        
        return z[:N_grid],z[N_grid:(N_grid + num_dot)] 

    def tf_solver_fixed_N(self,N_d):
        '''
        Uses the self.mask object and number of electrons in each dot.
        The lead potentials are taken from self.mu_l and not explicitly given.
        
        It solves the equation V - mu + K n = 0 iteratively for the mask and n(x). The constraint of n = 0 in the barriers is implemented according to the mask.

        Returns a n(x) : electron density and mu_d : dot potentials 
        ''' 
        if(len(N_d) != self.mask.mask_info['num_dot']):
            # number of dots in N_d does not match in the mask
            # This means that mu_d provided is wrong or there is a problem with the mask.
            raise ValueError('Calculation of mu_x failed. Check number of dots in N_d and the mask') 
   
        # Formulate the problem as A z = b
        # z = (n mu_d mu_bar) 

        A = self.create_fixed_N_A(N_d) 
        b = self.create_fixed_N_b(N_d)

        z = np.linalg.solve(A,b)

        num_dot = len(N_d)
        N_grid = len(self.mask.mask)
        return z[:N_grid],z[N_grid:(N_grid + num_dot)] 

    def tf_solver_fixed_N_opt(self,N_d):
        '''
        Takes in a mask object and number of electrons on each dot.
        The lead potentials are taken from self.mu_l and are not explicitly given.

        Optional Solver based on scipy.optimize.minimize. Has the added advantage that n >= 0 can be implemented as a constraint.
        '''

        def energy(n,V,mu_x,K):
            return np.sum((V - mu_x)*n) + 0.5*np.sum(n*np.dot(K,n))
      
        def dot_constraint_function(n,mask,N_d):
            err = 0.0
            for key,val in self.mask.mask_info.items():
                if (key[0] == 'd'):
                    err += np.abs(np.sum(n[val[0]:(val[1]+1)]) - N_d[int(key[1:])])
            return err
                         
        # easy way to create the required mu_x profile using mu_d = [0] 
        mu_x = self.calculate_mu_x_from_mask([0]*self.mask.mask_info['num_dot'])

        cons = {'type':'eq','fun':(lambda x: dot_constraint_function(x,self.mask,N_d))}

        # initial guess for the solution
        n_0 = np.zeros(len(self.V))
        # bounds parameter defines that n >= 0
        root = scipy.optimize.minimize(lambda x : energy(x,self.V,mu_x,self.K),n_0,bounds=[(0,None)]*len(n_0),constraints=cons)

        # find the dot_potentials using V + Kn = mu
        # The dot potential is found from mu calculated at the center of each dot
        mu_d = np.zeros(self.mask.mask_info['num_dot'])
        mu_vec = self.V + np.dot(self.K,root.x)
        for key,val in self.mask.mask_info.items():
            if key[0] == 'd':
                dot_index = int(key[1:])
                dot_center_index = int(0.5*(val[0] + val[1]))
                mu_d[dot_index] = mu_vec[dot_center_index] 
        return root.x,mu_d

    # Iterative fixed mu solver is not very useful since if the mu is fixed, the turning points do not change. Hence there is nothing to iterative over.
    # If you are thinking that the potential could be replaced by the effective potential V_eff = V + Kn in each iteration, by the solver V + Kn = mu, hence there are no changes in thr mask.
    def tf_iterative_solver_fixed_mu(self,mu_d,N_lim = 10):
        '''
        Solve the TF problem iteratively until the mask converges.

        In each iteration, the potential is updated to the effective potential 
        V_eff = V + K.n

        The iteration ends when the mask converges to a fixed value or N_lim iterations are reached.
        '''
        old_mask = self.mask.mask
        i = 0
        while(i < N_lim):
            n,N_d = self.tf_solver_fixed_mu(mu_d) 
           
            self.mask.calculate_new_mask_turning_points(self.V,self.mu_l,mu_d)
            self.mask.calculate_mask_info_from_mask()
             
            if(old_mask == self.mask.mask):
                break  
            old_mask = self.mask.mask
            i += 1
        if(i == N_lim):
            raise Exception("Mask failed to converge in Thomas Fermi iterative fixed mu solver.")  
        return n,N_d

    def tf_iterative_solver_fixed_N(self,N_d,N_lim = 10,strategy='opt_iter'):
        '''
        Solve the TF problem iteratively until the mask converges.

        In each iteration, the potential is updated to the effective potential 
        V_eff = V + K.n

        The iteration ends when the mask converges to a fixed value or N_lim iterations are reached.
        '''
        # simple strategy uses the fixed N solver only once
        # does not iterate to find a mask beyond the prelim_mask
        if strategy == 'simple':
           n,mu_d = self.tf_solver_fixed_N(N_d) 
           return n,mu_d

        # simple_iter strategy uses the fixed N solver iteratively
        # in each iteration, depending on the new chemical potential(s) found, the turning points
        # and the mask is updated until the mask converges to a fixed value
        # N_d is recalcualted from the new mask to handle cases where new smaller dots are formed
        elif strategy == 'simple_iter':
            old_mask = self.mask.mask
            i = 0
            while(i < N_lim):
                n,mu_d = self.tf_solver_fixed_N(N_d) 
              
                self.mask.calculate_new_mask_turning_points(self.V,self.mu_l,mu_d)
                self.mask.calculate_mask_info_from_mask()
               
                if(old_mask == self.mask.mask):
                    break  
                old_mask = self.mask.mask
                N_d = self.calculate_N_d_from_n(n)
                i += 1
            if(i == N_lim):
                raise Exception("Mask failed to converge in Thomas Fermi simple iterative fixed N solver.")  
            return n,mu_d
        
        elif strategy == 'opt': 
            n,mu_d = self.tf_solver_fixed_N_opt(N_d)
            return n,mu_d

        elif strategy == 'opt_iter': 
            old_mask = self.mask.mask
            i = 0
            while(i < N_lim):
                n,mu_d = self.tf_solver_fixed_N_opt(N_d) 
               
                self.mask.calculate_new_mask_turning_points(self.V,self.mu_l,mu_d)
                self.mask.calculate_mask_info_from_mask()
             
                if(old_mask == self.mask.mask):
                    break  
                old_mask = self.mask.mask
                N_d = self.calculate_N_d_from_n(n)
                i += 1
            if(i == N_lim):
                raise Exception("Mask failed to converge in Thomas Fermi iterative fixed N solver.")  
            return n,mu_d

        else:
            raise Exception("Unknown strategy in iterative fixed N solver.")

    def calculate_thomas_fermi_energy(self,n,mu_d):
        '''
        Input: 
            n : electron density
            mu_d : dot potentials
        Output:
            E : Thomas-Fermi energy

        E = V n + 1/2 n K n
        '''
        N_d = self.calculate_N_d_from_n(n)
        # mu_x with only the leads
        mu_d_tmp = [0.0]*len(mu_d)
        mu_x = self.calculate_mu_x_from_mask(mu_d_tmp)
        E = np.sum((self.V - mu_x)*n) + 0.5 * np.sum(n*np.dot(self.K,n.T)) -0.0* np.sum(mu_d*N_d)
        return E

    def calculate_N_d_from_n(self,n):
        '''
        Input:
            n : electron density
        Output:
            N_d : vector of size mask.mask_info['num_dot'] with number of electrons on each dot according to self.mask and n
        '''
        N_d = np.zeros(self.mask.mask_info['num_dot'])
        for key,val in self.mask.mask_info.items():
            if key[0] == 'd':
                dot_index = int(key[1:])
                N_d[dot_index] = np.sum(n[val[0]:(val[1] + 1)])
        
        return N_d
