# Module for storing the basic physics parameters. Physics class is defined here.

import numpy as np
import scipy.constants

class Physics():
    '''
    Physics class is the base class for Thomas-Fermi calculations. Physical quantities stored here are:
    Scales:
    E_scale  : energy scale of the problem, all energies are measured in this scale (eV)
    dx_scale : scale for nearest neighbour distance in x-gird (nm)

    Physical Parameters:
    kT       : temperature in eV
    x        : x-grid
    list_b   : list of the gate params used to generate the potential profile
    V(x)     : array same size as len(x), storing the potential V(x), the nearest neighbour separation is given by dx_scale
    K_onsite : onsite interaction strength, used in calculating the K matrix
    sigma    : softening paramter of the matrix, used in calculating the K matrix
    x_0      : screening length in calculation of the K matrix
    K(x,x')  : 2D matrix for storing the interaction energy between points x and x',
    mu_l     : (mu_left,mu_right) 2-tuple for storing the lead potentials
    battery_weight : rate for transport throught a battery, set to a high number > 1000
    short_circuit_current : The current to be returned when there is no barrier in the device
    '''

    def __init__(self,physics):
        '''
        Input:
            physics is a dict with the keys
            (E_scale,dx_scale,kT,x,V,K_onsite,sigma,x_0,mu_l,battery_weight,short_circuit_current)
        Output
            None
        '''
        self.E_scale = physics['E_scale']
        self.dx_scale = physics['dx_scale']
        # set the WKB scale
        # \sqrt(2 m_e)/h_bar * sqrt(E_scale) * dx_scale
        
        self.WKB_scale = np.sqrt(2*scipy.constants.m_e*scipy.constants.e) * (1.0e-9) / scipy.constants.hbar
        self.attempt_rate_scale = np.sqrt(2*scipy.constants.e/scipy.constants.m_e) * 1e9
        
        self.kT = physics['kT']
        self.x = physics['x']
        self.list_b = physics['list_b']
        self.V = physics['V']
        self.K_onsite = physics['K_onsite']
        self.sigma = physics['sigma']
        self.x_0 = physics['x_0']
       
        # disabled for now. Otherwise battery nodes not identified. 
        #if mu_l[0] != mu_l[1]:
        #    raise ValueError("Finite bias calculation. Feature not yet available!") 
        
        self.mu_l = physics['mu_l']
        self.battery_weight = physics['battery_weight']
        self.short_circuit_current = physics['short_circuit_current']
        
        # The K-matrix is calculated only once. 
        self.calculate_K_matrix(self.x,self.K_onsite,self.sigma,self.x_0)

    def calculate_K_matrix(self,x,K_onsite,sigma,x_0):
        '''
        Input: 
            x : x-grid
            K_onsite : energy scale for the K matrix for the onsite interaction, measured as a multiple of E_scale
            sigma    : softening paramter to prevent blow up at the same point
            x_0      : screening length
        Output:
            None

        K(x1,x2) = K_onsite / sqrt((x1 - x2)^2 + sigma^2) * np.exp(-abs(x1-x2)/x_0)
        '''
        dx = np.abs(x[:,np.newaxis] - x)
        self.K = (K_onsite * np.exp(-1.0*dx/x_0))/np.sqrt(dx**2 + sigma**2)
