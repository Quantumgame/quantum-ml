# Function to create a potential profile 
import numpy as np

def wire_profile(x,param):
    '''
    V(x-mean) = peak * log(sqrt(h^2 + x^2)/rho)
    '''
    (peak,mean,h,rho) = param
    return peak*np.log((1.0/(rho))*np.sqrt((x-mean)**2 + h**2))

def V_x_wire(x,list_b):
    '''
    Input:
    x : 1d linear grid
    list_b : list of gate parameters as (V,mu,h,rho) where V(x) = V*ln(sqrt(h^2 + (x-mu)^2)/rho)
    
    Output:
    V(x) : potential profile
    '''
    
    wire_profiles = [wire_profile(x,p) for p in list_b]
    V = np.sum(wire_profiles,axis=0)
        
    return V
