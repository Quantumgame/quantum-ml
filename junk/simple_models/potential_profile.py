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
    
    V = np.zeros(len(x))
    for i in range(len(list_b)): 
        V += wire_profile(x,list_b[i])
        
    return V
