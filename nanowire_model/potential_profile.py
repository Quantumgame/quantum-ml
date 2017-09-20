# Function to create a potential profile 
import numpy as np

def wire1_profile(x,param):
    '''
    V(x-mean) = peak * log(sqrt(h^2 + x^2)/rho)
    '''
    (peak,mean,h,rho) = param
    dx = np.abs(x-mean)
    return peak*np.log((1.0/(rho))*np.sqrt(dx**2 + h**2))*np.exp(-dx/(0.1*h))

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

def wire_profile(x,param):
    '''
    V(x-mean) = peak * log(sqrt(h^2 + x^2)/rho)
    '''
    (peak,mean,h,rho) = param
    dx = np.abs(x-mean)
    screen = 0.1
    return (peak/np.log(h/rho))*np.log(np.sqrt(dx**2 + h**2)/rho)*np.exp(-dx/screen)

