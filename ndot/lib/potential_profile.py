# Module to create potential profiles
import numpy as np

def gauss(x,mean,sigma,peak):
    return peak*np.exp(-(x-mean)**2/(2*sigma**2))

def wire_profile(x,mean,impact,peak):
    return peak*np.log((1.0)/(impact)*np.sqrt((x-mean)**2 + impact**2))

def single_dot_V_x_gauss(x,d,b1,b2):
    '''
    Input:
    x : 1d linear grid
    d : dot potential paramters
    b1,b2 : barrier parameters:
    
    Output:
    V(x) : potential landscape 
    
    The dot potential is modelled as a parabolic well with depth V_d and 
    size dot_size. The two barriers are modelled as Gaussaians. The 
    parameters are passed as the following tuples.
    
     d = (dot_size,V_d)
    b1 = (V_b1,mu_b1,sigma_b1)
    b2 = (V_b2,mu_b2,sigma_b2)
    
    Note: It is responsiblity of the caller to ensure that the dot_size is
    within the range of x.
    '''
    
    (dot_size,V_d) = d
    (V_b1,mu_b1,sigma_b1) = b1
    (V_b2,mu_b2,sigma_b2) = b2
    
    V = np.zeros(len(x))

    # dot potential
    k = 2*np.abs(V_d)/(dot_size**2)
    # outside the dot, the parabolic potential should be zero, faster 
    # way to ensure that rather than iterating with an if condition
    x_dot = np.array(map(lambda x:x if np.abs(x) < dot_size else dot_size,x))
    V += 0.5*k*x_dot**2 + V_d
    
    # barriers
    V += gauss(x,mu_b1,sigma_b1,V_b1)
    V += gauss(x,mu_b2,sigma_b2,V_b2)
    
    return V

def single_dot_V_x_wire(x,d,b1,b2):
    '''
    Input:
    x : 1d linear grid
    d : dot potential paramters
    b1,b2 : barrier parameters:
    
    Output:
    V(x) : potential landscape 
   
    The dot potential is modelled as a potential form cylindrical metal gates. V ~ ln(r) potential.
    
     d = (V_d,mu_d,impact_dot)
    b1 = (V_b1,mu_b1,impact_b1)
    b2 = (V_b2,mu_b2,impact_b2)
   
    impact parameter is a measure of the dot size.
    Note: It is responsiblity of the caller to ensure that the dot_size is
    within the range of x.
    '''
    
    (V_d,mu_d,impact_dot) = d
    (V_b1,mu_b1,impact_b1) = b1
    (V_b2,mu_b2,impact_b2) = b2
    
    V = np.zeros(len(x))

    # dot potential
    V += wire_profile(x,mu_d,impact_dot,V_d) 
    # barriers
    V += wire_profile(x,mu_b1,impact_b1,V_b1)
    V += wire_profile(x,mu_b2,impact_b2,V_b2)
    
    return V
