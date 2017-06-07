# Module for Thomas-Fermi calculation of electron density along a 1D potential

class Physics():
    '''
    Physics class is the base class for Thomas-Fermi calculations. Physical quantities stored here are:
    Scales:
    E_scale  : energy scale of the problem, all energies are measured in this scale
               default (eV)
    dx_scale : scale for nearest neighbour distance in x-gird (m)

    Physical Parameters:
    kT       : temperature in eV
    x        : x-grid
    V(x)     : array same size as len(x), storing the potential V(x), the nearest neighbour separation is given by dx_scale
    K(x,x')  : 2D matrix for storing the interaction energy between points x and x',
    '''

    def __init__(self,physics):
        '''
        Input:
            physics = (E_scale,dx_scale,kT,x,V)
        Output
            None
        '''
        (E_scale,dx_scale,kT,x,V) = physics
        self.E_scale = E_scale
        self.dx_scale = dx_scale
        
        self.kT = kT
        self.x = x
        self.V = V
   
   def calculate_K_matrix(self,E_0,sigma=(self.x[1]-self.x[0])):
        '''
        Input: 
            E_0   : energy scale for the K matrix for the onsite interaction, measured as a multiple of E_scale
            sigma : softening paramter to prevent blow up at the same point
        Output:
            K     : matrix of size x.size times x.size

        K(x1,x2) = E_scale / sqrt((x1 - x2)^2 + sigma^2)
        '''
        self.K = E_0/np.sqrt((x[:, np.newaxis] - x)**2 + sigma**2)
    
class Thomas_Fermi(Physics):
    '''
    Subclass of the Physics class. Used to calculate electron density n(x) for a fixed number of electrons in the islands.
    The mask (labelling of the islands is assumed as an input as is not calculated here)
    '''

    def __init__(self,physics):
        Physics.__init__(self,physics)

