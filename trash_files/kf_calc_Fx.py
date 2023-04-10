########################################################################
## Calculates the Jacobian of the system dynamics equation 
########################################################################
import numpy as np


def kf_calc_Fx(t, x, u):
    """
    Calculates the Jacobian of the system dynamics equation,
    n is number of states.
    
    Parameters
    ----------
    t : float
        
    x : numpy.ndarray (n,1)
        state vector

    u : numpy.ndarray
        input vector

    Returns
    -------
    DFx : numpy.ndarray (n,n)
        Jacobian of the system dynamics equation
        
    """
    n = x.size
    
    # calculate Jacobian matrix of system dynamics
    DFx = np.zeros([4, 4])
    
    return DFx
        