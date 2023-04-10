########################################################################
## Calculates the system dynamics equation f(x,u,t)
########################################################################
import numpy as np
import matplotlib.pyplot as plt
def kf_calc_f(t, x, u):
    """
    Calculates the system dynamics equation f(x,u,t)
    
    Parameters
    ----------
    t : float
    
    x : numpy.ndarray (n,1)
        state vector
        
    u : numpy.ndarray
        input vector
        
    Returns
    -------
    xdot : numpy.ndarray (n,1)
        time derivative of the state vector, system dynamics
    """
    
    n       = x.size
    xdot    = np.zeros([n,1])
    # system dynamics go here
    
    xdot[0] = u[0]
    xdot[1] = u[1]
    xdot[2] = u[2]
    xdot[3] = 0
        
    return xdot
        
        
