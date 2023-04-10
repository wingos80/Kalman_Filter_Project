import numpy as np

########################################################################
## Calculates the system output equations h(x,u,t)
########################################################################

def kf_calc_h(t, x, u):
    """
    Calculates the system output equations h(x,u,t),
    nm (=3) is number of outputs.
    
    Parameters
    ----------
    t : float
        
    x : numpy.ndarray (n,1)
        state vector

    u : numpy.ndarray
        input vector

    Returns
    -------
    zpred : numpy.ndarray (nm,1)
        system output equations
    """
    
    # output equations go here
    zpred = np.zeros((3,1))
    zpred[0] = np.arctan2(x[2],x[0])*(1+x[3])
    zpred[1] = np.arctan2(x[1],np.sqrt(x[0]**2+x[2]**2))
    zpred[2] = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
    return zpred
        
        