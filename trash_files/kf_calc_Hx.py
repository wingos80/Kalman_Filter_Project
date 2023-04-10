########################################################################
## Calculates the Jacobian of the output dynamics equation 
########################################################################
import numpy as np

def kf_calc_Hx(t, x, u):
    """
    Calculates the Jacobian of the output dynamics equation, 
    n is number of states, nm (=3) is number of outputs.

    Parameters
    ----------
    t : float

    x : numpy.ndarray (n,1)
        state vector

    u : numpy.ndarray
        input vector

    Returns
    -------
    Hx : numpy.ndarray (nm,n)   
        Jacobian of the output dynamics equation

    """
    n = x.size
    
    # calculate Jacobian matrix of system dynamics
    Hx = np.zeros([3, n])
    # derivatives of h1
    Hx[0,0] = -x[2]/(x[0]**2 + x[2]**2)*(1 + x[3])
    Hx[0,2] = x[0]/(x[0]**2 + x[2]**2)*(1 + x[3])
    Hx[0,3] = np.arctan2(x[2],x[0])

    # derivatives of h2
    Hx[1,0] = -x[1]*x[2]/(np.sqrt(x[0]**2 + x[2]**2)*(x[0]**2 + x[1]**2 + x[2]**2))
    Hx[1,1] = np.sqrt(x[0]**2 + x[2]**2)/(x[0]**2 + x[1]**2 + x[2]**2)
    Hx[1,2] = -x[0]*x[1]/(np.sqrt(x[0]**2 + x[2]**2)*(x[0]**2 + x[1]**2 + x[2]**2))

    # derivatives of h3
    Hx[2,0] = x[0]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    Hx[2,1] = x[1]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    Hx[2,2] = x[2]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)


    return Hx
        