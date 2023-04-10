########################################################################
# Functions called by the Kalman Filter
# 
#   Author: Wing Chan, adapted from Coen de Visser
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################


import numpy as np


def rk4(fn, xin, uin, t):
    """
    4th order Runge-Kutta method for solving ODEs
    
    Parameters
    ----------
    fn : function
        function handle for the system dynamics equation f(x,u,t)

    xin : numpy.ndarray
        initial state vector
    
    uin : numpy.ndarray
        input vector

    t : numpy.ndarray
        time vector

    Returns
    -------
    t : numpy.ndarray
        time vector (same as input)

    x : numpy.ndarray
        state vector
        """
    
    a   = t[0]
    b   = t[1]
    w   = xin
    N   = 2
    h   = (b - a)/N
    t   = a

    for j in range(1, N+1):
        K1  = h*fn(t, w, uin)
        K2  = h*fn(t+h/2, w+K1/2, uin)
        K3  = h*fn(t+h/2, w+K2/2, uin)
        K4  = h*fn(t+h, w+K3, uin)
        
        w   = w + (K1 + 2*K2 + 2*K3 +K4)/6
        t   = a+j*h
    
    return t, w
        
        
    
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
        