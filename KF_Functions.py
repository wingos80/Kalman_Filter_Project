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
        function handle for the derivatives of the state vector

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

    xout : numpy.ndarray
        state vector values at next time step
        """
    
    a   = t[0]
    b   = t[1]
    x   = xin
    N   = 2
    h   = (b - a)/N
    t   = a

    for j in range(1, N+1):
        K1  = h*fn(t, x, uin)
        K2  = h*fn(t+h/2, x+K1/2, uin)
        K3  = h*fn(t+h/2, x+K2/2, uin)
        K4  = h*fn(t+h, x+K3, uin)
        
        x   = x + (K1 + 2*K2 + 2*K3 +K4)/6
        t   = a+j*h
    xout = x
    return t, xout
        
        
# x = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz]
# u = [Ax, Ay, Az, p, q, r]
def kf_calc_f(t, X, U):
    """
    Calculates the system dynamics equation f(X,U,t)
    
    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector
        
    U : numpy.ndarray
        input vector
        
    Returns
    -------
    Xdot : numpy.ndarray (n,1)
        time derivative of the state vector, system dynamics
    """
    
    n       = X.size
    Xdot    = np.zeros([n,1])
    g = 9.80665             # gravitational acceleration [m/s^2]

    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]

    # system dynamics go here
    A = u*np.cos(theta) + (v*np.sin(theta) + w*np.cos(phi))*np.sin(theta)  # saving some big terms to make expressions more readable
    B = (v*np.cos(phi) - w*np.sin(phi))                                    # saving some big terms to make expressions more readable

    Xdot[0] = A*np.cos(psi) - B*np.sin(psi) + Wx
    Xdot[1] = A*np.sin(psi) + B*np.cos(psi) + Wy
    Xdot[2] = -u*np.sin(theta) + (v*np.sin(phi) + w*np.cos(phi))*np.cos(theta) + Wz
    Xdot[3] = Ax - g*np.sin(theta) + r*v - q*w
    Xdot[4] = Ay + g*np.cos(theta)*np.sin(phi) + p*w - r*u
    Xdot[5] = Az + g*np.cos(theta)*np.cos(phi) + q*u - p*v
    Xdot[6] = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    Xdot[7] = q*np.cos(phi) - r*np.sin(phi)
    Xdot[8] = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
    Xdot[9] = 0
    Xdot[10] = 0
    Xdot[11] = 0

    return Xdot
        

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
        