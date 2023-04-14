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
    Calculates the system dynamics equation f(X,U,t),
    n (=18) is number of states.
    
    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
        
    U : numpy.ndarray (m,1)
        input vector, U = [Ax, Ay, Az, p, q, r]^T
        
    Returns
    -------
    Xdot : numpy.ndarray (n,1)
        time derivative of the state vector, system dynamics
    """
    
    n       = X.size
    Xdot    = np.zeros([n,1])
    g = 9.80665                # gravitational acceleration [m/s^2]

    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]
    #####################################################   
    ## System dynamics go here
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)

    A = u*cos_theta + (v*sin_theta + w*cos_phi)*sin_theta  # saving some big terms to make expressions more readable
    B = (v*cos_phi - w*sin_phi)                                    # saving some big terms to make expressions more readable

    Xdot[0] = A*cos_psi - B*sin_psi + Wx
    Xdot[1] = A*sin_psi + B*cos_psi + Wy
    Xdot[2] = -u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta + Wz
    Xdot[3] = Ax - g*sin_theta + r*v - q*w
    Xdot[4] = Ay + g*cos_theta*sin_phi + p*w - r*u
    Xdot[5] = Az + g*cos_theta*cos_phi + q*u - p*v
    Xdot[6] = p + q*sin_phi*tan_theta + r*cos_phi*tan_theta
    Xdot[7] = q*cos_phi - r*sin_phi
    Xdot[8] = q*sin_phi/cos_theta + r*cos_phi/cos_theta
    Xdot[9:] = 0

    return Xdot
        

def kf_calc_Fx(t, X, U):
    """
    Calculates the Jacobian of the system dynamics equation,
    n (=18) is number of states.
    
    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
        
    U : numpy.ndarray (m,1)
        input vector, U = [Ax, Ay, Az, p, q, r]^T
        
    Returns
    -------
    DFx : numpy.ndarray (n,n)
        Jacobian of the system dynamics equation
        
    """
    n = X.size
    DFx = np.zeros([n, n])
    g = 9.80665                # gravitational acceleration [m/s^2]
    
    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]

    #####################################################
    ## Calculate Jacobian matrix of system dynamics
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)

    # F1 derivatives
    DFx_1 = np.zeros([1,n])
    DFx_1[0,3] = cos_theta*cos_psi
    DFx_1[0,4] = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    DFx_1[0,5] = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    DFx_1[0,6] = (v*cos_phi - w*sin_phi)*sin_theta*cos_psi - (-v*sin_phi - w*cos_phi)*sin_psi
    DFx_1[0,7] = (-u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta)*cos_psi
    DFx_1[0,8] = -(u*cos_theta + (v*sin_phi + w*cos_phi)*sin_theta)*sin_psi - (v*cos_phi - w*sin_phi)*cos_psi
    DFx_1[0,9] = 1
    DFx[0,:] = DFx_1

    # F2 derivatives
    DFx_2 = np.zeros([1,n])
    DFx_2[0,3] = cos_theta*sin_psi
    DFx_2[0,4] = sin_phi*sin_theta*sin_psi + cos_phi*cos_psi
    DFx_2[0,5] = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    DFx_2[0,6] = (v*cos_phi - w*sin_phi)*sin_theta*sin_psi + (-v*sin_phi - w*cos_phi)*cos_psi
    DFx_2[0,7] = (-u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta)*sin_psi
    DFx_2[0,8] = (u*cos_theta + (v*sin_phi + w*cos_phi)*sin_theta)*cos_psi - (v*cos_phi - w*sin_phi)*sin_psi
    DFx_2[0,10] = 1
    DFx[1,:] = DFx_2

    # F3 derivatives
    DFx_3 = np.zeros([1,n])
    DFx_3[0,3] = -sin_theta
    DFx_3[0,4] = sin_phi*cos_theta
    DFx_3[0,5] = cos_phi*cos_theta
    DFx_3[0,6] = (v*cos_phi - w*sin_phi)*cos_theta 
    DFx_3[0,7] = -u*cos_theta - (v*sin_theta + w*cos_theta)*sin_theta
    DFx_3[0,11] = 1
    DFx[2,:] = DFx_3

    # F4 derivatives
    DFx_4 = np.zeros([1,n])
    DFx_4[0,4] = r
    DFx_4[0,5] = -q
    DFx_4[0,7] = -g*cos_theta
    DFx_4[0,12:] = np.array([1, 0, 0, 0, -w, v], dtype=object)
    DFx[3,:] = DFx_4

    # F5 derivatives
    DFx_5 = np.zeros([1,n])
    DFx_5[0,3] = -r
    DFx_5[0,5] = p
    DFx_5[0,6] = g*cos_phi*cos_theta
    DFx_5[0,12:] = np.array([0, 1, 0, w, 0, -u], dtype=object)
    DFx[4,:] = DFx_5

    # F6 derivatives
    DFx_6 = np.zeros([1,n])
    DFx_6[0,3] = q
    DFx_6[0,4] = -p
    DFx_6[0,6] = -g*cos_theta*sin_phi
    DFx_6[0,7] = -g*sin_theta*cos_phi
    DFx_6[0,12:] = np.array([0, 0, 1, -v, u, 0], dtype=object)
    DFx[5,:] = DFx_6

    # F7 derivatives
    DFx_7 = np.zeros([1,n])
    DFx_7[0,6] = q*cos_phi*tan_theta - r*sin_phi*tan_theta
    DFx_7[0,7] = q*sin_phi/(cos_theta**2) + r*cos_phi/(cos_theta**2)
    DFx_7[0,15:] = np.array([1, sin_phi*tan_theta, cos_phi*tan_theta], dtype=object)
    DFx[6,:] = DFx_7

    # F8 derivatives
    DFx_8 = np.zeros([1,n])
    DFx_8[0,6] = -q*sin_phi - r*cos_phi
    DFx_8[0,16] = cos_phi
    DFx_8[0,17] = -sin_phi
    DFx[7,:] = DFx_8

    # F9 derivatives
    DFx_9 = np.zeros([1,n])
    DFx_9[0,6] = q*cos_phi/cos_theta - r*sin_phi/cos_theta
    DFx_9[0,7] = q*sin_phi*sin_theta/(cos_theta**2) + r*cos_phi*sin_theta/(cos_theta**2)
    DFx_9[0,16] = sin_phi/cos_theta
    DFx_9[0,17] = cos_phi/cos_theta
    DFx[8,:] = DFx_9

    return DFx
        

def kf_calc_h(t, X, U):
    """
    Calculates the system output equations h(x,u,t),
    nm (=12) is number of outputs.
    
    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
        
    U : numpy.ndarray (m,1)
        input vector, U = [Ax, Ay, Az, p, q, r]^T
        
    Returns
    -------
    Zpred : numpy.ndarray (nm,1)
        system output equations
    """
    n = X.size
    Zpred = np.zeros((12,1))
    g = 9.80665                # gravitational acceleration [m/s^2]

    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]

    #####################################################   
    ## Output equations go here
    #####################################################
    
    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)

    A = u*cos_theta + (v*sin_theta + w*cos_phi)*sin_theta  # saving some big terms to make expressions more readable
    B = (v*cos_phi - w*sin_phi)                                    # saving some big terms to make expressions more readable

    Zpred[0] = x
    Zpred[1] = y
    Zpred[2] = z
    Zpred[3] = A*cos_psi - B*sin_psi + Wx
    Zpred[4] = A*sin_psi + B*cos_psi + Wy
    Zpred[5] = -u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta + Wz
    Zpred[6] = phi
    Zpred[7] = theta
    Zpred[8] = psi
    Zpred[9] = np.sqrt(u**2 + v**2 + w**2)
    Zpred[10] = np.arctan2(w,u)
    Zpred[11] = np.arctan2(v,np.sqrt(u**2 + w**2))

    return Zpred
        

def kf_calc_Hx(t, X, U):
    """
    Calculates the Jacobian of the output dynamics equation, 
    n (=18) is number of states, nm (=12) is number of outputs.

    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
        
    U : numpy.ndarray (m,1)
        input vector, U = [Ax, Ay, Az, p, q, r]^T
        
    Returns
    -------
    Hx : numpy.ndarray (nm,n)   
        Jacobian of the output dynamics equation

    """
    n = X.size
    DHx = np.zeros([12, n])
    g = 9.80665                # gravitational acceleration [m/s^2]
    
    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]

    #####################################################
    ## Calculate Jacobian matrix of output equations
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)


    # H1 derivatives
    DHx_1 = np.zeros([1,n])
    DHx_1[0, 0] = 1
    DHx[0,:] = DHx_1

    # H2 derivatives
    DHx_2 = np.zeros([1,n])
    DHx_2[0, 1] = 1
    DHx[1,:] = DHx_2

    # H3 derivatives
    DHx_3 = np.zeros([1,n])
    DHx_3[0, 2] = 1
    DHx[2,:] = DHx_3

    # H4 derivatives
    DHx_4 = np.zeros([1,n])
    DHx_4[0,3] = cos_theta*cos_psi
    DHx_4[0,4] = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    DHx_4[0,5] = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    DHx_4[0,6] = (v*cos_phi - w*sin_phi)*sin_theta*cos_psi - (-v*sin_phi - w*cos_phi)*sin_psi
    DHx_4[0,7] = (-u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta)*cos_psi
    DHx_4[0,8] = -(u*cos_theta + (v*sin_phi + w*cos_phi)*sin_theta)*sin_psi - (v*cos_phi - w*sin_phi)*cos_psi
    DHx_4[0,9] = 1
    DHx[3,:] = DHx_4
    
    # H5 derivatives
    DHx_5 = np.zeros([1,n])
    DHx_5[0,3] = cos_theta*sin_psi
    DHx_5[0,4] = sin_phi*sin_theta*sin_psi + cos_phi*cos_psi
    DHx_5[0,5] = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    DHx_5[0,6] = (v*cos_phi - w*sin_phi)*sin_theta*sin_psi + (-v*sin_phi - w*cos_phi)*cos_psi
    DHx_5[0,7] = (-u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta)*sin_psi
    DHx_5[0,8] = (u*cos_theta + (v*sin_phi + w*cos_phi)*sin_theta)*cos_psi - (v*cos_phi - w*sin_phi)*sin_psi
    DHx_5[0,10] = 1
    DHx[4,:] = DHx_5

    # H6 derivatives
    DHx_6 = np.zeros([1,n])
    DHx_6[0,3] = -sin_theta
    DHx_6[0,4] = sin_phi*cos_theta
    DHx_6[0,5] = cos_phi*cos_theta
    DHx_6[0,6] = (v*cos_phi - w*sin_phi)*cos_theta 
    DHx_6[0,7] = -u*cos_theta - (v*sin_theta + w*cos_theta)*sin_theta
    DHx_6[0,11] = 1
    DHx[5,:] = DHx_6

    # H7 derivatives
    DHx_7 = np.zeros([1,n])
    DHx_7[0,6] = 1
    DHx[6,:] = DHx_7

    # H8 derivatives
    DHx_8 = np.zeros([1,n])
    DHx_8[0,7] = 1
    DHx[7,:] = DHx_8

    # H9 derivatives
    DHx_9 = np.zeros([1,n])
    DHx_9[0,8] = 1
    DHx[8,:] = DHx_9

    # H10 derivatives
    DHx_10 = np.zeros([1,n])
    DHx_10[0,3] = u/(np.sqrt(u**2 + v**2 + w**2))
    DHx_10[0,4] = v/(np.sqrt(u**2 + v**2 + w**2))
    DHx_10[0,5] = w/(np.sqrt(u**2 + v**2 + w**2))
    DHx[9,:] = DHx_10

    # H11 derivatives
    DHx_11 = np.zeros([1,n])
    DHx_11[0,3] =-w/(u**2 + w**2)
    DHx_11[0,5] = u/(u**2 + w**2)
    DHx[10,:] = DHx_11

    # H12 derivatives
    DHx_12 = np.zeros([1,n])
    DHx_12[0,3] = -u*v/(np.sqrt(u**2 + w**2)*(u**2 + v**2 + w**2))
    DHx_12[0,4] = (u**2 + w**2)/(u**2 + v**2 + w**2)
    DHx_12[0,5] = -w*v/(np.sqrt(u**2 + w**2)*(u**2 + v**2 + w**2))
    DHx[11,:] = DHx_12

    return DHx
        