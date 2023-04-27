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
    N   = 1
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


def kf_finite_difference(dx, Ys, step_size=10):
    """
    Function that numerically differentiates the given Ys array along axis=1,
    using a finite difference method. The step size of the differentiator can
    be adjusted to reduce noise.
    Parameters
    ----------
    dx : float
        step size of the input array
    Ys : numpy.ndarray
        input array
    step_size : int
        step size of the finite differentiator, bigger step means less noisey derivative 
    Returns
    -------
    first_derivative : numpy.ndarray
        first derivative of the input array
    second_derivative : numpy.ndarray
        second derivative of the input array
    """
    # Initialize arrays to store derivatives
    first_derivative, second_derivative = np.zeros_like(Ys), np.zeros_like(Ys)
    _ = np.zeros((Ys.shape[0],1))                                                     # useful array to prepend to the derivative arrays

    # First derivative
    temp = np.diff(Ys[:,::step_size], n=1, prepend=_)/(step_size*dx)                  # array containing the first derivatives
    temp[:,0] = temp[:,3]
    temp2 = np.arange(0, Ys.shape[1])

    # Interpolate the first derivative array to the original Ys array size
    for i, val in enumerate(Ys):
        first_derivative[i,:] = np.interp(temp2, temp2[::step_size], temp[i,:])

    # Second derivative
    temp = np.diff(first_derivative[:,::step_size], n=1, prepend=_)/(step_size*dx)    # array containing the second derivatives
    temp[:,0] = temp[:,3]
    temp2 = np.arange(0, Ys.shape[1])

    # Interpolate the second derivative array to the original Ys array size
    for i, val in enumerate(Ys):
        second_derivative[i,:] = np.interp(temp2, temp2[::step_size], temp[i,:])

    return first_derivative, second_derivative


def kf_calc_Fc(m, rho, S, Vs, accs):
    """
    Calculates the control force coefficients Cx, Cy, Cz
    Parameters
    ----------
    m : float
        mass [kg]
    rho : float
        air density [kg/m^3]
    S : float
        wing area [m^2]
    Vs : numpy.ndarray (1,n)
        airspeed [m/s]
    accs : numpy.ndarray (3,n)
        linear accelerations [m/s^2]
    Returns
    -------
    Cx's : numpy.ndarray (3,n)
        control force coefficients along the x, y, z axes"""
    return m*accs/(0.5*rho*S*Vs**2)


def kf_calc_Mc(rho, b, c, S, I, Vs, rates, accs):
    """
    Calculates the control moment coefficients Cm, Cl, Cn
    Parameters
    ----------
    rho : float
        air density [kg/m^3]
    b : float
        wing span [m]
    c : float
        mean aerodynamic chord [m]
    S : float
        wing area [m^2]
    I : numpy.ndarray (3,1)
        moment of inertia matrix [kg*m^2]
    Vs : numpy.ndarray (1,n)
        airspeed [m/s]
    rates : numpy.ndarray (3,n)
        angular rates [rad/s]
    accs : numpy.ndarray (3,n)
        linear accelerations [m/s^2]
    Returns
    -------
    Cl : float
        control moment coefficient around the roll axis
    Cm : float
        control moment coefficient around the pitch axis
    Cn : float
        control moment coefficient around the yaw axis    
    """
    Cl = (accs[0]*I[0] + rates[1]*rates[2]*(I[2] - I[1]) - (rates[0]*rates[1] + accs[2])*I[3])/(0.5*rho*Vs**2*S*b)
    Cm = (accs[1]*I[1] + rates[1]*rates[0]*(I[0] - I[2]) + (rates[0]**2 - rates[2]**2)*I[3])/(0.5*rho*Vs**2*S*c)
    Cn = (accs[2]*I[2] + rates[0]*rates[1]*(I[1] - I[0]) + (rates[1]*rates[2] - accs[0])*I[3])/(0.5*rho*Vs**2*S*b)
    return np.array([Cl, Cm, Cn])


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
    Lx, Ly, Lz, Lp, Lq, Lr = X[12], X[13], X[14], X[15], X[16], X[17]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]
    #####################################################   
    ## System dynamics go here
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)

    A = u*cos_theta + (v*sin_phi + w*cos_phi)*sin_theta  # saving some big terms to make expressions more readable
    B = (v*cos_phi - w*sin_phi)                                    # saving some big terms to make expressions more readable

    Xdot[0] = A*cos_psi - B*sin_psi + Wx
    Xdot[1] = A*sin_psi + B*cos_psi + Wy
    Xdot[2] = -u*sin_theta + (v*sin_phi + w*cos_phi)*cos_theta + Wz
    Xdot[3] = (Ax - Lx) - g*sin_theta + (r - Lr)*v - (q - Lq)*w
    Xdot[4] = (Ay - Ly) + g*cos_theta*sin_phi + (p - Lp)*w - (r - Lr)*u
    Xdot[5] = (Az - Lz)+ g*cos_theta*cos_phi + (q - Lq)*u - (p - Lp)*v
    Xdot[6] = (p - Lp) + (q - Lq)*sin_phi*tan_theta + (r - Lr)*cos_phi*tan_theta
    Xdot[7] = (q - Lq)*cos_phi - (r - Lr)*sin_phi
    Xdot[8] = (q - Lq)*sin_phi/cos_theta + (r - Lr)*cos_phi/cos_theta

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
    Lx, Ly, Lz, Lp, Lq, Lr = X[12], X[13], X[14], X[15], X[16], X[17]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]

    #####################################################
    ## Calculate Jacobian matrix of system dynamics
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    tan_theta = np.tan(theta)

    # saving some commonly used terms to speed up code
    C = (v*sin_phi + w*cos_phi)
    D = sin_theta*cos_psi
    E = sin_theta*sin_psi

    # F1 derivatives
    DFx[0,3] = cos_theta*cos_psi
    DFx[0,4] = sin_phi*D - cos_phi*sin_psi
    DFx[0,5] = cos_phi*D + sin_phi*sin_psi
    DFx[0,6] = (v*cos_phi - w*sin_phi)*D - (-v*sin_phi - w*cos_phi)*sin_psi
    DFx[0,7] = (-u*sin_theta + C*cos_theta)*cos_psi
    DFx[0,8] = -(u*cos_theta + C*sin_theta)*sin_psi - (v*cos_phi - w*sin_phi)*cos_psi
    DFx[0,9] = 1

    # F2 derivatives
    DFx[1,3] = cos_theta*sin_psi
    DFx[1,4] = sin_phi*E + cos_phi*cos_psi
    DFx[1,5] = cos_phi*E - sin_phi*cos_psi
    DFx[1,6] = (v*cos_phi - w*sin_phi)*E + (-v*sin_phi - w*cos_phi)*cos_psi
    DFx[1,7] = (-u*sin_theta + C*cos_theta)*sin_psi
    DFx[1,8] = (u*cos_theta + C*sin_theta)*cos_psi - (v*cos_phi - w*sin_phi)*sin_psi
    DFx[1,10] = 1

    # F3 derivatives
    DFx[2,3] = -sin_theta
    DFx[2,4] = sin_phi*cos_theta
    DFx[2,5] = cos_phi*cos_theta
    DFx[2,6] = (v*cos_phi - w*sin_phi)*cos_theta 
    DFx[2,7] = -u*cos_theta - (v*sin_theta + w*cos_theta)*sin_theta
    DFx[2,11] = 1

    # F4 derivatives
    DFx[3,4] = (r - Lr)
    DFx[3,5] = -(q - Lq)
    DFx[3,7] = -g*cos_theta
    DFx[3,12:] = np.array([-1, 0, 0, 0, w, -v], dtype=object)

    # F5 derivatives
    DFx[4,3] = -(r - Lr)
    DFx[4,5] = (p - Lp)
    DFx[4,6] = g*cos_phi*cos_theta
    DFx[4,12:] = np.array([0, -1, 0, -w, 0, u], dtype=object)

    # F6 derivatives
    DFx[5,3] = (q - Lq)
    DFx[5,4] = -(p - Lp)
    DFx[5,6] = -g*cos_theta*sin_phi
    DFx[5,7] = -g*sin_theta*cos_phi
    DFx[5,12:] = np.array([0, 0, -1, v, -u, 0], dtype=object)
    # F7 derivatives
    DFx[6,6] = (q - Lq)*cos_phi*tan_theta - (r - Lr)*sin_phi*tan_theta
    DFx[6,7] = ((q - Lq)*sin_phi + (r - Lr)*cos_phi)/(cos_theta**2)
    DFx[6,15:] = np.array([-1, -sin_phi*tan_theta, -cos_phi*tan_theta], dtype=object)

    # F8 derivatives
    DFx[7,6] = -(q - Lq)*sin_phi - (r - Lr)*cos_phi
    DFx[7,16] = -cos_phi
    DFx[7,17] = sin_phi

    # F9 derivatives
    DFx[8,6] = (q - Lq)*cos_phi/cos_theta - (r - Lr)*sin_phi/cos_theta
    DFx[8,7] = ((q - Lq)*sin_phi*sin_theta + (r - Lr)*cos_phi*sin_theta)/(cos_theta**2)
    DFx[8,16] = -sin_phi/cos_theta
    DFx[8,17] = -cos_phi/cos_theta

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

    # saving some commonly used terms to speed up code
    C = (v*sin_phi + w*cos_phi)
    D = sin_theta*cos_psi
    E = sin_theta*sin_psi
    V = np.sqrt(u**2 + v**2 + w**2)

    # H1 derivatives
    DHx[0, 0] = 1

    # H2 derivatives
    DHx[1, 1] = 1

    # H3 derivatives
    DHx[2, 2] = 1

    # H4 derivatives
    DHx[3,3] = cos_theta*cos_psi
    DHx[3,4] = sin_phi*D - cos_phi*sin_psi
    DHx[3,5] = cos_phi*D + sin_phi*sin_psi
    DHx[3,6] = (v*cos_phi - w*sin_phi)*D - (-v*sin_phi - w*cos_phi)*sin_psi
    DHx[3,7] = (-u*sin_theta + C*cos_theta)*cos_psi
    DHx[3,8] = -(u*cos_theta + C*sin_theta)*sin_psi - (v*cos_phi - w*sin_phi)*cos_psi
    DHx[3,9] = 1
    
    # H5 derivatives
    DHx[4,3] = cos_theta*sin_psi
    DHx[4,4] = sin_phi*E + cos_phi*cos_psi
    DHx[4,5] = cos_phi*E - sin_phi*cos_psi
    DHx[4,6] = (v*cos_phi - w*sin_phi)*E + (-v*sin_phi - w*cos_phi)*cos_psi
    DHx[4,7] = (-u*sin_theta + C*cos_theta)*sin_psi
    DHx[4,8] = (u*cos_theta + C*sin_theta)*cos_psi - (v*cos_phi - w*sin_phi)*sin_psi
    DHx[4,10] = 1

    # H6 derivatives
    DHx[5,3] = -sin_theta
    DHx[5,4] = sin_phi*cos_theta
    DHx[5,5] = cos_phi*cos_theta
    DHx[5,6] = (v*cos_phi - w*sin_phi)*cos_theta 
    DHx[5,7] = -u*cos_theta - (v*sin_theta + w*cos_theta)*sin_theta
    DHx[5,11] = 1

    # H7 derivatives
    DHx[6,6] = 1

    # H8 derivatives
    DHx[7,7] = 1

    # H9 derivatives
    DHx[8,8] = 1

    # H10 derivatives
    DHx[9,3] = u/(V)
    DHx[9,4] = v/(V)
    DHx[9,5] = w/(V)

    # H11 derivatives
    DHx[10,3] =-w/(u**2 + w**2)
    DHx[10,5] = u/(u**2 + w**2)

    # H12 derivatives
    DHx[11,3] = -u*v/(np.sqrt(u**2 + w**2)*(V**2))
    DHx[11,4] = (u**2 + w**2)/(V**2)
    DHx[11,5] = -w*v/(np.sqrt(u**2 + w**2)*(V**2))\

    return DHx
        