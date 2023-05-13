########################################################################
# Functions called by the Kalman Filter
# 
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 27-04-2023
########################################################################
import numpy as np
import matplotlib.pyplot as plt
from Optimize import *
from scipy.optimize import minimize
class model:
    """A class for storing the model data and performing the model parameter estimations
    Currently can estimate the model parameters  using: OLS, MLE, or RLS (MLE likelihood function likely incorrect)"""
    def __init__(self, regression_matrix, name="unnamed model", verbose=False):
        self.name              = name    # name of the model
        self.regression_matrix = regression_matrix    # the regression matrix of size nxm, m=num of parameters and n=num of measurements 
        self.n_params = self.regression_matrix.shape[1]
        self.measurements      = None    # the measurements of size nx1
        self.verbose           = verbose
        self.trigger = True

        name_of_var = 1


    def OLS_estimate(self):
        """
        Performs an ordinary least squares estimation of the model parameters
        """
        if self.verbose: print(f'\nEstimating parameters with ordinary least squares...\n')
        A = self.regression_matrix
        self.OLS_cov = np.linalg.inv(A.T@A)
        self.OLS_params = self.OLS_cov@A.T@self.measurements

        self.OLS_y = (A@self.OLS_params)
        self.OLS_RMSE = self.calc_RMSE(self.OLS_y, self.measurements)  # calculating the RMSE of the OLS estimate
        if self.verbose: print(f'\nFinished OLS\n')

    def MLE_estimate(self, solver="ES"):
        """
        Performs a maximum likelihood estimation of the model parameters
        """
        # covaraince is NxN, with N = number of measurements.
        # function to maximize:
        #  y = self.measurements
        #  p = self.model_evaluant
        #  np.log(2*np.pi)**(N/2)*np.linalg.det(cov)**0.5 + 0.5(y-p)@np.linal.inv(cov)@(y-p).T
        
        # two possible errors:
        # 1. the log likelihood should be max(-np.log(2*np.pi)**(N/2)*np.linalg.det(cov)**0.5 - 0.5(y-p)@np.linal.inv(cov)@(y-p))
        # 2. the cov matrix should be MxM, M being number of params
        if self.verbose: print(f'\nEstimating parameters with maximum likelihood...\n')
        g    = 20
        n    = 100                         # number of particles
        A    = self.regression_matrix

        if solver=="ES":
            if self.verbose: print(f'\nRunning Evolutionary Strategies to optimize the Maximum Likelihood...\n')
            test = ES(fitness_function=self.log_likelihood, num_dimensions=self.n_params, num_generations=200, num_offspring_per_individual=6, verbose=False)
            if self.verbose: print(f'\nFinished Evolutionary Strategies Maximum Likelihood \n')
            self.MLE_params = test.run().reshape(self.n_params,1)
            self.MLE_best   = test.group_best_fit
        elif solver=="scipy":
            if self.verbose: print(f'\nRunning Scipy to optimize the Maximum Likelihood...\n')
            self.MLE_params = np.random.rand(self.n_params,1)
            self.MLE_params = minimize(self.log_likelihood, self.MLE_params).x
            if self.verbose: print(f'\nFinished Scipy Maximum Likelihood \n')
            self.MLE_best   = 1
        else:
            raise ValueError(f"Solver {solver} not recognized. Please choose either 'ES' or 'scipy'")
        
        self.MLE_y      = A@self.MLE_params.reshape(self.n_params,1)       
        self.MLE_RMSE   = self.calc_RMSE(self.MLE_y, self.measurements)    # calculating the RMSE of the MLE estimate
        if self.verbose: print(f'\nFinished MLE\n')

    def RLS_estimate(self):
        """
        Performs a recursive least squares estimation of the model parameters
        """
        if self.verbose: print(f'\nEstimating parameters with recursive least squares...\n')
        RLS_params          = np.zeros((self.n_params,1))   # initializing the RLS parameters
        P                   = np.eye(self.n_params)         # initializing the RLS covariance matrix
        A                   = self.regression_matrix        # shorter names for readability
        y                   = self.measurements             # shorter names for readability    
        N                   = y.size                        # number of measurements

        self.RLS_all_params = np.zeros((self.n_params,N))   # initializing the RLS parameters for all measurements

        for i in range(N):
            smol_a          = A[[i],:]                      
            smol_y          = y[[i],:]
            RLS_gain        = P@smol_a.T@np.linalg.inv(smol_a@P@smol_a.T+1)
            RLS_params = RLS_params + RLS_gain@(smol_y - smol_a@RLS_params)
            P               = (np.eye(self.n_params) - RLS_gain@smol_a)@P
            self.RLS_all_params[:,[i]] = RLS_params

            if self.verbose:
                plt.figure(1)
                plot_y = A@RLS_params
                y_dots = y[:i+1,0]
                x_dots = np.arange(i+1)
                plt.scatter(x_dots, y_dots, label="Measurements", s=1, marker="x", alpha=0.6)
                plt.plot(plot_y, label="RLS")
                plt.grid()
                plt.ylim(-150,150)
                plt.pause(0.05)
                plt.clf()

        self.RLS_y      = A@RLS_params
        self.RLS_cov    = P
        self.RLS_params = RLS_params
        self.RLS_RMSE = self.calc_RMSE(self.RLS_y, self.measurements)    # calculating the RMSE of the RLS estimate
        if self.verbose: print(f'\nFinished RLS\n')

    def log_likelihood(self, MLE_params):
        """
        Calculates an augmented log likelihood of the model given the measurements and the model parameters
        """
        MLE_params = MLE_params.reshape(self.n_params,1)    # making the params into a column vector so that it can be used in matrix calculations
        y = self.measurements
        p = self.regression_matrix@MLE_params
        N = y.size
        epsilon = y-p
        ensemble_cov = epsilon.T@epsilon/N
        likelihood = 0.5/ensemble_cov*epsilon.T@epsilon + N/2*np.log(ensemble_cov) + 0.5*np.log(2*np.pi)
        # likelihood = np.sum(epsilon**2)
        # likelihood = np.sqrt(epsilon.T@epsilon)   
        # likelihood = np.sqrt(np.sum(epsilon**2))
        return likelihood


    def calc_RMSE(self, model_y, measured_y):
        """
        Calculates the root mean squared error between the model output and the measured output
        """
        return np.sqrt(np.sum((model_y - measured_y)**2)/self.measurements.size)



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
    
    # First derivative
    temp = (np.diff(Ys[:,:-step_size:step_size], n=1) + np.diff(Ys[:,step_size::step_size], n=1))/(2*step_size*dx)                  # array containing the first derivatives
    temp = np.append(temp[:,0].reshape(Ys.shape[0],1), np.append(temp, temp[:,-1].reshape(Ys.shape[0],1), axis=1), axis=1)                                                # append the last column of the array to the end
    temp2 = np.arange(0, Ys.shape[1])

    # Interpolate the first derivative array to the original Ys array size
    for i, val in enumerate(Ys):
        first_derivative[i,:] = np.interp(temp2, temp2[::step_size], temp[i,:])

    # # Second derivative
    temp = (np.diff(first_derivative[:,:-step_size:step_size], n=1) + np.diff(first_derivative[:,step_size::step_size], n=1))/(2*step_size*dx)                  # array containing the first derivatives
    temp[:,0] = temp[:,1]
    temp[:,-1] = temp[:,-2]
    temp = np.append(temp[:,0].reshape(Ys.shape[0],1), np.append(temp, temp[:,-1].reshape(Ys.shape[0],1), axis=1), axis=1)                                                # append the last column of the array to the end

    # Interpolate the first derivative array to the original Ys array size
    for i, val in enumerate(Ys):
        second_derivative[i,:] = np.interp(temp2, temp2[::step_size], temp[i,:])

    return first_derivative, second_derivative


def kf_calc_Fc(m, rho, S, Vs, accs):
    """
    Calculates the control force coefficients along the 3 body
    axes x, y, and z: Cx, Cy, Cz
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
    Calculates the control moment coefficients along the 3 body
    axes roll, pithc, and yaw: Cl, Cm, Cn
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
        

def kf_calc_Fu(t, X, U):
    
    n = X.size
    nu = U.size
    DFu = np.zeros([n, nu])
    g = 9.80665                # gravitational acceleration [m/s^2]
    
    # saving the individual state and input names to make the code more readable
    u, v, w, phi, theta = X[3], X[4], X[5], X[6], X[7]

    #####################################################
    ## Calculate Jacobian matrix of system dynamics
    #####################################################

    # saving the trig function values to make computations faster
    sin_phi, cos_phi, cos_theta = np.sin(phi), np.cos(phi), np.cos(theta)
    tan_theta = np.tan(theta)


    # F1 derivatives
    DFu[3,0] = 1
    DFu[3,4] = -w
    DFu[3,5] = v

    # F2 derivatives
    DFu[4,1] = 1
    DFu[4,3] = w
    DFu[4,5] = -u

    # F3 derivatives
    DFu[5,2] = 1
    DFu[5,3] = -v
    DFu[5,4] = u

    # F4 derivatives
    DFu[6,3] = 1
    DFu[6,4] = sin_phi*tan_theta
    DFu[6,5] = cos_phi*tan_theta

    # F5 derivatives
    DFu[7,4] = cos_phi
    DFu[7,5] = -sin_phi

    # F6 derivatives
    DFu[8,4] = sin_phi/cos_theta
    DFu[8,5] = cos_phi/cos_theta
    return DFu
        

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
        

def c2d(Fx, dt, n_plus=0):
    """
    function to transform a continuus time state space model to a discrete time state space model
    using the matrix exponential method
    Parameters
    ----------
    Fx : numpy.ndarray (n,n)
        state matrix
    dt : float
        sampling time
    n_plus : int
        0 for the A matrix, 1 for the control matrices (B, G...)
    Returns
    -------
    Fx_d : numpy.ndarray (n,n)
        discrete time state matrix"""
    Fx_d = np.zeros_like(Fx)
    for n in range(10):
        temp = np.eye(Fx.shape[0])
        for i in range(n):
            temp = temp@Fx
        Fx_d += temp*(dt**(n+n_plus))/np.math.factorial(n+n_plus)
    return Fx_d


def OLS_estimation(alphas_k, betas_k, V, ang_rate, FCs_k, MCs_k, U_k, b, c):
    
    """
    Function to estimate the parameters of the force and moment coefficient models using OLS
    DEPRECATED
    Parameters
    ----------
    """
    # Formulating an OLS estimation of the parameters for the force coefficient models:
    # FCs[0,:] = CX = CX0 + CX_alpha*alpha + CX_alpha2*alpha**2 + CX_q*qc/V + CX_delta_e*delta_e + CX_Tc*Tc
    # FCs[1,:] = CY = CY0 + CY_beta*beta + CY_p*pb/2V + CY_r*rb/2V + CY_delta_a*delta_a + CY_delta_r*delta_r
    # FCs[2,:] = CZ = CZ0 + CZ_alpha*alpha + CZ_q*qc/V + CZ_de*de + CZ_Tc*Tc
    N = alphas_k.size
    
    A = np.zeros((N, 6))                                      # regression matrix 
    A[:,0] = 1
    A[:,1] = alphas_k
    A[:,2] = alphas_k**2
    A[:,3] = (ang_rate[1,:]*c)/V
    A[:,4] = U_k[1,:]
    A[:,5] = U_k[3,:]
    cov_CX = np.linalg.inv(A.T@A)
    theta_CX = cov_CX@A.T@FCs_k[0,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the CX polynomial

    A = np.zeros((N, 6))                                      # regression matrix
    A[:,0] = 1
    A[:,1] = betas_k
    A[:,2] = (ang_rate[0,:]*b)/(2*V)
    A[:,3] = (ang_rate[2,:]*b)/(2*V)
    A[:,4] = U_k[0,:]
    A[:,5] = U_k[2,:]
    cov_CY = np.linalg.inv(A.T@A)
    theta_CY = cov_CY@A.T@FCs_k[1,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the CY polynomial

    A = np.zeros((N, 5))                                       # regression matrix
    A[:,0] = 1
    A[:,1] = alphas_k
    A[:,2] = (ang_rate[1,:]*c)/V
    A[:,3] = U_k[1,:]
    A[:,4] = U_k[3,:]
    cov_CZ = np.linalg.inv(A.T@A)
    theta_CZ = cov_CZ@A.T@FCs_k[2,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the CZ polynomial


    # Formulating an OLS estimation of the parameters for the moment coefficient models:
    # MCs[0,:] = Cl = Cl0 + Cl_beta*beta + Cl_p*pb/2V + Cl_r*rb/2V + Cl_delta_a*delta_a + Cl_delta_r*delta_r
    # MCs[1,:] = Cm = Cm0 + Cm_alpha*alpha + Cm_q*qc/V + Cm_delta_e*delta_e + Cm_Tc*Tc
    # MCs[2,:] = Cn = Cn0 + Cn_beta*beta + Cn_p*pb/2V + Cn_r*rb/2V + Cn_delta_a*delta_a + Cn_delta_r*delta_r

    A = np.zeros((N, 6))                                       # regression matrix
    A[:,0] = 1
    A[:,1] = betas_k
    A[:,2] = (ang_rate[0,:]*b)/(2*V)
    A[:,3] = (ang_rate[2,:]*b)/(2*V)
    A[:,4] = U_k[0,:]
    A[:,5] = U_k[2,:]
    cov_Cl = np.linalg.inv(A.T@A)
    theta_Cl = cov_Cl@A.T@MCs_k[0,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the Cl polynomial

    A = np.zeros((N, 5))                                      # regression matrix
    A[:,0] = 1
    A[:,1] = alphas_k
    A[:,2] = (ang_rate[1,:]*c)/V
    A[:,3] = U_k[1,:]
    A[:,4] = U_k[3,:]
    cov_Cm = np.linalg.inv(A.T@A)
    theta_Cm = cov_Cm@A.T@MCs_k[1,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the Cm polynomial

    A = np.zeros((N, 6))                                      # regression matrix
    A[:,0] = 1
    A[:,1] = betas_k
    A[:,2] = (ang_rate[0,:]*b)/(2*V)
    A[:,3] = (ang_rate[2,:]*b)/(2*V)
    A[:,4] = U_k[0,:]
    A[:,5] = U_k[2,:]
    cov_Cn = np.linalg.inv(A.T@A)
    theta_Cn = cov_Cn@A.T@MCs_k[2,:]            # theta = (A.T*A)^-1*A.T*C_m, parameters for the Cn polynomial


    # Storing the master model 
    full_model = np.array([theta_CX, theta_CY, theta_CZ, theta_Cl, theta_Cm, theta_Cn])    # storing all the force and moment coefficients
    covariances = np.array([cov_CX, cov_CY, cov_CZ, cov_Cl, cov_Cm, cov_Cn])              # storing all the covariances
    return full_model, covariances

