import numpy as np
import time, sys, os, control.matlab

class IEKF:
    """
    A class to implement the Iterated Extended Kalman Filter (IEKF) for a nonlinear system
    """
    def __init__(self, N, nm, dt=0.01, epsilon=10**(-10), maxIterations=200):
        """
        Initialize the IEKF class
        
        Parameters
        ----------
        N : int
            number of samples in the data set
            
        nm : int
            number of measurements, output dimension 
            
        dt : float (optional)
            time step (default is 0.01)
            
        epsilon : float (optional)
            IEKF iteration difference threshold (default is 10**(-10))
            
        maxIterations : int (optional)
            maximum amount of iterations per IEKF step (default is 200)    
        """
        
        # set kalman filter parameters
        self.epsilon = epsilon          # IEKF iteration difference threshold
        self.max_itr = maxIterations    # maximum amount of iterations per IEKF step
        self.err     = 2*epsilon        # initialize error
        self.itr     = 0                # initialize iteration counter
        self.eta2    = None             # initialize eta2

        # set some system parameters
        self.N       = N                # number of samples
        self.nm      = nm               # number of measurements

        # initialize time
        self.t_k     = 0                # initial time at k
        self.t_k1    = dt               # initial time at k+1
        self.dt      = dt               # time step


    def setup_system(self, x_0, f, h, Fx, Hx, B, G, integrator):
        """
        Set up the system dynamics, output equations, initial guess of system
        state.

        Parameters
        ----------
        x_0 : np.array
            Initial guess of system state

        f : function
            System dynamics function
        
        h : function
            Output equation function

        Fx : function
            Jacobian of system dynamics function

        Hx : function    
            Jacobian of output equation function

        B : np.array
            Input matrix

        G : np.array    
            Input noise matrix

        integrator : function
            selected integration scheme for integrating the system dynamics
        """
        # x(0|0) = E(x_0)
        self.x_k1_k1 = x_0                # initial guess of system state
        self.n       = self.x_k1_k1.size  # tracking number of states

        self.B       = B                  # input matrix
        self.m       = self.B.shape[1]    # tracking number of inputs

        self.G       = G                  # system noise matrix

        # system dynamics and outputs
        self.f       = f                  # system dynamics
        self.h       = h                  # output equation
        self.Fx      = Fx                 # Jacobian of system dynamics
        self.Hx      = Hx                 # Jacobian of output equation

        # saving the integation scheme chosen
        self.integrator = integrator

        # set up memory vectors
        self.setup_traces()


    def setup_traces(self):
        """ 
        Set up the memory vectors for the values we want to trace
        """

        self.XX_k1_k1    = np.zeros([self.n, self.N])   # memory for filter state estimate
        self.PP_k1_k1    = np.zeros([self.n, self.N])   # memory for filter state covariance
        self.STD_x_cor   = np.zeros([self.n, self.N])   # memory for filter state standard deviation
        self.ZZ_pred     = np.zeros([self.nm, self.N])  # memory for filter measurement estimate
        self.STD_z       = np.zeros([self.nm, self.N])  # memory for filter measurement standard deviation

        self.IEKFitcount = np.zeros([self.N])           # memory for IEKF iteration count
        self.eye_n       = np.eye(self.n)               # identity matrix of size n for use in computations


    def setup_covariances(self, P_stds, Q_stds, R_stds):
        """
        Set up the system state and noise covariance matrices

        Parameters
        ----------
        P_stds : list
            List of standard deviations for the initial state estimate covariance matrix

        Q_stds : list
            List of standard deviations for the system noise covariance matrix

        R_stds : list
            List of standard deviations for the measurement noise covariance matrix
        """

        self.P_0 = np.diag([x**2 for x in P_stds]) # P(0|0) = P(0)
        self.Q = np.diag([x**2 for x in Q_stds])
        self.R = np.diag([x**2 for x in R_stds])

        self.P_k1_k1 = self.P_0


    def predict_and_discretize(self, U_k):
        """
        Predict the next state and discretize the system dynamics and output
        equations

        Parameters
        ----------
        U_k : np.array
            Input vector for the k-th time step
        """

        # x(k+1|k) (prediction)
        self.t, self.x_k1_k   = self.integrator(self.f, self.x_k1_k1, U_k, [self.t_k, self.t_k1])   # add in U_k vector

        # Calc Jacobians, Phi(k+1, k), and Gamma(k+1, k)
        F_jacobian  = self.Fx(0, self.x_k1_k, U_k)
        ss_B        = control.matlab.ss(F_jacobian, self.B, np.zeros((self.nm, self.n)), np.zeros((self.nm, self.m)))  # state space model with A and B matrices, to identify phi and psi matrices
        ss_G        = control.matlab.ss(F_jacobian, self.G, np.zeros((self.nm, self.n)), np.zeros((self.nm, self.n)))  # state space model with A and G matrices, to identify phi and gamma matrices
        

        # Continuous to discrete time transformation of state space matrices
        Psi         = control.matlab.c2d(ss_B, self.dt).B   # discretized B matrix
        Phi         = control.matlab.c2d(ss_G, self.dt).A   # discretized A matrix
        Gamma       = control.matlab.c2d(ss_G, self.dt).B   # discretized G matrix

        # P(k+1|k) (prediction covariance matrix)
        self.P_k1_k = Phi@self.P_k1_k1@Phi.transpose() + Gamma@self.Q@Gamma.transpose()
        self.eta2   = self.x_k1_k


    def run_iteration(self, U_k, Z_k):
        """
        Run one iteration of the IEKF

        Parameters
        ----------
        U_k : np.array
            Input vector for the k-th time step

        Z_k : np.array
            Measurement vector for the k-th time step
        """
        self.itr +=1
        self.eta1 = self.eta2

        # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
        H_jacobian  = self.Hx(0, self.eta1, U_k)
        
        # Observation and observation error predictions
        self.z_k1_k      = self.h(0, self.eta1, U_k)                            # prediction of observation (for validation)   
        P_zz        = H_jacobian@self.P_k1_k@H_jacobian.transpose() + self.R    # covariance matrix of observation error (for validation)   

        # Raise exception in case the covariance matrix is too small
        try:
            self.std_z       = np.sqrt(P_zz.diagonal())          # standard deviation of observation error (for validation)    
        except:
            self.std_z       = np.zeros([self.nm, 1])                 # standard deviation of observation error (for validation)  

        # K(k+1) (gain), Kalman Gain
        Kalman_Gain             = self.P_k1_k@H_jacobian.transpose()@np.linalg.inv(P_zz)
    
        # New observation
        temp = np.reshape(Z_k, (3,1))                  # Need to reshape this Z array to a column vector
        eta2        = self.x_k1_k + Kalman_Gain@(temp - self.z_k1_k - H_jacobian@(self.x_k1_k - self.eta1))
        self.err         = np.linalg.norm(eta2-self.eta1)/np.linalg.norm(self.eta1)  # difference in updated state estimate 
                                                                                     # and previous state estimate
        self.H_jacobian  = H_jacobian
        self.Kalman_Gain = Kalman_Gain
        self.eta2        = eta2


    def update(self, k):
        """
        Update the state and state covariance estimates, and store the results
        """

        # Updating the state estimate
        self.x_k1_k1 = self.eta2

        # Making some local variables for readability
        K   = self.Kalman_Gain
        H_j = self.H_jacobian

        # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 
        P_k1_k1          = (self.eye_n - K@H_j)@self.P_k1_k@(self.eye_n - K@H_j).transpose() + K@self.R@K.transpose()    
        self.std_x_cor   = np.sqrt(P_k1_k1.diagonal())        # standard deviation of state estimation error (for validation)

        # Update to next time step
        self.t_k         = self.t_k1 
        self.t_k1        = self.t_k1 + self.dt

        self.P_k1_k1     = P_k1_k1

        # Store results, need to flatten the arrays to store in a matrix
        self.ZZ_pred[:,k]    = self.z_k1_k.flatten()              # predicted observation
        self.XX_k1_k1[:,k]   = self.x_k1_k1.flatten()             # estimated state
        self.PP_k1_k1[:,k]   = self.P_k1_k1.diagonal().flatten()  # estimated state covariance (for validation)
        self.STD_x_cor[:,k]  = self.std_x_cor.flatten()           # standard deviation of state estimation error (for validation)
        self.STD_z[:,k]      = self.std_z.flatten()               # standard deviation of observation error (for validation)

        self.IEKFitcount[k] = self.itr
        self.itr = 0
        self.err = 2*self.epsilon


    def not_converged(self):
        """
        Check if the IEKF has converged
        """
        bool_val = self.err > self.epsilon and self.itr < self.max_itr
        if self.itr > self.max_itr:
            print('Maximum number of iterations reached')
            print(f'Delta eta: {self.err}, epsilon: {self.epsilon}')
        return bool_val
    

