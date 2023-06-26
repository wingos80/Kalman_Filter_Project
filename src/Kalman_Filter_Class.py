import numpy as np
import time, sys, os, control.matlab
from src.KF_Functions import *

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


    def setup_system(self, x_0, f, h, Fx, Fu, Hx, B, G, integrator):
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

        # self.G       = G                  # system noise matrix

        # system dynamics and outputs
        self.f       = f                  # system dynamics
        self.h       = h                  # output equation
        self.Fx      = Fx                 # Jacobian of system dynamics wrt states
        self.Fu      = Fu                 # Jacobian of system dynamics wrt inputs
        self.Hx      = Hx                 # Jacobian of output equation

        # saving the integation scheme chosen
        self.integrator = integrator

        if B.shape[0] != self.n:
            raise ValueError(f'B matrix dimension 0 must be the same size as the state vector (B.shape[0] = {B.shape[0]}, self.n = {self.n})')
        if G.shape[0] != self.n:
            raise ValueError(f'G matrix dimension 0 must be the same size as the state vector (G.shape[0] = {G.shape[0]}, self.n = {self.n})')
        
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
        self.innovations = np.zeros([1, self.N])  # memory for filter measurement innovations

        self.itr_counts = np.zeros([self.N])           # memory for IEKF iteration count
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

        if len(P_stds) != self.n:
            raise ValueError(f'Number of state estimate stds must match number of states (len(P_stds) = {len(P_stds)}, self.n = {self.n})')
        if len(Q_stds) != self.m:
            raise ValueError(f'Number of system noise stds must match number of system noise signals (len(Q_stds) = {len(Q_stds)}, self.m = {self.m})')
        if len(R_stds) != self.nm:
            raise ValueError(f'Number of measurement noise stds must match number of measurements (len(R_stds) = {len(R_stds)}, self.nm = {self.nm})')


    def predict_and_discretize(self, U_k):
        """
        Predict the next state and discretize the system dynamics and output equations
        Parameters
        ----------
        U_k : np.array
            Input vector for the k-th time step
        """

        # x(k+1|k) (prediction)
        self.t, self.x_k1_k   = self.integrator(self.f, self.x_k1_k1, U_k, [self.t_k, self.t_k1])   # add in U_k vector

        # TEMPORARY TEST FOR THE SYSTEM NOISE MATRIX, G is time variant so i need to make it update every time with the new speeds
        
        u, v, w = self.x_k1_k[3,0], self.x_k1_k[4,0], self.x_k1_k[5,0]
        phi, theta = self.x_k1_k[6,0], self.x_k1_k[7,0]
        G       = np.zeros([self.n, U_k.size])                      # system noise matrix
        G[3:6, 0:3] = -np.eye(3)                                    # accelerometer noise (has a negative because the Ax in the model should be Am MINUS bias MINUS noise!!!!)
        G[3:9, 3:]  = np.array([[0, w, -v], 
                                [-w, 0, u], 
                                [v, -u, 0], 
                                [-1, -np.sin(phi) * np.tan(theta), -np.cos(phi) * np.tan(theta)], 
                                [0, -np.cos(phi), np.sin(phi)], 
                                [0, -np.sin(phi) / np.cos(theta), -np.cos(phi) / np.cos(theta)]], dtype=object)  # rate gyro noise

        # Calc Jacobians, Phi(k+1, k), and Gamma(k+1, k)
        Fx_jacobian  = self.Fx(0, self.x_k1_k, U_k)
        Fw_jacobian  = G

        # discretizing the state transition and input matrices according to how C.C. de Visser does it in his PhD thesis
        Phi = c2d(Fx_jacobian, self.dt)
        Gamma = c2d(Fx_jacobian, self.dt, n_plus=1)@Fw_jacobian

        # ss_G        = control.matlab.ss(Fx_jacobian, G, np.zeros((self.nm, self.n)), np.zeros((self.nm, G.shape[1])))  # state space model with A and G matrices, to identify phi and gamma matrices
        # Phi         = control.matlab.c2d(ss_G, self.dt).A
        # Gamma       = control.matlab.c2d(ss_G, self.dt).B

        # P(k+1|k) (prediction covariance matrix)
        self.P_k1_k = Phi@self.P_k1_k1@Phi.transpose() + Gamma@self.Q@Gamma.transpose()
        self.eta2   = self.x_k1_k
        self.F_jacobian = Fx_jacobian


    def check_obs_rank(self, Hx, Fx):
        """
        Check the observability of the state
        Parameters
        ----------
        Hx : np.array
            Jacobian of the output equation
        Fx : np.array
            Jacobian of the system dynamics
        """
        rank = np.zeros([self.nm*self.n, self.n])
        rank[0:self.nm, :] = Hx
        for i in range(1, self.n):
            rank[i*self.nm:(i+1)*self.nm, :] = rank[(i-1)*self.nm:i*self.nm, :]@Fx

        #  code to test if i have constructed the obsvervability matrix correctly
        # rank2 = control.matlab.obsv(Fx, Hx)
        # test = rank-rank2
        rankHF = np.linalg.matrix_rank(rank)
        return rankHF


    def run_one_iteration(self, U_k, Z_k, k):
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
        eta1 = self.eta2
        # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
        H_j  = self.Hx(0, eta1, U_k)
        
        # Check observability of state
        if (k == 0 and self.itr == 1):
            F_jacobian  = self.F_jacobian
            rankHF  = self.check_obs_rank(H_j, F_jacobian);
            if (rankHF < self.n):
                print('\n\n\n\n**********************WARNING**********************\n\n')
                print(f'The current states are not observable; rank of Observability Matrix is {rankHF}, should be {self.n}\n')

        residual = Z_k.reshape(-1,1) - self.h(0, eta1, U_k)

        # Calculate the Kalman Gain
        res_covariance = H_j@self.P_k1_k@H_j.T + self.R
        self.std_z  = np.sqrt(res_covariance.diagonal())                        # standard deviation of observation error (for validation)
        Kalman_Gain = self.P_k1_k@H_j.T@np.linalg.inv(res_covariance)

        # Update the state estimation
        eta2 = self.x_k1_k + Kalman_Gain@(residual - H_j@(self.x_k1_k - eta1))
        self.err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)       # difference in updated state estimate and previous state estimate
        
        self.H_jacobian  = H_j
        self.Kalman_Gain = Kalman_Gain
        self.eta2 = eta2
        self.residual = residual


    def update(self, U_k, k):
        """
        Update the state and state covariance estimates, and store the results
        Parameters
        ----------
        k : int
            Current iteration step
        U_k: np.array
            Input vector for the k-th time step
        """

        # Updating the state estimate
        self.x_k1_k1 = self.eta2
        self.x_k1_k1_dot = self.f(0, self.x_k1_k1, U_k) # for validation, im 
        self.z_k1_k1 = self.h(0, self.eta2, U_k) 
        
        # Making some local variables for readability
        K   = self.Kalman_Gain
        H_j = self.H_jacobian

        # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 
        # self.P_k1_k1 = (self.eye_n - K@H_j)@self.P_k1_k
        self.P_k1_k1     = (self.eye_n - K@H_j)@self.P_k1_k@(self.eye_n - K@H_j).transpose() + K@self.R@K.transpose()    
        self.std_x_cor   = np.sqrt(self.P_k1_k1.diagonal())        # standard deviation of state estimation error (for validation)

        # # calculate the kalman filter 'innovation', difference in measured and predicted observation
        innovation = np.mean(self.residual)

        # Store results, need to flatten the arrays to store in a matrix
        self.ZZ_pred[:,k]    = self.z_k1_k1.flatten()              # predicted observation
        self.XX_k1_k1[:,k]   = self.x_k1_k1.flatten()             # estimated state
        self.PP_k1_k1[:,k]   = self.P_k1_k1.diagonal().flatten()  # estimated state covariance (for validation)
        self.STD_x_cor[:,k]  = self.std_x_cor.flatten()           # standard deviation of state estimation error (for validation)
        self.STD_z[:,k]      = self.std_z.flatten()               # standard deviation of observation error (for validation)
        self.innovations[:,k] = innovation                        # kalman filter 'innovation'

        # Update to next time step
        self.t_k           = self.t_k1 
        self.t_k1          = self.t_k1 + self.dt
        self.itr_counts[k] = self.itr
        self.itr = 0
        self.err = 2*self.epsilon


    def not_converged(self, k):
        """
        Check if the IEKF has converged
        Parameters
        ----------
        k : int
            Current iteration step
        """
        bool_val = self.err > self.epsilon and self.itr < self.max_itr
        if self.itr >= self.max_itr:
            print(f'\nMaximum number of iterations reached at step = {k}')
            print(f'Delta eta: {self.err}, epsilon: {self.epsilon}\n')
        return bool_val
    

