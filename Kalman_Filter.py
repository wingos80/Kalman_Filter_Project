########################################################################
# Python implementation of Iterated Extended Kalman Filter,
# generates a csv file with the filtered data points.
# 
#   Author: Wing Chan, adapted from Coen de Visser
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################
import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *


np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    


########################################################################
## Data I/O managing
########################################################################

filename = 'data/F16traindata_CMabV_2023.csv'
train_data = genfromtxt(filename, delimiter=',').T

C_m        = train_data[0]                    # Measured C_m
Z          = train_data[1:4]                  # Measured alpha, beta, velocity
U          = train_data[4:]                   # Measured u, v, w

alpha_m    = Z[0]                             # Measured alpha
beta_m     = Z[1]                             # Measured beta
V_m        = Z[2]                             # Measured velocity

result_file = open(f"data/F16traindata_CMabV_2023_kf.csv", "w")
result_file.write(f"Cm, alpha_m, beta_m, V_t, alpha_m_kf, beta_m_kf, V_t_kf, alpha_t_kf, C_a_up\n")


########################################################################
## Set simulation parameters
########################################################################

n               = 4                          # state dimension (not used)
nm              = 3                          # measurement dimension
m               = 3                          # input dimension (not used)
dt              = 0.01                       # time step (s)
N               = len(U[0])                  # number of samples
epsilon         = 10**(-10)                  # IEKF threshold
maxIterations   = 200                        # maximum amount of iterations per sample

printfigs       = False                      # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = np.array([[150],[0],[0],[-0.6]])     # initial estimate of optimal value of x_k1_k1

B           = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])               # input matrix
# Initial estimate for covariance matrix
std_x_0   = 1                                     # initial standard deviation of state prediction error
P_stds    = [std_x_0, std_x_0, std_x_0, 1]

# System noises, all noise are white (unbiased and uncorrelated in time)
std_w_u = 1*10**(-3)                              # standard deviation of u noise
std_w_v = 1*10**(-3)                              # standard deviation of v noise
std_w_w = 1*10**(-3)                              # standard deviation of w noise
std_w_C = 0                                       # standard deviation of Caup noise
Q_stds  = [std_w_u, std_w_v, std_w_w, std_w_C]

G       = np.eye(4)                           # system noise matrix

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_nu_a = 0.035             # standard deviation of alpha noise
std_nu_b = 0.010             # standard deviation of beta noise
std_nu_V = 0.110             # standard deviation of velocity noise
R_stds   = [std_nu_a, std_nu_b, std_nu_V]

########################################################################
## Run the Kalman filter
########################################################################

tic           = time.time()

# Initialize the Kalman filter object
kalman_filter = IEKF(N, nm, dt, epsilon, maxIterations)

# Set up the system in the Kalman filter
kalman_filter.setup_system(E_x_0, kf_calc_f, kf_calc_h, kf_calc_Fx, kf_calc_Hx, B, G, rk4)

# Set up the noise in the Kalman filter
kalman_filter.setup_covariances(P_stds, Q_stds, R_stds)

# Run the filter through all N samples
for k in range(N):
    if k % 100 == 0:
        tonc = time.time()
        print(f'Sample {k} of {N} ({k/N*100:.3f} %), time elapsed: {tonc-tic:.2f} s')
        print(f'    Current estimate of C_a_up: {kalman_filter.x_k1_k1[-1,0]:.4f}\n')
    
    # Picking out the k-th entry in the input and measurement vectors
    U_k = U[:,k]  
    Z_k = Z[:,k]

    # Predict and discretize the system
    kalman_filter.predict_and_discretize(U_k)
    
    # Running iterations of the IEKF
    while kalman_filter.not_converged():
        kalman_filter.run_iteration(U_k, Z_k)

    # Once converged, update the state and state covariances estimates
    kalman_filter.update(k)
    

toc = time.time()

print(f'Elapsed time: {toc-tic:.5f} s')

# Saving the kalman filtered measurements (predics)
alpha_m_kf = kalman_filter.ZZ_pred[0]         # Predicted alpha from KF
beta_m_kf  = kalman_filter.ZZ_pred[1]         # Predicted beta from KF
V_m_kf     = kalman_filter.ZZ_pred[2]         # Predicted velocity from KF

########################################################################
## Reconstructing true alpha
########################################################################

C_a_up     = kalman_filter.XX_k1_k1[3,-1]     # Taking the last estimate of C_a_up
alpha_t    = alpha_m/(1+C_a_up)               # Reconstructing true alpha, noise is 
                                              #  assumed unbiased thus this estimation of 
                                              #  alpha is unbiased as well
alpha_t_kf = alpha_m_kf/(1+C_a_up)            # Reconstructing true alpha from KF filtered alpha

# experimenting with using an exponentially weighted moving average to filter alpha_m
rho = 0.95 # Rho value for smoothing

s_prev = 0 # Initial value ewma value


ewma, ewma_bias_corr = np.empty(0), np.empty(0)  # Empty arrays to hold the smoothed data

for i in range(N):
    y = alpha_m[i]
    # Variables to store smoothed data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho* s_prev + (1-rho)*y
    s_cur_bc = s_cur/(1-(rho**(i+1)))

    # Append new smoothed value to array
    ewma = np.append(ewma,s_cur)
    ewma_bias_corr = np.append(ewma_bias_corr,s_cur_bc)

    s_prev = s_cur

# Note to self, EWMA is much easier to implement, does not require any knowledge of the system, has a lot less places where bugs can occur, and also 
# computes much faster than the IEKF (perhaps for other variants of the KF as well). However, if we inspect the output of EWMA, we can see that in some 
# occasions the EWMA fails to predict the correct signs of the signal, which might be extremely costly in some applications. Moreover, there is a time 
# delay associated with the signal computed with EWMA, and EWMA can only smooth out signals, it won't be able to reconstruct some unmeasurable states 
# such as the true alpha or upwash coefficient: but KF's can. Moreover, KF's can be used to estimate the variance of the states, which is not possible 
# with EWMA (but there is a way to build confidence intervals for EWMA, in page 8 of: http://www.wiley.com.tudelft.idm.oclc.org/legacy/wileychi/marketmodels/chapter5.pdf). 

# In conclusion, KF's are more powerful than EWMA, but EWMA is much easier to implement and faster.

C_a_up     = kalman_filter.XX_k1_k1[3,:]      # Upwash coefficient prediction over time

# writing all results to a csv file
for k in range(N):
    C_m_k = C_m[k]
    a_m_k = alpha_m[k]
    b_m_k = beta_m[k]
    V_m_k = V_m[k]
    a_m_kf_k = alpha_m_kf[k]
    b_m_kf_k = beta_m_kf[k]
    V_m_kf_k = V_m_kf[k]
    a_t_kf_k = alpha_t_kf[k]
    C_a_up_k = C_a_up[k]
    result_file.write(f"{C_m[k]}, {a_m_k}, {b_m_k}, {V_m_k}, {a_m_kf_k}, {b_m_kf_k}, {V_m_kf_k}, {a_t_kf_k}, {C_a_up_k}\n")

result_file.close()


if __name__ == "__main__":
    ########################################################################
    ## Plotting results
    ########################################################################


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(alpha_m, beta_m, V_m, c='r', marker='o', label='Measured', s=1)
    plt.title('Measured alpha, beta, V', fontsize = 18)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(alpha_t_kf, beta_m_kf, V_m_kf, c='b', marker='o', label='KF', s=1)
    plt.title('Predicted true alpha, beta, V', fontsize = 18)

    plt.show()


    # Plot the upwash coefficient estimate generated by the kalman filter
    x      = dt*np.arange(0, N, 1)
    ys  = {'Estimated C_a_up' : [C_a_up, 1.0]}

    plotter(x, ys, 'C_a_up estimate evolution', 'Time [s]', 'C_a_up [-]', printfigs)

    # Plotting the true alpha and the reconstructed alpha
    ys  = { 'True alpha'              : [alpha_t    , 0.4],
            'Measured alpha'          : [alpha_m    , 0.4],
            'True alpha from KF'      : [alpha_t_kf , 1.0],
            'Predicted alpha from KF' : [alpha_m_kf , 1.0],
            'ewma alpha'              : [ewma       , 1.0]}

    plotter(x, ys, 'Reconstructed alpha', 'Time [s]', 'alpha [-]', printfigs)

    # Plot variance of all states
    ys  = { 'Estimated variance of u'      : [kalman_filter.PP_k1_k1[0,:], 1.0],
            'Estimated variance of v'      : [kalman_filter.PP_k1_k1[1,:], 1.0],
            'Estimated variance of w'      : [kalman_filter.PP_k1_k1[2,:], 1.0],
            'Estimated variance of C_a_up' : [kalman_filter.PP_k1_k1[3,:], 1.0]}

    plotter(x, ys, 'Variance of states', 'Time [s]', 'Variance [-]', printfigs)

    # Plot number of IEKF iterations at each IEKF step

    ys  = {'Number of IEKF iterations' : [kalman_filter.IEKFitcount, 1.0]}

    plotter(x, ys, 'Number of IEKF iterations', 'Time [s]', 'Number of iterations [-]', printfigs)

    plt.show()