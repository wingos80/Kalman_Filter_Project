########################################################################
# Python implementation of Iterated Extended Kalman Filter 
# 
#   Author: Wing Chan, adapted from Coen de Visser
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
# Set random seed for reproducibility
np.random.seed(7)

########################################################################
## Data I/O managing
########################################################################

filename = 'data/F16traindata_CMabV_2023.csv'
train_data = genfromtxt(filename, delimiter=',').T

C_m = train_data[0]
Z_k = train_data[1:4]
U_k = train_data[4:]


result_file = open(f"kf2_results.csv", "w")
result_file.write(f"C_a_up\n")


########################################################################
## Set simulation parameters
########################################################################

n               = 4                          # state dimension
nm              = 3                          # output dimension
m               = 3                          # input dimension
dt              = 0.01                       # time step (s)
N               = len(C_m)                   # number of samples
epsilon         = 10**(-10)                  # IEKF threshold
doIEKF          = True                       # If false, EKF without iterations is used
maxIterations   = 200                        # maximum amount of iterations per sample

printfigs       = False                      # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = np.array([[150],[0],[0],[-0.6]])    # initial estimate of optimal value of x_k1_k1

B           = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])               # input matrix
G           = np.zeros((4,4))                     # input noise matrix, why is this 4x4 and not 4x3?

# Initial estimate for state covariance matrix
std_x_0   = 1                # initial standard deviation of state prediction error (for first 3 states)
P_0       = np.array([[std_x_0**2, 0, 0, 0],
                      [0, std_x_0**2, 0, 0],
                      [0, 0, std_x_0**2, 0], 
                      [0, 0, 0, std_x_0**2]])         # initial covariance of state prediction error

# System noise statistics, all noise are white (unbiased and uncorrelated in time)
std_w_u = 1*10**(-3)         # standard deviation of u noise
std_w_v = 1*10**(-3)         # standard deviation of v noise
std_w_w = 1*10**(-3)         # standard deviation of w noise
std_w_C = 0                  # standard deviation of Caup noise

Q         = np.array([[std_w_u**2, 0, 0, 0],
                        [0, std_w_v**2, 0, 0],
                        [0, 0, std_w_w**2, 0], 
                        [0, 0, 0, std_w_C**2]])   # variance of system noise

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_nu_a = 0.035             # standard deviation of alpha noise
std_nu_b = 0.010             # standard deviation of beta noise
std_nu_V = 0.110             # standard deviation of velocity noise

R         = np.array([[std_nu_a**2, 0, 0],
                    [0, std_nu_b**2, 0],
                    [0, 0, std_nu_V**2]])         # variance of system noise

########################################################################
## Initialize Extended Kalman filter
########################################################################

t_k         = 0
t_k1        = dt

# allocate space to store traces
XX_k1_k1    = np.zeros([n, N])
PP_k1_k1    = np.zeros([n, N])
STD_x_cor   = np.zeros([n, N])
STD_z       = np.zeros([nm, N])
ZZ_pred     = np.zeros([nm, N])
IEKFitcount = np.zeros([N, 1])

# initialize state estimation and error covariance matrix
x_k1_k1     = np.array(E_x_0)   # x(0|0) = E(x_0)
P_k1_k1     = np.array(P_0)     # P(0|0) = P(0)


########################################################################
## Run the Kalman filter
########################################################################

tic          = time.time()

# Run the filter through all N samples
for k in range(0, N):
    if k % 100 == 0:
        tonc = time.time()
        print(f'Sample {k} of {N} ({k/N*100:.3f} %), time elapsed: {tonc-tic:.2f} s')
    
    # x(k+1|k) (prediction)
    t, x_k1_k   = rk4(kf_calc_f, x_k1_k1, U_k[:,k], [t_k, t_k1])   # add in U_k vector

    # Calc Jacobians Phi(k+1, k) and Gamma(k+1, k)
    Fx          = kf_calc_Fx(0, x_k1_k, U_k[:,k])
    Hx          = kf_calc_Hx(0, x_k1_k, U_k[:,k])
    # Continuous to discrete time transformation of Df(x,u,t)
    ss_B        = control.matlab.ss(Fx, B, Hx, np.zeros((nm, m)))
    ss_G        = control.matlab.ss(Fx, G, np.zeros((nm, n)), np.zeros((nm, n)))
    Psi         = control.matlab.c2d(ss_B, dt).B
    Phi         = control.matlab.c2d(ss_G, dt).A
    Gamma       = control.matlab.c2d(ss_G, dt).B

    # P(k+1|k) (prediction covariance matrix)
    P_k1_k      = Phi@P_k1_k1@Phi.transpose() + Gamma@Q@Gamma.transpose()
    
    # Run the Iterated Extended Kalman filter
    eta2    = x_k1_k
    err     = 2*epsilon
    itts    = 0
    
    while (err > epsilon):
        if (itts >= maxIterations):
            print(f'Terminating IEKF: exceeded max iterations ({maxIterations})')
            print(f'Delta eta: {err}, epsilon: {epsilon}')
            break
        
        itts    = itts + 1
        eta1    = eta2
            
        # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
        Hx           = kf_calc_Hx(0, eta1, U_k[:,k])
        
        # Observation and observation error predictions
        z_k1_k      = kf_calc_h(0, eta1, U_k[:,k])          # prediction of observation (for validation)   
        P_zz        = Hx@P_k1_k@Hx.transpose() + R          # covariance matrix of observation error (for validation)   

        # Do try in case the covariance matrix is too small
        try:
            std_z       = np.sqrt(P_zz.diagonal())          # standard deviation of observation error (for validation)    
        except:
            std_z       = np.zeros([nm, 1])                 # standard deviation of observation error (for validation)  

        # K(k+1) (gain), Kalman Gain
        K             = P_k1_k@Hx.transpose()@np.linalg.inv(P_zz)
    
        # New observation
        temp = np.reshape(Z_k[:,k], (3,1))                  # Need to reshape this Z array to a column vector
        eta2        = x_k1_k + K@(temp - z_k1_k - Hx@(x_k1_k - eta1))
        err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)  

    IEKFitcount[k]  = itts
    x_k1_k1         = eta2
    
    # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 
    P_k1_k1     = (np.eye(n) - K*Hx)*P_k1_k*(np.eye(n) - K*Hx).transpose() + K*R*K.transpose()    
    std_x_cor   = np.sqrt(P_k1_k1.diagonal())        # standard deviation of state estimation error (for validation)

    # Next step
    t_k         = t_k1 
    t_k1        = t_k1 + dt
    
    # store results, need to flatten the arrays to store in a matrix
    ZZ_pred[:,k]    = z_k1_k.flatten()              # predicted observation
    XX_k1_k1[:,k]   = x_k1_k1.flatten()             # estimated state
    PP_k1_k1[:,k]   = P_k1_k1.diagonal().flatten()  # estimated state covariance (for validation)
    STD_x_cor[:,k]  = std_x_cor.flatten()           # standard deviation of state estimation error (for validation)
    STD_z[:,k]      = std_z.flatten()               # standard deviation of observation error (for validation)

    result_file.write(f"{x_k1_k1[-1,0]}\n")

result_file.close()

toc         = time.time()

# calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k            

print(f'IEKF completed run with {N} samples in {toc-tic:.2f} seconds.')


########################################################################
## Plotting results
########################################################################

x   = np.arange(0, N, 1)
y1  = XX_k1_k1[3,:]         # Estimated C_a_up
fig = plt.figure(figsize=(12,6))
ax  = fig.add_subplot(1, 1 ,1 )
ax.plot(x, y1, label='Estimated C_a_up')
plt.xlim(0,N)
plt.grid(True)
plt.title('Estimated C_a_up')
plt.legend()

#  plot variance of all states
y1  = PP_k1_k1[0,:]         # Estimated variance of u
y2  = PP_k1_k1[1,:]         # Estimated variance of v
y3  = PP_k1_k1[2,:]         # Estimated variance of w
y4  = PP_k1_k1[3,:]         # Estimated variance of C_a_up
fig = plt.figure(figsize=(12,6))
ax  = fig.add_subplot(1, 1 ,1 )
ax.plot(x, y1, label='Estimated variance of u')
ax.plot(x, y2, label='Estimated variance of v')
ax.plot(x, y3, label='Estimated variance of w')
ax.plot(x, y4, label='Estimated variance of C_a_up')
plt.xlim(0,N)
plt.grid(True)
plt.title('Estimated variance of all states')
plt.legend()

# plot number of IEKF iterations at each IEKF step
y1  = IEKFitcount           # Number of IEKF iterations
fig = plt.figure(figsize=(12,6))
ax  = fig.add_subplot(1, 1 ,1 )
ax.plot(x, y1, label='Number of IEKF iterations')
plt.xlim(0,N)
plt.grid(True)
plt.title('Number of IEKF iterations')
plt.legend()

plt.show()


########################################################################
## Reconstructing true alpha
########################################################################

alpha_m     = Z_k[0]                     # Measured alpha
alpha_m_kf  = ZZ_pred[0]              # Predicted alpha from KF

C_a_up      = XX_k1_k1[3,-1]              # Taking the last estimate of C_a_up
alpha_true  = alpha_m/(1+C_a_up)      # Reconstructing true alpha, noise is assumed unbiased thus this estimation of 
                                     # alpha is unbiased as well
alpha_true_kf = alpha_m_kf/(1+C_a_up) # Reconstructing true alpha from KF
# plotting and comparing the true alpha and the reconstructed alpha

y1  = alpha_true                     # True alpha
y2  = alpha_m                        # Measured alpha
fig = plt.figure(figsize=(12,6))
ax  = fig.add_subplot(1, 1 ,1 )
ax.plot(x, y2, label='Measured alpha')
ax.plot(x, y1, label='True alpha')
ax.plot(x, alpha_true_kf, label='True alpha from KF')
plt.xlim(0,N)
plt.grid(True)
plt.title('True alpha and Measured alpha')
plt.legend()

plt.show()