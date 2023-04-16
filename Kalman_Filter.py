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

filename = 'data/da3211_2_measurements.csv'
train_data = genfromtxt(filename, delimiter=',').T
train_data = train_data[:, 1:]

xyz        = train_data[0:3]                   # First 3 columns are the simulated xyz's 
Z          = train_data[3:15]                  # First 9 columns are the gps measurements, last 3 are the airdata sensor measurements
U          = train_data[-6:]                   # These are the IMU measurements

# result_file = open(f"data/F16traindata_CMabV_2023_kf.csv", "w")
# result_file.write(f"Cm, alpha_m, beta_m, V_t, alpha_m_kf, beta_m_kf, V_t_kf, alpha_t_kf, C_a_up\n")

########################################################################
## Set simulation parameters
########################################################################

n               = 18                          # state dimension (not used)
nm              = 12                          # measurement dimension
m               = 6                          # input dimension (not used)
dt              = 0.01                       # time step (s)
N               = len(U[0])                  # number of samples
epsilon         = 10**(-10)                  # IEKF threshold
maxIterations   = 200                        # maximum amount of iterations per sample

printfigs       = False                      # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

########################################################################
## Set initial values for states and statistics
## X : numpy.ndarray (n,1)
##     state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
## 
## U : numpy.ndarray (m,1)
##     input vector, U = [Ax, Ay, Az, p, q, r]^T
########################################################################
E_x_0       = np.zeros([18,1])                                              # initial estimate of optimal value of x_k1_k1
E_x_0[3:9]  = Z[3:9, 0].reshape(6,1)                                        # initial estimate of velocity and flight angles
E_x_0[9:12] = np.array([[2], [-8], [1]])                                    # initial estimate of Wind velocities
E_x_0[12:]  = np.array([[0.02], [0.02], [0.02], [0.003], [0.003], [0.003]]) # initial estimate of lambda

B           = np.zeros([18,6])               # input matrix, TODO: check if this is correct

# Initial estimate for covariance matrix
std_x_0 = 1.1                                    # initial standard deviation of state prediction error
std_x_1 = 1*10**(-3)                            # initial standard deviation of state prediction error
std_x_2 = 1*10**(+1)                            # initial standard deviation of state prediction error
P_stds  = [std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2, std_x_0**2]

# System noises, all noise are white (unbiased and uncorrelated in time)
std_b_x = 0.02                                  # standard deviation of accelerometer x measurement noise
std_b_y = 0.02                                  # standard deviation of accelerometer y measurement noise
std_b_z = 0.02                                  # standard deviation of accelerometer z measurement noise
std_b_p = 0.003                                 # standard deviation of rate gyro p measurement noise 
std_b_q = 0.003                                 # standard deviation of rate gyro q measurement noise  
std_b_r = 0.003                                 # standard deviation of rate gyro r measurement noise  
Q_stds  = [std_b_x**2, std_b_y**2, std_b_z**2, std_b_p**2, std_b_q**2, std_b_r**2]

G       = np.zeros([18, 6])                      # system noise matrix
G[3:6, 0:3] = np.eye(3)                                                                   # accelerometer noise
G[3:6, 3:]  = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])                              # rate gyro noise
G[6:9, 3:]  = np.array([[1, 1, 1], [0, 1, -1], [0, 1, 1]])                                # rate gyro noise

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_gps_x = 2.5                                 # standard deviation of GPS x position measurement noise
std_gps_y = std_gps_x                           # standard deviation of GPS y position measurement noise
std_gps_z = std_gps_x                           # standard deviation of GPS z position measurement noise
std_gps_u = 0.02                                # standard deviation of GPS u velocity measurement noise
std_gps_v = std_gps_u                           # standard deviation of GPS v velocity measurement noise
std_gps_w = std_gps_u                           # standard deviation of GPS w velocity measurement noise
std_gps_phi = 0.05                              # standard deviation of GPS phi measurement noise
std_gps_theta = std_gps_phi                     # standard deviation of GPS theta measurement noise
std_gps_psi = std_gps_phi                       # standard deviation of GPS psi measurement noise

std_ads_v = 0.1                                 # standard deviation of air data sensors true airspeed measurement noise
std_ads_alpha = 0.1                             # standard deviation of air data sensors alpha measurement noise
std_ads_beta = 0.1                              # standard deviation of air data sensors beta measurement noise

R_stds   = [std_gps_x**2, std_gps_y**2, std_gps_z**2, std_gps_u**2, std_gps_v**2, std_gps_w**2, std_gps_phi**2, std_gps_theta**2, std_gps_psi**2, std_ads_v**2, std_ads_alpha**2, std_ads_beta**2]

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
        print(f'    Current estimate of system states:\n{kalman_filter.x_k1_k1}\n')
    
    # Picking out the k-th entry in the input and measurement vectors
    U_k = U[:,k]  
    Z_k = Z[:,k]

    # Predict and discretize the system
    kalman_filter.predict_and_discretize(U_k)
    
    # Running iterations of the IEKF
    while kalman_filter.not_converged():
        kalman_filter.run_iteration(U_k, Z_k, k)

    # Once converged, update the state and state covariances estimates
    kalman_filter.update(k)
    

toc = time.time()

print(f'Elapsed time: {toc-tic:.5f} s')

########################################################################
## Plotting some of the results
########################################################################

# Saving the kalman filtered measurements (predicts)
xyz_kf             = kalman_filter.XX_k1_k1[0:3]          # Predicted position from KF
Winds              = kalman_filter.XX_k1_k1[9:12]         # Predicted alpha from KF
Winds_covariances  = kalman_filter.PP_k1_k1[9:12]         # Predicted alpha covariance from KF
new_lambdas        = kalman_filter.XX_k1_k1[12:]          # Predicted lambda from KF
lambda_covariances = kalman_filter.PP_k1_k1[12:]          # Predicted lambda covariance from KF
kf_u               = kalman_filter.ZZ_pred[3]             # Predicted u from KF
unkf_u             = Z[3]


x      = dt*np.arange(0, N, 1)
ys     = {'raw u\'s': [unkf_u, 0.9], 
          'kf u\'s': [kf_u, 0.9]}
make_plots(x, [ys], 'raw and kalman-filtered u velocities', 'Time [s]', ['u [m/s]'], printfigs)

ys1 = {'Wind x': [Winds[0], 0.9],
      'Wind y': [Winds[1], 0.9],
      'Wind z': [Winds[2], 0.9]}
ys2 = {'Wind x variance': [Winds_covariances[0], 0.9],
      'Wind y variance': [Winds_covariances[1], 0.9],
      'Wind z variance': [Winds_covariances[2], 0.9]}

make_plots(x, [ys1, ys2], 'Wind over time', 'Time [s]', ['Wind [m/s]', 'Wind variance [m2/s2]'], printfigs, log=1)

ys1 = {'lambda Ax': [new_lambdas[0], 0.9],
        'lambda Ay': [new_lambdas[1], 0.9],
        'lambda Az': [new_lambdas[2], 0.9]}
ys2 = {'lambda Ax variance': [lambda_covariances[0], 0.9],
        'lambda Ay variance': [lambda_covariances[1], 0.9],
        'lambda Az variance': [lambda_covariances[2], 0.9]}
make_plots(x, [ys1, ys2], 'acceleration biases variance over time', 'Time [s]', ['Bias [m/s2]', 'Variance [m2/s4]'], printfigs, log=1)

ys1 = {'lambda p': [new_lambdas[3], 0.9],
        'lambda q': [new_lambdas[4], 0.9],
        'lambda r': [new_lambdas[5], 0.9]}
ys2 = {'lambda p variance': [lambda_covariances[3], 0.9],
        'lambda q variance': [lambda_covariances[4], 0.9],
        'lambda r variance': [lambda_covariances[5], 0.9]}
make_plots(x, [ys1, ys2], 'angle bias variances over time', 'Time [s]', ['Bias [rad/s]', 'Variance [rad2/s2]'], printfigs, log=1)

ys = {'Iterations taken by IEKF': [kalman_filter.itr_counts, 0.9]}
make_plots(x, [ys], 'Iterations taken by IEKF', 'Time [s]', ['Iterations'], printfigs)
plt.show()


# Use 3D scatter plot to visualize the airplane position over time
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Simulated', s=1)
ax.scatter(xyz_kf[0], xyz_kf[1], xyz_kf[2], c='b', marker='o', label='Kalman filtered', s=1)
plt.title(f'{filename}\'s reconstructed flight path', fontsize = 18)
plt.xlabel('x (m)', fontsize = 14)
plt.ylabel('y (m)', fontsize = 14)
ax.set_zlabel('z (m)', fontsize = 14)
ax.legend()
plt.show()

# # writing all results to a csv file
# for k in range(N):
#     result_file.write(f"{C_m[k]}, {a_m_k}, {b_m_k}, {V_m_k}, {a_m_kf_k}, {b_m_kf_k}, {V_m_kf_k}, {a_t_kf_k}, {C_a_up_k}\n")

# result_file.close()
