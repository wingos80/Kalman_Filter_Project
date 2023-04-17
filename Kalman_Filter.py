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
E_x_0       = np.zeros([18,1])                                              # initial estimate of x_k1_k1
E_x_0[3:9]  = Z[3:9, 0].reshape(6,1)                                        # initial estimate of velocity and flight angles
E_x_0[9:12] = np.array([[2], [-8], [1]])                                    # initial estimate of Wind velocities
E_x_0[12:]  = np.array([[0.02], [0.02], [0.02], [0.003], [0.003], [0.003]]) # initial estimate of lambda

B           = np.zeros([18,6])               # input matrix

# Initial estimate for covariance matrix
std_x_0 = 1                                     # initial standard deviation of state prediction error
std_x_1 = 1*10**(-3)                            # initial standard deviation of state prediction error
std_x_2 = 1*10**(+1)                            # initial standard deviation of state prediction error
P_stds  = [std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0]

# System noises, all noise are white (unbiased and uncorrelated in time)
std_b_x = 0.02                                  # standard deviation of accelerometer x measurement noise
std_b_y = 0.02                                  # standard deviation of accelerometer y measurement noise
std_b_z = 0.02                                  # standard deviation of accelerometer z measurement noise
std_b_p = 0.003                                 # standard deviation of rate gyro p measurement noise 
std_b_q = 0.003                                 # standard deviation of rate gyro q measurement noise  
std_b_r = 0.003                                 # standard deviation of rate gyro r measurement noise  
Q_stds  = [std_b_x, std_b_y, std_b_z, std_b_p, std_b_q, std_b_r]

u, v, w = E_x_0[3], E_x_0[4], E_x_0[5]
phi, theta, psi = E_x_0[6], E_x_0[7], E_x_0[8]
sin_phi_tan_theta = np.sin(phi) * np.tan(theta)
cos_phi_tan_theta = np.cos(phi) * np.tan(theta)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
sin_phi_div_cos_theta = np.sin(phi) / np.cos(theta)
cos_phi_div_cos_theta = np.cos(phi) / np.cos(theta)
# TEMPORARY TEST FOR THE SYSTEM NOISE MATRIX, G is time variant so i need to make it update every time with the new speeds
G       = np.zeros([18, 6])                      # system noise matrix
G[3:6, 0:3] = -np.eye(3)                                                                        # accelerometer noise (has a negative because the Ax in the model should be Am MINUS bias MINUS noise!!!!)
G[3:9, 3:]  = np.array([[0, w, -v], [-w, 0, u], [v, -u, 0], [1, -sin_phi_tan_theta, -cos_phi_tan_theta], [0, -cos_phi, sin_phi], [0, -sin_phi_div_cos_theta, -cos_phi_div_cos_theta]])  # rate gyro noise

# G       = np.zeros([18, 6])                      # system noise matrix
# G[3:6, 0:3] = -np.eye(3)                                                                        # accelerometer noise (has a negative because the Ax in the model should be Am MINUS bias MINUS noise!!!!)
# G[3:9, 3:]  = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0], [1, 1, 1], [0, 1, -1], [0, 1, 1]])  # rate gyro noise

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

R_stds   = [std_gps_x, std_gps_y, std_gps_z, std_gps_u, std_gps_v, std_gps_w, std_gps_phi, std_gps_theta, std_gps_psi, std_ads_v, std_ads_alpha, std_ads_beta]

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
        # print(f'    Current estimate of system states:\n{kalman_filter.x_k1_k1}\n')
    bing = time.time()
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

    bong = time.time()
#     print(f'    Iteration time: {bong-bing:.5f} s')

                                        
toc = time.time()

print(f'Elapsed time: {toc-tic:.5f} s')


########################################################################
## Saving the results
########################################################################

file_Xs   = filename.split('_measurements.')[0] + '_Xs-kf_measurements.csv'
file_Zs   = filename.split('_measurements.')[0] + '_Zs-kf_measurements.csv'
result_file_Xs = open(file_Xs, "w")
result_file_Zs = open(file_Zs, "w")

# writing the column headings in the resulte file, ADS stands for airdata sensors
result_file_Xs.write(f"x_kf, y_kf, z_kf, u_kf, v_kf, w_kf, phi_kf, theta_kf, psi_kf, Wx_kf, Wy_kf, Wz_kf, Lx_kf, Ly_kf, Lz_kf, Lp_kf, Lq_kf, Lr_kf")
result_file_Zs.write(f"x_gps, y_gps, z_gps, u_gps, v_gps, w_gps, phi_gps, theta_gps, psi_gps, v_ads, alpha_ads, beta_ads")

Xs, Zs = kalman_filter.XX_k1_k1, kalman_filter.ZZ_pred
# writing every entry of the kalman filter states to the result file
for k in range(N):
    result_file_Xs.write(f"{Xs[0,k]}, {Xs[1,k]}, {Xs[2,k]}, {Xs[3,k]}, {Xs[4,k]}, {Xs[5,k]}, {Xs[6,k]}, {Xs[7,k]}, {Xs[8,k]}, {Xs[9,k]}, {Xs[10,k]}, {Xs[11,k]}, {Xs[12,k]}, {Xs[13,k]}, {Xs[14,k]}, {Xs[15,k]}, {Xs[16,k]}, {Xs[17,k]}")
    result_file_Zs.write(f"{Zs[0,k]}, {Zs[1,k]}, {Zs[2,k]}, {Zs[3,k]}, {Zs[4,k]}, {Zs[5,k]}, {Zs[6,k]}, {Zs[7,k]}, {Zs[8,k]}, {Zs[9,k]}, {Zs[10,k]}, {Zs[11,k]}")

    
# Saving the kalman filtered measurements (predicts)
xyz_kf             = Xs[0:3]          # Predicted position from KF
Winds              = Xs[9:12]         # Predicted alpha from KF
Winds_covariances  = kalman_filter.PP_k1_k1[9:12]         # Predicted alpha covariance from KF
new_lambdas        = Xs[12:]          # Predicted lambda from KF
lambda_covariances = kalman_filter.PP_k1_k1[12:]          # Predicted lambda covariance from KF
kf_u               = Zs[3]             # Predicted u from KF
unkf_u             = Z[3]

########################################################################
## Plotting some of the results
########################################################################


x      = dt*np.arange(0, N, 1)
# ys     = {'raw u\'s': [unkf_u, 0.9], 
#           'kf u\'s': [kf_u, 0.9]}
# make_plots(x, [ys], 'raw and kalman-filtered u velocities', 'Time [s]', ['u [m/s]'], printfigs)

print(f' shapes of XX_k1_k1 and ZZ_pred: {Xs.shape}, {kalman_filter.ZZ_pred.shape}')
dX_Z = Xs[:9,:] - kalman_filter.ZZ_pred[:9,:]
ys = {'u from Z': [kalman_filter.ZZ_pred[3,:], 0.3],
      'v from Z': [kalman_filter.ZZ_pred[4,:], 0.3],
      'w from Z': [kalman_filter.ZZ_pred[5,:], 0.3],
      'u from X': [Xs[3,:], 0.9],
      'v from X': [Xs[4,:], 0.9],
      'w from X': [Xs[5,:], 0.9],}
make_plots(x, [ys], 'Difference between predicted and measured states', 'Time [s]', ['Difference [m/s]'], printfigs)

# ys1 = {'Wind x': [Winds[0], 0.9],
#       'Wind y': [Winds[1], 0.9],
#       'Wind z': [Winds[2], 0.9]}
# ys2 = {'Wind x variance': [Winds_covariances[0], 0.9],
#       'Wind y variance': [Winds_covariances[1], 0.9],
#       'Wind z variance': [Winds_covariances[2], 0.9]}

# make_plots(x, [ys1, ys2], 'Wind over time', 'Time [s]', ['Wind [m/s]', 'Wind variance [m2/s2]'], printfigs, log=1)

# ys1 = {'lambda Ax': [new_lambdas[0], 0.9],
#         'lambda Ay': [new_lambdas[1], 0.9],
#         'lambda Az': [new_lambdas[2], 0.9]}
# ys2 = {'lambda Ax variance': [lambda_covariances[0], 0.9],
#         'lambda Ay variance': [lambda_covariances[1], 0.9],
#         'lambda Az variance': [lambda_covariances[2], 0.9]}
# make_plots(x, [ys1, ys2], 'acceleration biases variance over time', 'Time [s]', ['Bias [m/s2]', 'Variance [m2/s4]'], printfigs, log=1)

# ys1 = {'lambda p': [new_lambdas[3], 0.9],
#         'lambda q': [new_lambdas[4], 0.9],
#         'lambda r': [new_lambdas[5], 0.9]}
# ys2 = {'lambda p variance': [lambda_covariances[3], 0.9],
#         'lambda q variance': [lambda_covariances[4], 0.9],
#         'lambda r variance': [lambda_covariances[5], 0.9]}
# make_plots(x, [ys1, ys2], 'angle bias variances over time', 'Time [s]', ['Bias [rad/s]', 'Variance [rad2/s2]'], printfigs, log=1)

ys = {'Iterations taken by IEKF': [kalman_filter.itr_counts, 0.9]}
make_plots(x, [ys], 'Iterations taken by IEKF', 'Time [s]', ['Iterations'], printfigs)
plt.show()


# Use 3D scatter plot to visualize the airplane position over time
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Simulated', s=1)
ax.scatter(xyz_kf[0], xyz_kf[1], xyz_kf[2], c='b', marker='o', label='Kalman filtered (xyz state)', s=1)
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
