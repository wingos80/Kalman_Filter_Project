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

filename = 'data/dr3211_1_measurements.csv'
# filename = 'data/dr3211_1.csv'
train_data = genfromtxt(filename, delimiter=',').T
train_data = train_data[:, 1:]

xyz        = train_data[0:3]                   # First 3 columns are the simulated xyz's 
Z          = train_data[3:15]                  # First 9 columns are the gps measurements, last 3 are the airdata sensor measurements
U          = train_data[-6:]                   # These are the IMU measurements

########################################################################
## Set simulation parameters
########################################################################

n               = 18                         # state dimension (not used)
nm              = 12                         # measurement dimension
m               = 6                          # input dimension (not used)
dt              = 0.01                       # time step [s]
num_samples     = len(U[0])                  # number of samples
epsilon         = 10**(-12)                  # IEKF threshold
maxIterations   = 500                        # maximum amount of iterations per sample

printfigs       = True                       # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

# aircraft parameters
Ixx = 11187.8       # [kgm^2]
Iyy = 22854.8       # [kgm^2]
Izz = 31974.8       # [kgm^2]
Ixz = 1930.1        # [kgm^2]

mass = 4500         # [kg]
b = 13.3250         # [m]
S = 24.99           # [m^2]
c = 1.991           # [m]


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
# E_x_0[12:]  = np.array([[0.02], [0.02], [0.02], [0.003*np.pi/180], [0.003*np.pi/180], [0.003*np.pi/180]]) # initial estimate of lambda (biases), angular biases in radians
E_x_0[12:]  = np.array([[0.152251], [-0.0078], [0.051224], [-0.00053], [-0.00015], [-0.00263]]) # initial estimate of lambda (biases), angular biases in radians

B           = np.zeros([18,6])               # input matrix

# Initial estimate for covariance matrix
std_x_0 = 0.0001                                   # initial standard deviation of state prediction error
std_x_1 = 1*10**(-3)                            # initial standard deviation of state prediction error
std_x_2 = 1*10**(+1)                            # initial standard deviation of state prediction error
state_estimate_stds  = [std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_1, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0, std_x_0]

# System noises, all noise are white (unbiased and uncorrelated in time)
std_b_x = 0.02                                  # standard deviation of accelerometer x measurement noise
std_b_y = 0.02                                  # standard deviation of accelerometer y measurement noise
std_b_z = 0.02                                  # standard deviation of accelerometer z measurement noise
std_b_p = 0.003*np.pi/180                       # standard deviation of rate gyro p measurement noise, in radians 
std_b_q = 0.003*np.pi/180                       # standard deviation of rate gyro q measurement noise, in radians  
std_b_r = 0.003*np.pi/180                       # standard deviation of rate gyro r measurement noise, in radians  
system_noise_std  = [std_b_x, std_b_y, std_b_z, std_b_p, std_b_q, std_b_r]

G       = np.zeros([18, 6])                      # system noise matrix
G[3:6, 0:3] = -np.eye(3)                                                                        # accelerometer noise (has a negative because the Ax in the model should be Am MINUS bias MINUS noise!!!!)
G[3:9, 3:]  = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0], [1, 1, 1], [0, 1, -1], [0, 1, 1]])  # rate gyro noise

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_gps_x = 2.5                                 # standard deviation of GPS x position measurement noise
std_gps_y = std_gps_x                           # standard deviation of GPS y position measurement noise
std_gps_z = std_gps_x                           # standard deviation of GPS z position measurement noise
std_gps_u = 0.02                                # standard deviation of GPS u velocity measurement noise
std_gps_v = std_gps_u                           # standard deviation of GPS v velocity measurement noise
std_gps_w = std_gps_u                           # standard deviation of GPS w velocity measurement noise
std_gps_phi = 0.05*np.pi/180                    # standard deviation of GPS phi measurement noise, in radians
std_gps_theta = std_gps_phi                     # standard deviation of GPS theta measurement noise, in radians
std_gps_psi = std_gps_phi                       # standard deviation of GPS psi measurement noise, in radians

std_ads_v = 0.1                                 # standard deviation of air data sensors true airspeed measurement noise
std_ads_alpha = 2.0*np.pi/180                   # standard deviation of air data sensors alpha measurement noise, in radians
std_ads_beta = 3.5*np.pi/180                    # standard deviation of air data sensors beta measurement noise, in radians

measurement_noise_stds   = [std_gps_x, std_gps_y, std_gps_z, std_gps_u, std_gps_v, std_gps_w, std_gps_phi, std_gps_theta, std_gps_psi, std_ads_v, std_ads_alpha, std_ads_beta]

########################################################################
## Run the Kalman filter
########################################################################
tic           = time.time()

# Initialize the Kalman filter object
kalman_filter = IEKF(N=num_samples, nm=nm, dt=dt, epsilon=epsilon, maxIterations=maxIterations)

# Set up the system in the Kalman filter
kalman_filter.setup_system(x_0=E_x_0, f=kf_calc_f, h=kf_calc_h, Fx=kf_calc_Fx, Hx=kf_calc_Hx, B=B, G=G, integrator=rk4)

# Set up the noise in the Kalman filter
kalman_filter.setup_covariances(P_stds=state_estimate_stds, Q_stds=system_noise_std, R_stds=measurement_noise_stds)

# Run the filter through all N samples
for k in range(num_samples):
    if k % 100 == 0:
        tonc = time.time()
        print(f'Sample {k} of {num_samples} ({k/num_samples*100:.3f} %), time elapsed: {tonc-tic:.2f} s')
        print(f'    Current estimate of system states:\n{kalman_filter.x_k1_k1}\n')
    bing = time.time()
    # Picking out the k-th entry in the input and measurement vectors
    U_k = U[:,k]  
    Z_k = Z[:,k]

    # Predict and discretize the system
    kalman_filter.predict_and_discretize(U_k)
    
    # Running iterations of the IEKF
    while kalman_filter.not_converged(k):
        kalman_filter.run_one_iteration(U_k, Z_k, k)

    # Once converged, update the state and state covariances estimates
    kalman_filter.update(U_k, k)

    bong = time.time()
    
toc = time.time()

print(f'Elapsed time: {toc-tic:.5f} s')

########################################################################
## Saving the results into csv files
########################################################################
file_Xs   = filename.split('_measurements.')[0] + '_filtered_measurements.csv'
result_file_Xs = open(file_Xs, "w")

# writing the column headings in the results file
result_file_Xs.write(f"x_kf, y_kf, z_kf, u_kf, v_kf, w_kf, phi_kf, theta_kf, psi_kf, Wx_kf, Wy_kf, Wz_kf, Lx_kf, Ly_kf, Lz_kf, Lp_kf, Lq_kf, Lr_kf\n")
Xs, Zs = kalman_filter.XX_k1_k1, kalman_filter.ZZ_pred

# Note, the second deriviatives of the angles will only be calculated for the values from the indices [2:-2] of the angle arrays!!!
rate_dots_X = np.zeros_like(Xs[6:9])                                                     # angular accceleations in body frame [rad/s^2]

_, rate_dots_X[:, 2:-2] = kf_finite_difference(dt, Xs[6:9])                                      # second derivative of the flight angles, from finite difference
rate_dots_X[:, :2] = rate_dots_X[:, 2].reshape(3,1)                                           # just making the first two angular accelerations to be constant (ZOH)
rate_dots_X[:, -2:] = rate_dots_X[:, -3].reshape(3,1)                                         # just making the last two angular accelerations to be constant (ZOH)

# writing every entry of the kalman filter states to the result file
for k in range(num_samples):
    result_file_Xs.write(f"{Xs[0,k]}, {Xs[1,k]}, {Xs[2,k]}, {Xs[3,k]}, {Xs[4,k]}, {Xs[5,k]}, {Xs[6,k]}, {Xs[7,k]}, {Xs[8,k]}, {Xs[9,k]}, {Xs[10,k]}, {Xs[11,k]}, {Xs[12,k]}, {Xs[13,k]}, {Xs[14,k]}, {Xs[15,k]}, {Xs[16,k]}, {Xs[17,k]}\n")

# Saving the kalman filtered predictions
Winds              = Xs[9:12]                             # Predicted alpha from KF
Winds_covariances  = kalman_filter.PP_k1_k1[9:12]         # Predicted alpha covariance from KF
new_lambdas        = Xs[12:]                              # Predicted lambda from KF
lambda_covariances = kalman_filter.PP_k1_k1[12:]          # Predicted lambda covariance from KF

########################################################################
## Plotting all the filtered data and the kalman estimated values
########################################################################
colors=['C0','C1','C2','C0','C1','C2']
file = filename.split('_m')[0].replace('data/', '')
x      = dt*np.arange(0, num_samples, 1)
ys = {r'raw x': [Z[0], 0.3],
      r'raw y': [Z[1], 0.3],
      r'raw z': [Z[2], 0.3],
      r'KF x':  [Xs[0], 1.0],
      r'KF y':  [Xs[1], 1.0],
      r'KF z':  [Xs[2], 1.0]}
ys2 = {r'$\sigma^2(x)$': [kalman_filter.PP_k1_k1[0], 1.0],
       r'$\sigma^2(y)$': [kalman_filter.PP_k1_k1[1], 1.0],
       r'$\sigma^2(z)$': [kalman_filter.PP_k1_k1[2], 1.0]}
make_plots(x, [ys, ys2], f'{file} raw and kalman-filtered positions', r'$Time [s]$', [r'$x [m]$', r'$x^2 [m^2]$'], colors=colors, save=printfigs, log=1)

ys     = {r'raw u': [Z[3], 0.3],
          r'raw v': [Z[4], 0.3],
          r'raw w': [Z[5], 0.3],
          r'kf u':  [Zs[3], 1.0],
          r'kf v':  [Zs[4], 1.0],
          r'kf w':  [Zs[5], 1.0],}
ys2 = {r'$\sigma^2(u)$': [kalman_filter.PP_k1_k1[3], 1.0],
       r'$\sigma^2(v)$': [kalman_filter.PP_k1_k1[4], 1.0],
       r'$\sigma^2(w)$': [kalman_filter.PP_k1_k1[5], 1.0]}
make_plots(x, [ys, ys2], f'{file} raw and kalman-filtered velocities', r'$Time [s]$', [r'$u [m/s]$', r'$u^2 [m^2/s^2]$'], colors=colors, save=printfigs, log=1)

ys      = {r'raw $\phi$':   [Z[6], 0.3],
           r'raw $\theta$': [Z[7], 0.3],
           r'raw $\psi$':   [Z[8], 0.3], 
           r'kf $\phi$':    [Xs[6], 1.0],
           r'kf $\theta$':  [Xs[7], 1.0],
           r'kf $\psi$':    [Xs[8], 1.0]}
ys2     = {r'$\sigma^2(\phi) $': [kalman_filter.PP_k1_k1[6], 1.0],
           r'$\sigma^2(\theta2)$': [kalman_filter.PP_k1_k1[7], 1.0],
           r'$\sigma^2(\psi) $': [kalman_filter.PP_k1_k1[8], 1.0]}
make_plots(x, [ys, ys2], f'{file} raw and kalman-filtered angles', r'$Time [s]$', [r'$\phi [rad]$', r'$\phi^2 [rad^2]$'], colors=colors, save=printfigs, log=1)

ys1 = {r'$W_x$': [Winds[0], 1.0],
      r'$W_y$': [Winds[1], 1.0],
      r'$W_z$': [Winds[2], 1.0]}
ys2 = {r'$\sigma^2(W_x)$': [Winds_covariances[0], 1.0],
      r'$\sigma^2(W_y)$': [Winds_covariances[1], 1.0],
      r'$\sigma^2(W_z)$': [Winds_covariances[2], 1.0]}

make_plots(x, [ys1, ys2], f'{file} Wind over time', r'$Time [s]$', [r'$Wind [m/s]$', r'$Variance [m^2/s^2]$'], save=printfigs, log=1)

ys1 = {r'$\lambda_{x_r}$': [new_lambdas[0], 1.0],
        r'$\lambda_{y_r}$': [new_lambdas[1], 1.0],
        r'$\lambda_{z_r}$': [new_lambdas[2], 1.0]}
ys2 = {r'$\sigma^2(\lambda_{x_r})$': [lambda_covariances[0], 1.0],
        r'$\sigma^2(\lambda_{y_r})$': [lambda_covariances[1], 1.0],
        r'$\sigma^2(\lambda_{z_r})$': [lambda_covariances[2], 1.0]}
make_plots(x, [ys1, ys2], f'{file} acceleration biases variance over time', r'$Time [s]$', [r'$Acceleration Bias [m/s^2]$', r'$Variance [m^2/s^4]$'], save=printfigs, log=1)

ys1 = {r'$\lambda_{p_r}$': [new_lambdas[3], 1.0],
        r'$\lambda_{q_r}$': [new_lambdas[4], 1.0],
        r'$\lambda_{r_r}$': [new_lambdas[5], 1.0]}
ys2 = {r'$\sigma^2(\lambda_{p_r})$': [lambda_covariances[3], 1.0],
        r'$\sigma^2(\lambda_{q_r})$': [lambda_covariances[4], 1.0],
        r'$\sigma^2(\lambda_{r_r})$': [lambda_covariances[5], 1.0]}
make_plots(x, [ys1, ys2], f'{file} angle bias variances over time', r'$Time [s]$', [r'$Anuglar Rate Bias [rad/s]$', r'$Variance [rad^2/s^2]$'], save=printfigs, log=1)

# ys = {'IMU input 1': [U[0, :], 0.7],
#       'IMU input 2': [U[1, :], 0.7],
#       'IMU input 3': [U[2, :], 0.7]}
# make_plots(x, [ys], f'{file} IMU inputs', r'$Time [s]$', [r'$IMU input$'], save=False)

# ys = {'IMU input 4': [U[3, :], 0.7],
#       'IMU input 5': [U[4, :], 0.7],  
#       'IMU input 6': [U[5, :], 0.7]}  
# make_plots(x, [ys], f'{file} IMU inputs', r'$Time [s]$', [r'$IMU input$'], save=False)

ys = {'Iterations taken by IEKF': [kalman_filter.itr_counts, 1.0]}
make_plots(x, [ys], f'{file} Iterations taken by IEKF', r'$Time [s]$', [r'$Iterations$'], save=printfigs)

ys = {'innovation': [kalman_filter.innovations[0], 1.0]}
make_plots(x, [ys], f'{file} innovation', r'$Time [s]$', [r'$Innovation$'], save=printfigs)

plt.clf()
# plt.show()

# # Use 3D scatter plot to visualize the airplane position over time
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Simulated', s=1)
# plt.title(f'{filename}\'s reconstructed flight path', fontsize = 18)
# plt.xlabel('x (m)', fontsize = 14)
# plt.ylabel('y (m)', fontsize = 14)
# ax.set_zlabel('z (m)', fontsize = 14)
# ax.legend()
# plt.show()
