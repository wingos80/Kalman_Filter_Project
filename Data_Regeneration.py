########################################################################
# Data preprocessing, recreating the flight path with numerical integration,
# and re-creating the data measruements (IMU, GPS, airdata sensors)
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################
import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
from KF_Functions import *
# import time, sys, os, control.matlab
np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    


########################################################################
## Data I/O managing
########################################################################

# time, phi, theta, psi, u_n, v_n, w_n = None
Wx, Wy, Wz = 2, -8, 1

filename   = 'data/de3211_1.csv'
train_data = genfromtxt(filename, delimiter=',').T
train_data = train_data[:, 1:]

time       = train_data[0]
phi        = train_data[1]*np.pi/180
theta      = train_data[2]*np.pi/180
psi        = train_data[3]*np.pi/180
p          = train_data[4]
q          = train_data[5]
r          = train_data[6]
vtas       = train_data[7]
alpha      = train_data[9]*np.pi/180
beta       = train_data[10]*np.pi/180
ax         = train_data[12]
ay         = train_data[13]
az         = train_data[14]
u_n        = train_data[25]
v_n        = train_data[26]
w_n        = train_data[27]

result_file = open(f"data/F16traindata_CMabV_2023_kf.csv", "w")
result_file.write(f"Cm, alpha_m, beta_m, V_t, alpha_m_kf, beta_m_kf, V_t_kf, alpha_t_kf, C_a_up\n")

########################################################################
## Set up the simulation variables
########################################################################

# Initial state vector, 
#  X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz]^T
#  U = [Ax, Ay, Az, p, q, r]^T
X = np.array([[0],[0],[0], [u_n[0]], [v_n[0]], [w_n[0]], [phi[0]], [theta[0]], [psi[0]], [Wx], [Wy], [Wz]])
U  = np.zeros([6,1])

# Initializing arrays to store the flight path, GPS measurements, and airdata measurements
xyz            = np.zeros([3, len(time)])       # x, y, z
gps_t          = np.zeros([9, len(time)])       # x, y, z, u, v, w, phi, theta, psi
airdata_t      = np.zeros([3, len(time)])       # Vtas, alpha, beta
imu_t          = np.zeros([6, len(time)])       # Ax, Ay, Az, p, q, r

# Storing the GPS and airdata measurements that are already known from the csv filess
gps_t[3:,:]    = np.array([[u_n], [v_n], [w_n], [phi], [theta], [psi]])
airdata_t[:,0] = np.array([[vtas], [alpha], [beta]])


# Running an numerical integration to recreate the flight
for k in range(len(time)-1):
    t_vector     = [time[k], time[k+1]]
    t_vector, X = rk4(kf_calc_f, X, U, t_vector)       # rk4 to integrate state vector to next time step

    # Picking out select states to re-create flight path, GPS measurements, and airdata measurements
    xyz[:, k+1]   = X[0:3,0]
    gps_t[:3,k+1] = X[0:3,0]
    
    X[3:] = np.array([[u_n[k+1]], [v_n[k+1]], [w_n[k+1]], [phi[k+1]], [theta[k+1]], [psi[k+1]], [Wx], [Wy], [Wz]])

# Use 3D scatter plot to visualize the airplane position over time
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Measured', s=1)
plt.title('Reconstructed flight path', fontsize = 18)
plt.xlabel('x (m)', fontsize = 14)
plt.ylabel('y (m)', fontsize = 14)
ax.set_zlabel('z (m)', fontsize = 14)
plt.show()