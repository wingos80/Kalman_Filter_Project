# time, phi, theta, psi, p, q, r, vtas, Mach, alpha, beta, gamma, Ax, Ay, Az, da, de, dr, dta, dte, dtr, flaps, gear, Tc1, Tc2, u_n, v_n, w_n = 1
# "time", "phi", "theta", "psi", "p", "q", "r", "vtas", "Mach", "alpha", "beta", "gamma", "Ax", "Ay", "Az", "da", "de", "dr", "dta", "dte", "dtr", "flaps", "gear", "Tc1", "Tc2", "u_n", "v_n", "w_n"
########################################################################
# Data preprocessing, trying to regenerate all positions, and GPS, IMU, airdata measurements 
# for the given flight
# 
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
u_n        = train_data[25]
v_n        = train_data[26]
w_n        = train_data[27]

result_file = open(f"data/F16traindata_CMabV_2023_kf.csv", "w")
result_file.write(f"Cm, alpha_m, beta_m, V_t, alpha_m_kf, beta_m_kf, V_t_kf, alpha_t_kf, C_a_up\n")

# Initial state vector, 12x1
X0 = np.array([[0],[0],[0], [u_n[0]], [v_n[0]], [w_n[0]], [phi[0]], [theta[0]], [psi[0]], [Wx], [Wy], [Wz]])
X1 = np.zeros_like(X0)
U  = np.zeros([6,1])

xyz = np.zeros([3, len(time)])
for k in range(len(time)-1):
    t_vector = [time[k], time[k+1]]
    t, X1 = rk4(kf_calc_f, X0, U, t_vector)

    #  only pick out the new xyz states and store them
    xyz[:, k+1] = X1[0:3,0]
    
    X1[3:] = np.array([[u_n[k+1]], [v_n[k+1]], [w_n[k+1]], [phi[k+1]], [theta[k+1]], [psi[k+1]], [Wx], [Wy], [Wz]])
    X0 = X1

#  use 3D scatter plot to visualize the xyz positions over time
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Measured', s=1)
plt.title('Reconstructed flight path', fontsize = 18)
plt.xlabel('x (m)', fontsize = 14)
plt.ylabel('y (m)', fontsize = 14)
ax.set_zlabel('z (m)', fontsize = 14)
plt.show()