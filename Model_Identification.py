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
## Set aircraft parameters
########################################################################

# aircraft parameters
Ixx = 11187.8       # [kgm^2]
Iyy = 22854.8       # [kgm^2]
Izz = 31974.8       # [kgm^2]
Ixz = 1930.1        # [kgm^2]
I = [Ixx, Iyy, Izz, Ixz]

mass = 4500         # [kg]
rho = 1.225         # [kg/m^3]
b = 13.3250         # [m]
S = 24.99           # [m^2]
c = 1.991           # [m]

FCs    = None
MCs    = None
Xs     = None
U      = None
Vs     = None
alphas = None
betas  = None

os.chdir("data/filtered/a/")
files = os.listdir(os.getcwd())
os.chdir("../../..")

for filename in files:
    print(f"\n\nProcessing data for {filename}...\n\n")

    ########################################################################
    ## Data I/O managing
    ########################################################################

    # Change the current working directory to the data folder and process all original csv files
    train_data = genfromtxt("data/filtered/a/" + filename, delimiter=',').T
    train_data = train_data[:, 1:]

    Xs_k       = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
    Winds      = train_data[9:12,:]                 # retreiving wind velocities
    Biases     = train_data[12:18,:]                # retreiving accelerometer and gyro biases
    U_k        = train_data[18:23,:]                # retriving da,de,dr,tc1,tc2
    Vs_k       = train_data[23,:]                   # retreiving the airspeed
    alphas_k   = train_data[24,:]                   # retreiving the alpha
    betas_k    = train_data[25,:]                   # retreiving the beta
    dt = 0.01

    train_data = genfromtxt("data/regenerated/" + filename.replace("filtered", "measurements"), delimiter=',').T
    train_data = train_data[:, 1:]
    IMU          = train_data[15:21]                 # These are the IMU measurements
    # Calculate the wind vector
    Wind = Winds[:,-25:].mean(axis=1)
    print(f"Wind vector: {Wind}")

    # Calculate the biases
    Bias = Biases[:,-25:].mean(axis=1)
    print(f"Bias vector: {Bias}")
    ########################################################################
    ## Finding the force and moment derivative coefficients
    ########################################################################

    # finding the aircraft angular accelerations by calculating the second derivatives of the angles using finite difference
    ang_rate_dots_X, ang_accel_dots_X = kf_finite_difference(dt, Xs_k[6:9])           # finding derivative of the flight angles

    # finding the aircraft accelerations by calculating the first derivatives of the velocities using finite difference
    vel_rate_dots_X, _ = kf_finite_difference(dt, Xs_k[3:6])                          # finding derivative of the flight velocities

    FCs_k = kf_calc_Fc(mass, rho, S, Vs_k, vel_rate_dots_X)                             # calculating the Fc values
    MCs_k = kf_calc_Mc(rho, b, c, S, I, Vs_k, ang_rate_dots_X, ang_accel_dots_X)        # calculating the Mc values
    
    x = dt*np.arange(0, Xs_k.shape[1], 1)
    ys = {r'$\dot{p}$': [ang_accel_dots_X[0,:], 0.8],
          r'$\dot{q}$': [ang_accel_dots_X[1,:], 0.8],
          r'$\dot{r}$': [ang_accel_dots_X[2,:], 0.8]}
    # ys = {r'$A_x$': [vel_rate_dots_X[0,:], 0.8],
    #       r'$A_y$': [vel_rate_dots_X[1,:], 0.8],
    #       r'$A_z$': [vel_rate_dots_X[2,:], 0.8]}
    make_plots(x, [ys], "Angular Accelerations", "Time [s]", [r"Angular Acceleration $[rad/s^2]$"])
    # make_plots(x, [ys], "Accelerations", "Time [s]", [r"Acceleration $[m/s^2]$"])
    # ys = {r'IMU A_x': [IMU[0,:], 0.8],
    #       r'IMU A_y': [IMU[1,:], 0.8],
    #       r'IMU A_z': [IMU[2,:], 0.8]}
    # make_plots(x, [ys], "IMU Accelerations", "Time [s]", [r"Acceleration $[m/s^2]$"])
    # ys = {r'phi': [Xs_k[6,:], 0.8],
    #       r'theta': [Xs_k[7,:], 0.8],
    #       r'psi': [Xs_k[8,:], 0.8]}
    # make_plots(x, [ys], "Flight Angles", "Time [s]", [r"Flight Angle $[rad]$"])
    plt.show()
    if FCs is None:
        FCs, MCs, Xs, U, Vs, alphas, betas = FCs_k, MCs_k, Xs_k, U_k, Vs_k, alphas_k, betas_k
    else:
        FCs = np.concatenate((FCs, FCs_k), axis=1)
        MCs = np.concatenate((MCs, MCs_k), axis=1)
        Xs = np.concatenate((Xs, Xs_k), axis=1)
        U = np.concatenate((U, U_k), axis=1)
        Vs = np.concatenate((Vs, Vs_k), axis=0)
        alphas = np.concatenate((alphas, alphas_k), axis=0)
        betas = np.concatenate((betas, betas_k), axis=0)

# do a quick scatter plot of the FC[2] s a function of alpha and de
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(alphas[::3], betas[::3], Vs[::3], c='r', marker='o')
ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('V')
# ax.scatter(betas[::3], Vs[::3], MCs[2,::3], c='r', marker='o')
# ax.set_xlabel('beta')
# ax.set_ylabel('V')
# ax.set_zlabel('yawing moment coefficient')
# ax.scatter(alphas[::3], Vs[::3], MCs[1,::3])
# ax.set_xlabel('alpha')
# ax.set_ylabel('V')
# ax.set_zlabel('pitching moment coefficient')
plt.show()
# def kf_calc_Mc(rho, b, c, S, I, Vs, rates, accs):