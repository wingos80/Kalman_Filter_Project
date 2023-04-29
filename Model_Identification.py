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
    dt = 0.01

    train_data = genfromtxt("data/regenerated/" + filename.replace("filtered", "measurements"), delimiter=',').T
    train_data = train_data[:, 1:]
    IMU          = train_data[15:21]                 # These are the IMU measurements

    
    train_data = genfromtxt('data/raw/' + filename.split("_f")[0] + ".csv", delimiter=',').T
    train_data = train_data[:, 1:]

    # all angles should be in radians
    p          = train_data[4]
    
    # Calculate the wind vector
    Wind = Winds[:,-25:].mean(axis=1)

    # Calculate the biases
    Bias = Biases[:,-25:].mean(axis=1)
    ########################################################################
    ## Finding the force and moment derivative coefficients
    ########################################################################

    Vs_k     = np.sqrt(Xs_k[3,:]**2 + Xs_k[4,:]**2 + Xs_k[5,:]**2)              # calculating the airspeed from the velocities
    alphas_k = np.arctan2(Xs_k[5,:], Xs_k[3,:])                             # calculating the angle of attack from the velocities
    betas_k  = np.arctan2(Xs_k[4,:], np.sqrt(Xs_k[3,:]**2 + Xs_k[5,:]**2))                                     # calculating the sideslip angle from the velocities

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
    make_plots(x, [ys], f"figs/models/{filename} Angular Accelerations", "Time [s]", [r"Angular Accelerations $[rad/s^2]$"],save=True)
    ys = {r'$A_x$': [vel_rate_dots_X[0,:], 0.8],
          r'$A_y$': [vel_rate_dots_X[1,:], 0.8],
          r'$A_z$': [vel_rate_dots_X[2,:], 0.8]}
    ys2 = {'vx': [Xs_k[3,:], 0.8],
          'vy': [Xs_k[4,:], 0.8],
          'vz': [Xs_k[5,:], 0.8]}
    make_plots(x, [ys, ys2], f"figs/models/{filename} Accelerations and velocities", "Time [s]", [r"Acceleration $[m/s^2]$",r"Flight Velocity $[m/s]$"],save=True)
    ys = {'phi': [Xs_k[6,:], 0.8],
          'theta': [Xs_k[7,:], 0.8],
          'psi': [Xs_k[8,:], 0.8]}
    make_plots(x, [ys], f"figs/models/{filename} Flight Angles", "Time [s]", [r"Flight Angle $[rad]$"],save=True)
    ys = {'Vtas': [Vs_k, 0.8]}
    make_plots(x, [ys], f"figs/models/{filename} Airspeed", "Time [s]", [r"Airspeed $[m/s]$"],save=True)
    ys = {'alpha': [alphas_k, 0.8],
          'beta': [betas_k, 0.8]}
    make_plots(x, [ys], f"figs/models/{filename} Alpha and Beta", "Time [s]", [r"Angle$[rad]$"],save=True)

    # ys = {r'IMU A_x': [IMU[0,:], 0.8],
    #       r'IMU A_y': [IMU[1,:], 0.8],
    #       r'IMU A_z': [IMU[2,:], 0.8]}
    # make_plots(x, [ys], "IMU Accelerations", "Time [s]", [r"Acceleration $[m/s^2]$"])
    # ys = {r'phi': [Xs_k[6,:], 0.8],
    #       r'theta': [Xs_k[7,:], 0.8],
    #       r'psi': [Xs_k[8,:], 0.8]}
    # make_plots(x, [ys], "Flight Angles", "Time [s]", [r"Flight Angle $[rad]$"])
    # plt.show()
    
    # do a quick scatter plot of the FC[2] s a function of alpha and de
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(Vs_k[::3], alphas_k[::3], betas_k[::3], c='r', marker='o')
    ax.set_xlabel('Vtas')
    ax.set_ylabel('alpha')
    ax.set_zlabel('beta')

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(Vs_k[::3], alphas_k[::3], FCs_k[2,::3], c='r', marker='o')
    ax.set_xlabel('Vtas')
    ax.set_ylabel('alpha')
    ax.set_zlabel('FC[2]')
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
