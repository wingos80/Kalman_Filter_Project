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

# Change the current working directory to the data folder and process all original csv files
filename = 'data/all_results.csv'
train_data = genfromtxt(filename, delimiter=',').T
train_data = train_data[:, 1:]

Xs        = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
U         = train_data[18:23,:]                # retriving da,de,dr,tc1,tc2
Vs        = train_data[23,:]                   # retreiving the airspeed
alphas    = train_data[24,:]                   # retreiving the alpha
betas     = train_data[25,:]                   # retreiving the beta
dt = 0.01

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

Vs = np.sqrt(Xs[3]**2 + Xs[4]**2 + Xs[5]**2) # [m/s]
########################################################################
## Finding the force and moment derivative coefficients
########################################################################

# finding the aircraft angular accelerations by calculating the second derivatives of the angles using finite difference
ang_rate_dots_X, ang_accel_dots_X = kf_finite_difference(dt, Xs[6:9])           # finding derivative of the flight angles

# finding the aircraft accelerations by calculating the first derivatives of the velocities using finite difference
vel_rate_dots_X, _ = kf_finite_difference(dt, Xs[3:6])                          # finding derivative of the flight velocities

FCs = kf_calc_Fc(mass, rho, S, Vs, vel_rate_dots_X)                             # calculating the Fc values
MCs = kf_calc_Mc(rho, b, c, S, I, Vs, ang_rate_dots_X, ang_accel_dots_X)        # calculating the Mc values

print(f'shape of FCs: {FCs.shape}\nshape of vel_rate_dots_X: {vel_rate_dots_X.shape}\nshape of Xs[3:6]: {Xs[3:6].shape}\nshape of MCs: {MCs.shape}\nshape of ang_rate_dots_X: {ang_rate_dots_X.shape}\nshape of ang_accel_dots_X: {ang_accel_dots_X.shape}\nshape of Xs[6:9]: {Xs[6:9].shape}')

# do a quick scatter plot of the FC[2] s a function of alpha and de
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(train_data[18,:])
# ax.scatter(alphas, betas, MCs[2], c='r', marker='o')
ax.scatter(alphas, Vs, MCs[1])
plt.show()
# def kf_calc_Mc(rho, b, c, S, I, Vs, rates, accs):