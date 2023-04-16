########################################################################
# Goes to all the csv files in the data\ directory. And does data preprocessing, 
# recreating the flight path with numerical integration, and re-creating the 
# data measruements (IMU, GPS, airdata sensors)
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################
import numpy as np
import seaborn as sns
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt
from KF_Functions import *
# import time, sys, os, control.matlab
np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    

# Change the current working directory to the data folder and process all original csv files
os.chdir("data/")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".csv") and not filename.endswith("_measurements.csv") :
        
        ########################################################################
        ## Data I/O managing
        ########################################################################

        Wx, Wy, Wz = 2, -8, 1

        print(f"\n\nRegenerating data for {filename}...\n\n")

        # filename   = 'data/de3211_1.csv'
        train_data = genfromtxt(filename, delimiter=',').T
        train_data = train_data[:, 1:]

        times       = train_data[0]
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

        filename   = filename.split('.')[0] + '_measurements.csv'
        result_file = open(filename, "w")

        # ADS stands for airdata sensors
        result_file.write(f"XYZ_x, XYZ_y, XYZ_z, GPS_x, GPS_y, GPS_z, GPS_u, GPS_v, GPS_w, GPS_phi, GPS_theta, GPS_psi, ADS_vel, ADS_alpha, ADS_beta, IMU_Ax, IMU_Ay, IMU_Az, IMU_p, IMU_q, IMU_r\n")

        ########################################################################
        ## Set up the simulation variables
        ########################################################################

        # Initial state vector, 
        #  X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
        #  U = [Ax, Ay, Az, p, q, r]^T
        X = np.array([[0],[0],[0], [u_n[0]], [v_n[0]], [w_n[0]], [phi[0]], [theta[0]], [psi[0]], [Wx], [Wy], [Wz], [0], [0], [0], [0], [0], [0]])
        U = np.zeros([6,1])

        # Initializing arrays to store the flight path, GPS measurements, and airdata measurements
        xyz            = np.zeros([3, len(times)])       # x, y, z
        gps_t          = np.zeros([9, len(times)])       # x, y, z, u, v, w, phi, theta, psi
        airdata_t      = np.zeros([3, len(times)])       # Vtas, alpha, beta
        imu_t          = np.zeros([6, len(times)])       # Ax, Ay, Az, p, q, r

        # Storing the GPS and airdata measurements that are already known from the csv filess
        gps_t[3:,:]    = np.array([u_n, v_n, w_n, phi, theta, psi])
        airdata_t[:,:] = np.array([vtas, alpha, beta])
        imu_t[:,:]     = np.array([ax, ay, az, p, q, r])

        # Storing the initial measurements
        result_file.write(f"{xyz[0,0]}, {xyz[1,0]}, {xyz[2,0]}, {gps_t[0,0]}, {gps_t[1,0]}, {gps_t[2,0]}, {gps_t[3,0]}, {gps_t[4,0]}, {gps_t[5,0]}, {gps_t[6,0]}, {gps_t[7,0]}, {gps_t[8,0]}, {airdata_t[0,0]}, {airdata_t[1,0]}, {airdata_t[2,0]}, {imu_t[0,0]}, {imu_t[1,0]}, {imu_t[2,0]}, {imu_t[3,0]}, {imu_t[4,0]}, {imu_t[5,0]}\n")

        # Running an numerical integration to recreate the flight
        for k in range(len(times)-1):
            if k % 500 == 0:
                print(f"Time step: {k} of {len(times)}")
            t_vector     = [times[k], times[k+1]]
            t_vector, X = rk4(kf_calc_f, X, U, t_vector)       # rk4 to integrate state vector to next time step

            # Picking out select states to re-create flight path, GPS measurements, and airdata measurements
            xyz[:, k+1]   = X[0:3,0]
            gps_t[:3,k+1] = X[0:3,0]
            
            X[3:12] = np.array([[u_n[k+1]], [v_n[k+1]], [w_n[k+1]], [phi[k+1]], [theta[k+1]], [psi[k+1]], [Wx], [Wy], [Wz]])

            # Storing the measurements at each time step
            result_file.write(f"{xyz[0,k+1]}, {xyz[1,k+1]}, {xyz[2,k+1]}, {gps_t[0,k+1]}, {gps_t[1,k+1]}, {gps_t[2,k+1]}, {gps_t[3,k+1]}, {gps_t[4,k+1]}, {gps_t[5,k+1]}, {gps_t[6,k+1]}, {gps_t[7,k+1]}, {gps_t[8,k+1]}, {airdata_t[0,k+1]}, {airdata_t[1,k+1]}, {airdata_t[2,k+1]}, {imu_t[0,k+1]}, {imu_t[1,k+1]}, {imu_t[2,k+1]}, {imu_t[3,k+1]}, {imu_t[4,k+1]}, {imu_t[5,k+1]}\n")
            
        # Use 3D scatter plot to visualize the airplane position over time
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Measured', s=1)
        plt.title(f'{filename}\'s reconstructed flight path', fontsize = 18)
        plt.xlabel('x (m)', fontsize = 14)
        plt.ylabel('y (m)', fontsize = 14)
        ax.set_zlabel('z (m)', fontsize = 14)
        plt.show()