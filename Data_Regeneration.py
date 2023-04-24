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


########################################################################
## Start writing measurements to csv files
########################################################################
plot = False

# Change the current working directory to the data folder and process all original csv files
os.chdir("data/")
for filename in os.listdir(os.getcwd()):
    # Only process the raw csv files that are not measurements files
    if filename.endswith(".csv") and not filename.endswith("_measurements.csv") :
        
        ########################################################################
        ## Data I/O managing
        ########################################################################

        Wx, Wy, Wz = 2, -8, 1

        print(f"\n\nRegenerating data for {filename}...\n\n")

        train_data = genfromtxt(filename, delimiter=',').T
        train_data = train_data[:, 1:]

        # all angles should be in radians
        times       = train_data[0]
        phi        = train_data[1]
        theta      = train_data[2]
        psi        = train_data[3]
        p          = train_data[4]
        q          = train_data[5]
        r          = train_data[6]
        vtas       = train_data[7]
        alpha      = train_data[9]
        beta       = train_data[10]
        ax         = train_data[12]
        ay         = train_data[13]
        az         = train_data[14]
        da         = train_data[15]
        de         = train_data[16]
        dr         = train_data[17]
        tc1        = train_data[23]
        tc2        = train_data[24]
        u_n        = train_data[25]
        v_n        = train_data[26]
        w_n        = train_data[27]

        filename   = filename.split('.')[0] + '_measurements.csv'
        result_file = open(filename, "w")

        # writing the column headings in the resulte file, ADS stands for airdata sensors
        result_file.write(f"XYZ_x, XYZ_y, XYZ_z, GPS_x, GPS_y, GPS_z, GPS_u, GPS_v, GPS_w, GPS_phi, GPS_theta, GPS_psi, ADS_vel, ADS_alpha, ADS_beta, IMU_Ax, IMU_Ay, IMU_Az, IMU_p, IMU_q, IMU_r, CTRL_da, CTRL_de, CTRL_dr, CTRL_Tc1, CTRL_Tc2, time\n")

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
        result_file.write(f"{xyz[0,0]}, {xyz[1,0]}, {xyz[2,0]}, {gps_t[0,0]}, {gps_t[1,0]}, {gps_t[2,0]}, {gps_t[3,0]}, {gps_t[4,0]}, {gps_t[5,0]}, {gps_t[6,0]}, {gps_t[7,0]}, {gps_t[8,0]}, {airdata_t[0,0]}, {airdata_t[1,0]}, {airdata_t[2,0]}, {imu_t[0,0]}, {imu_t[1,0]}, {imu_t[2,0]}, {imu_t[3,0]}, {imu_t[4,0]}, {imu_t[5,0]}, {da[0]}, {de[0]}, {dr[0]}, {tc1[0]}, {tc2[0]}, {times[0]}\n")

        # Running an numerical integration to recreate the flight
        for k in range(len(times)-1):
            if k % 500 == 0:
                print(f"Time step: {k} of {len(times)}")
            dt = times[k+1] - times[k]
            xyz[:, k+1]   = xyz[:, k] + np.array([dt*u_n[k],dt*v_n[k],dt*w_n[k]])
            gps_t[:3,k+1] = xyz[:, k+1]

            # Storing the measurements at each time step
            result_file.write(f"{xyz[0,k+1]}, {xyz[1,k+1]}, {xyz[2,k+1]}, {gps_t[0,k+1]}, {gps_t[1,k+1]}, {gps_t[2,k+1]}, {gps_t[3,k+1]}, {gps_t[4,k+1]}, {gps_t[5,k+1]}, {gps_t[6,k+1]}, {gps_t[7,k+1]}, {gps_t[8,k+1]}, {airdata_t[0,k+1]}, {airdata_t[1,k+1]}, {airdata_t[2,k+1]}, {imu_t[0,k+1]}, {imu_t[1,k+1]}, {imu_t[2,k+1]}, {imu_t[3,k+1]}, {imu_t[4,k+1]}, {imu_t[5,k+1]}, {da[k+1]}, {de[k+1]}, {dr[k+1]}, {tc1[k+1]}, {tc2[k+1]}, {times[k+1]}\n")       

        if plot:
            # Use 3D scatter plot to visualize the airplane position over time
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='normal integrate', s=1)
            plt.title(f'{filename}\'s reconstructed flight path', fontsize = 18)
            plt.xlabel('x (m)', fontsize = 14)
            plt.ylabel('y (m)', fontsize = 14)
            ax.set_zlabel('z (m)', fontsize = 14)
            plt.show()