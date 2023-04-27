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
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from KF_Functions import *
from Plotter import *

np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    

########################################################################
## Start writing measurements to csv files
########################################################################
show_plot = False            # enable plotting
save      = True             # enable saving data

alpha_t = 1.0
alpha_n = 0.62
colors=['C0', 'C0', 'C1', 'C1', 'C2', 'C2']

# Change the current working directory to the data folder and process all original csv files
os.chdir("data/raw/")
files = os.listdir(os.getcwd())
os.chdir("../..")

bing = time.time()
for filename in files:
    ########################################################################
    ## Data I/O managing
    ########################################################################

    print(f"\n\nRegenerating data for {filename}...\n\n")

    train_data = genfromtxt('data/raw/'+filename, delimiter=',').T
    train_data = train_data[:, 1:]

    # all angles should be in radians
    times      = train_data[0]
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
    u_n        = train_data[25]    # velocities are in the local earth frame (north, East, Down frame!!)
    v_n        = train_data[26]    # velocities are in the local earth frame (north, East, Down frame!!)
    w_n        = train_data[27]    # velocities are in the local earth frame (north, East, Down frame!!)

    if save:
        result_filename = 'data/regenerated/' + filename.replace('.csv','_measurements.csv')
        result_file  = open(result_filename, "w")

        # writing the column headings in the resulte file, ADS stands for airdata sensors
        result_file.write(f"XYZ_x, XYZ_y, XYZ_z, GPS_x, GPS_y, GPS_z, GPS_u, GPS_v, GPS_w, GPS_phi, GPS_theta, GPS_psi, ADS_vel, ADS_alpha, ADS_beta, IMU_Ax, IMU_Ay, IMU_Az, IMU_p, IMU_q, IMU_r, CTRL_da, CTRL_de, CTRL_dr, CTRL_Tc1, CTRL_Tc2, time\n")

    ########################################################################
    ## Setting the parameters for generating the "real" data
    ########################################################################
    # GPS noise statistics
    std_gps_x = 2.5                                 # standard deviation of GPS x position measurement noise
    std_gps_y = std_gps_x                           # standard deviation of GPS y position measurement noise
    std_gps_z = std_gps_x                           # standard deviation of GPS z position measurement noise
    std_gps_u = 0.02                                # standard deviation of GPS u velocity measurement noise
    std_gps_v = std_gps_u                           # standard deviation of GPS v velocity measurement noise
    std_gps_w = std_gps_u                           # standard deviation of GPS w velocity measurement noise
    std_gps_phi = 0.05*np.pi/180                    # standard deviation of GPS phi measurement noise, in radians
    std_gps_theta = std_gps_phi                     # standard deviation of GPS theta measurement noise, in radians
    std_gps_psi = std_gps_phi                       # standard deviation of GPS psi measurement noise, in radians

    stds_gps = [[std_gps_x],[std_gps_y],[std_gps_z],[std_gps_u],[std_gps_v],[std_gps_w],[std_gps_phi],[std_gps_theta],[std_gps_psi]]

    # ADS noise statistics
    std_ads_v = 0.1                                 # standard deviation of air data sensors true airspeed measurement noise
    std_ads_alpha = 0.1*np.pi/180                   # standard deviation of air data sensors alpha measurement noise, in radians
    std_ads_beta = 0.1*np.pi/180                    # standard deviation of air data sensors beta measurement noise, in radians

    stds_ads = [[std_ads_v],[std_ads_alpha],[std_ads_beta]]

    # System noises statistics
    std_b_x = 0.02                                  # standard deviation of accelerometer x measurement noise
    std_b_y = 0.02                                  # standard deviation of accelerometer y measurement noise
    std_b_z = 0.02                                  # standard deviation of accelerometer z measurement noise
    std_b_p = 0.003*np.pi/180                       # standard deviation of rate gyro p measurement noise, in radians 
    std_b_q = 0.003*np.pi/180                       # standard deviation of rate gyro q measurement noise, in radians  
    std_b_r = 0.003*np.pi/180                       # standard deviation of rate gyro r measurement noise, in radians  

    stds_imu  = [[std_b_x], [std_b_y], [std_b_z], [std_b_p], [std_b_q], [std_b_r]]

    # System noise biases
    lambda_imu = np.array(stds_imu)

    ## Measurement noise vectors use to add noise to the data
    # GPS measurement noise vector, GPS = [x, y, z, u, v, w, phi, theta, psi], 9 because len(stds_gps)=9
    noise_gps = np.random.normal(np.zeros((9,1)), stds_gps, (9, len(times)))
    # Air data sensors measurement noise vector, ADS = [Vtas, alpha, beta], 3 because len(stds_ads)=3
    noise_ads = np.random.normal(np.zeros((3,1)), stds_ads, (3, len(times)))
    # IMU measurement noise vector, IMU = [Ax, Ay, Az, p, q, r], 6 because len(stds_imu)=6
    noise_imu = np.random.normal(np.zeros((6,1)), stds_imu, (6, len(times)))

    Wx, Wy, Wz = 2, -8, 1

    #  Adding wind velocities to the 3 velocity components
    u_n = u_n + Wx
    v_n = v_n + Wy
    w_n = w_n + Wz

    ########################################################################
    ## Set up the simulation variables
    ########################################################################
    # Initializing arrays to store the flight path, GPS measurements, and airdata measurements
    xyz            = np.zeros([3, len(times)])       # x, y, z
    gps_t          = np.zeros([9, len(times)])       # x, y, z, u, v, w, phi, theta, psi
    airdata_t      = np.zeros([3, len(times)])       # Vtas, alpha, beta
    imu_t          = np.zeros([6, len(times)])       # Ax, Ay, Az, p, q, r

    # Storing the GPS and airdata measurements that are already known from the csv filess
    gps_t[3:,:]    += np.array([u_n, v_n, w_n, phi, theta, psi]) + noise_gps[3:,:]
    gps_t[:3,:]    += noise_gps[:3,:]
    airdata_t[:,:] += np.array([vtas, alpha, beta]) + noise_ads[:,:]
    imu_t[:,:]     += np.array([ax, ay, az, p, q, r]) + lambda_imu + noise_imu[:,:]

    if save:
        # Storing the initial measurements
        result_file.write(f"{xyz[0,0]}, {xyz[1,0]}, {xyz[2,0]}, {gps_t[0,0]}, {gps_t[1,0]}, {gps_t[2,0]}, {gps_t[3,0]}, {gps_t[4,0]}, {gps_t[5,0]}, {gps_t[6,0]}, {gps_t[7,0]}, {gps_t[8,0]}, {airdata_t[0,0]}, {airdata_t[1,0]}, {airdata_t[2,0]}, {imu_t[0,0]}, {imu_t[1,0]}, {imu_t[2,0]}, {imu_t[3,0]}, {imu_t[4,0]}, {imu_t[5,0]}, {da[0]}, {de[0]}, {dr[0]}, {tc1[0]}, {tc2[0]}, {times[0]}\n")

    # Running numerical integration to recreate the flight positions
    for k in range(len(times)-1):
        # printing out progress in terminal
        if k % 500 == 0:
            print(f"Time step: {k} of {len(times)}")

        # Numerical integration to find positions
        dt = times[k+1] - times[k]
        xyz[:, k+1]   = xyz[:, k] + np.array([dt*u_n[k],dt*v_n[k],dt*w_n[k]])
        gps_t[:3,k+1] += xyz[:, k+1]
        
        if save:
            # Storing the measurements at each time step if desired
            result_file.write(f"{xyz[0,k+1]}, {xyz[1,k+1]}, {xyz[2,k+1]}, {gps_t[0,k+1]}, {gps_t[1,k+1]}, {gps_t[2,k+1]}, {gps_t[3,k+1]}, {gps_t[4,k+1]}, {gps_t[5,k+1]}, {gps_t[6,k+1]}, {gps_t[7,k+1]}, {gps_t[8,k+1]}, {airdata_t[0,k+1]}, {airdata_t[1,k+1]}, {airdata_t[2,k+1]}, {imu_t[0,k+1]}, {imu_t[1,k+1]}, {imu_t[2,k+1]}, {imu_t[3,k+1]}, {imu_t[4,k+1]}, {imu_t[5,k+1]}, {da[k+1]}, {de[k+1]}, {dr[k+1]}, {tc1[k+1]}, {tc2[k+1]}, {times[k+1]}\n")       
            
    bong = time.time()
    print(f'Elapsed time: {round(bong-bing,6)}s')
    ########################################################################
    ## Plotting results, and saving if desired
    ########################################################################
    figs_destination = 'figs/raw_and_noise_figs/' + filename.replace('.csv', '')

    # Use 3D scatter plot to visualize the airplane position over time
    fig = plt.figure(num=f'{figs_destination}_true_vs_noise_xyz.pdf')
    axx = fig.add_subplot(projection='3d')

    axx.scatter(xyz[0,::24], xyz[1,::24], xyz[2,::24], c='r', marker='o', label='True', alpha=0.83, s=0.5)
    axx.scatter(gps_t[0,::15], gps_t[1,::15], gps_t[2,::15], c='b', marker='o', label='Wind + noise', alpha=0.4, s=0.5)
    plt.xlabel('x (m)', fontsize = 14)
    plt.ylabel('y (m)', fontsize = 14)
    axx.set_zlabel('z (m)', fontsize = 14)
    lgnd = plt.legend(scatterpoints=6, fontsize=10)
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{figs_destination}_true_vs_noise_xyz.pdf')
    
    # 2d plots to visualize all system states and values over time
    x = times
    ys = {r'$\phi$ true': [phi, alpha_t],
        r'$\phi$ noise': [gps_t[6,:], alpha_n],
        r'$\theta$ true': [theta, alpha_t],
        r'$\theta$ noise': [gps_t[7,:], alpha_n],
        r'$\psi$ true': [psi, alpha_t],
        r'$\psi$ noise': [gps_t[8,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise angles', r'Time $[s]$', [r'angles $[rad]$'], save=save, colors=colors)

    ys = {'u true': [u_n, alpha_t],
        'u noise': [gps_t[3,:], alpha_n],
        'v true': [v_n, alpha_t],
        'v noise': [gps_t[4,:], alpha_n],
        'w true': [w_n, alpha_t],
        'w noise': [gps_t[5,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise velocities', r'Time $[s]$', [r'velocities $[m/s]$'], save=save, colors=colors)

    ys = {r'$A_x$ true': [ax, alpha_t],
            r'$A_x$ noise': [imu_t[0,:], alpha_n],
            r'$A_y$ true': [ay, alpha_t],
            r'$A_y$ noise': [imu_t[1,:], alpha_n],
            r'$A_z$ true': [az, alpha_t],
            r'$A_z$ noise': [imu_t[2,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise accelerations', r'Time $[s]$', [r'accelerations $[m/s^2]$'], save=save, colors=colors)

    ys = {'p true': [p, alpha_t],
            'p noise': [imu_t[3,:], alpha_n],
            'q true': [q, alpha_t],
            'q noise': [imu_t[4,:], alpha_n],
            'r true': [r, alpha_t],
            'r noise': [imu_t[5,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise angular velocities', r'Time $[s]$', [r'angular velocities $[rad/s]$'], save=save, colors=colors)

    ys = {r'$V_{TAS}$ true': [vtas, alpha_t],
            r'$V_{TAS}$ noise': [airdata_t[0,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise Vtas', r'Time $[s]$', [r'$V_{tas} [rad]$'], save=save, colors=colors)

    ys = {r'$\alpha$ true': [alpha, alpha_t],
            r'$\alpha$ noise': [airdata_t[1,:], alpha_n],
            r'$\beta$ true': [beta, alpha_t],
            r'$\beta$ noise': [airdata_t[2,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise alpha and beta', r'Time $[s]$', [r'angles [rad]'], save=save, colors=colors)
    
    ys = {'x true': [xyz[0,:], alpha_t],
            'x noise': [gps_t[0,:], alpha_n],
            'y true': [xyz[1,:], alpha_t],
            'y noise': [gps_t[1,:], alpha_n],
            'z true': [xyz[2,:], alpha_t],
            'z noise': [gps_t[2,:], alpha_n]}
    make_plots(x, [ys], f'{figs_destination} true vs noise positions', r'Time $[s]$', [r'positions $[m]$'], save=save, colors=colors)

    if show_plot:
        plt.show()

    plt.close('all')

boom = time.time()
print(f'Total time: {round(boom-bing,6)}s')