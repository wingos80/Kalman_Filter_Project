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
## Start writing measurements to csv files
########################################################################
save      = False             # enable saving data
show_plot = True            # enable plotting
printfigs = False             # enable saving figures

all_results_file = open("data/all_results.csv", "w")
all_results_file.write(f"x_kf, y_kf, z_kf, u_kf, v_kf, w_kf, phi_kf, theta_kf, psi_kf, Wx_kf, Wy_kf, Wz_kf, Lx_kf, Ly_kf, Lz_kf, Lp_kf, Lq_kf, Lr_kf, da, de, dr, Tc1, Tc2, V, alpha, beta\n")

# Change the current working directory to the data folder and process all original csv files
os.chdir("data/regenerated/noise/")
files = os.listdir(os.getcwd())
os.chdir("../../..")

time1 = time.time()
files = ['de3211_1_measurements.csv']
for filename in files:
    print(f"\n\nFiltering data for {filename}...\n\n")
    ########################################################################
    ## Data I/O managing
    ########################################################################

    # filename = 'data/de3211_1_measurements.csv'
    train_data = genfromtxt('data/regenerated/noise/' + filename, delimiter=',').T
    train_data = train_data[:, 1:]

    xyz        = train_data[0:3]                   # First 3 columns are the simulated xyz's 
    Z          = train_data[3:15]                  # First 9 columns are the gps measurements, last 3 are the airdata sensor measurements
    U          = train_data[15:21]                 # These are the IMU measurements
    CTRLs      = train_data[21:]                   # These are the control inputs


    train_data = genfromtxt('data/raw/' + filename.split('_m')[0] + '.csv', delimiter=',').T
    train_data = train_data[:, 1:]
    vtas = train_data[7]
    alphas = train_data[9]
    betas = train_data[10]
    ########################################################################
    ## Set simulation parameters
    ########################################################################

    n               = 18                         # state dimension (not used)
    nm              = 12                         # measurement dimension
    m               = 6                          # input dimension (not used)
    dt              = 0.01                       # time step [s]
    num_samples     = len(U[0])                  # number of samples
    epsilon         = 10**(-17)                  # IEKF threshold
    maxIterations   = 600                        # maximum amount of iterations per sample

    ########################################################################
    ## Set initial values for states and statistics
    ## X : numpy.ndarray (n,1)
    ##     state vector, X = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz, lambdax, lambday, lambdaz, lambdap, lambdaq, lambdar]^T
    ## U : numpy.ndarray (m,1)
    ##     input vector, U = [Ax, Ay, Az, p, q, r]^T
    ########################################################################
    E_x_0       = np.zeros([18,1])                                                                            # initial estimate of x_k1_k1
    E_x_0[3:9]  = Z[3:9, 0].reshape(6,1)                                                                      # initial estimate of velocity and flight angles
    E_x_0[9:12] = np.array([[2], [-8], [1]])                                                                  # initial estimate of Wind velocities
    E_x_0[12:]  = np.array([[0.02], [0.02], [0.02], [0.003*np.pi/180], [0.003*np.pi/180], [0.003*np.pi/180]]) # initial estimate of lambda (biases), angular biases in radians
    B           = np.zeros([18,6])                                                                            # input matrix

    # Initial state standard deviation estimates
    P_stds  = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 2, 8, 1, 0.002, 0.002, 0.002, 0.0003*np.pi/180, 0.0003*np.pi/180, 0.0003*np.pi/180]
    
    
    # System noises, all noise are white (unbiased and uncorrelated in time)
    std_b_x = 0.02                                  # standard deviation of accelerometer x measurement noise
    std_b_y = 0.02                                  # standard deviation of accelerometer y measurement noise
    std_b_z = 0.02                                  # standard deviation of accelerometer z measurement noise
    std_b_p = 0.003*np.pi/180                       # standard deviation of rate gyro p measurement noise, in radians 
    std_b_q = 0.003*np.pi/180                       # standard deviation of rate gyro q measurement noise, in radians  
    std_b_r = 0.003*np.pi/180                       # standard deviation of rate gyro r measurement noise, in radians  

    # system noise estimates (noise of the input IMU signals)
    Q_stds  = [std_b_x, std_b_y, std_b_z, std_b_p, std_b_q, std_b_r]

    # system noise input matrix, ACTUALLY NOT USED, input matrix needs to be set dynamically inside the kalman filter class
    G           = np.zeros([18, 6])                                # system noise matrix
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
    std_ads_alpha = 0.1*np.pi/180                   # standard deviation of air data sensors alpha measurement noise, in radians
    std_ads_beta = 0.1*np.pi/180                    # standard deviation of air data sensors beta measurement noise, in radians

    # measurement noise estimates (noise of the Z signal)
    R_stds   = [std_gps_x, std_gps_y, std_gps_z, std_gps_u, std_gps_v, std_gps_w, std_gps_phi, std_gps_theta, std_gps_psi, std_ads_v, std_ads_alpha, std_ads_beta]

    ########################################################################
    ## Run the Kalman filter
    ########################################################################
    tic           = time.time()

    # Initialize the Kalman filter object
    kalman_filter = IEKF(N=num_samples, nm=nm, dt=dt, epsilon=epsilon, maxIterations=maxIterations)

    # Set up the system in the Kalman filter
    kalman_filter.setup_system(x_0=E_x_0, f=kf_calc_f, h=kf_calc_h, Fx=kf_calc_Fx, Fu=kf_calc_Fu, Hx=kf_calc_Hx, B=B, G=G, integrator=rk4)

    # Set up the noise in the Kalman filter
    kalman_filter.setup_covariances(P_stds=P_stds, Q_stds=Q_stds, R_stds=R_stds)

    # Run the filter through all N samples
    for k in range(num_samples):
        # Print progress
        if k % 100 == 0:
            tonc = time.time()
            print(f'{filename}: Sample {k} of {num_samples} ({k/num_samples*100:.3f} %), time elapsed: {tonc-tic:.2f} s')
            print(f'    Current estimate of system states:\n{kalman_filter.x_k1_k1}\n')
        bing = time.time()
        
        # Picking out the k-th entry in the input and measurement vectors
        # if k == num_samples-1:
        #     U_k = U[:,k]
        # else:
        #     U_k = (U[:,k] + U[:,k+1])/2
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

    print(f'\nElapsed time: {toc-tic:.5f} s\n')
    
    ########################################################################
    ## Saving the results into csv files
    ########################################################################

    Xs, Zs          = kalman_filter.XX_k1_k1, kalman_filter.ZZ_pred
    result_filename = filename.replace('measurements', 'filtered')

    if save:
        result_file = open('data/filtered/normal/' + result_filename, "w")

        # writing the column headings in the results file
        result_file.write(f"x_kf, y_kf, z_kf, u_kf, v_kf, w_kf, phi_kf, theta_kf, psi_kf, Wx_kf, Wy_kf, Wz_kf, Lx_kf, Ly_kf, Lz_kf, Lp_kf, Lq_kf, Lr_kf, da, de, dr, Tc1, Tc2, V, alpha, beta\n")

        # writing every entry of the kalman filter states to the result file
        for k in range(num_samples):
            result_file.write(f"{Xs[0,k]},{Xs[1,k]},{Xs[2,k]},{Xs[3,k]},{Xs[4,k]},{Xs[5,k]},{Xs[6,k]},{Xs[7,k]},{Xs[8,k]},{Xs[9,k]},{Xs[10,k]},{Xs[11,k]},{Xs[12,k]},{Xs[13,k]},{Xs[14,k]},{Xs[15,k]},{Xs[16,k]},{Xs[17,k]},{CTRLs[0,k]},{CTRLs[1,k]},{CTRLs[2,k]},{CTRLs[3,k]},{CTRLs[4,k]},{Z[9,k]},{Z[10,k]},{Z[11,k]}\n")
            all_results_file.write(f"{Xs[0,k]},{Xs[1,k]},{Xs[2,k]},{Xs[3,k]},{Xs[4,k]},{Xs[5,k]},{Xs[6,k]},{Xs[7,k]},{Xs[8,k]},{Xs[9,k]},{Xs[10,k]},{Xs[11,k]},{Xs[12,k]},{Xs[13,k]},{Xs[14,k]},{Xs[15,k]},{Xs[16,k]},{Xs[17,k]},{CTRLs[0,k]},{CTRLs[1,k]},{CTRLs[2,k]},{CTRLs[3,k]},{CTRLs[4,k]},{Z[9,k]},{Z[10,k]},{Z[11,k]}\n")
        
        result_file.close()

    ########################################################################
    ## Plotting all the filtered data and the kalman estimated values
    ########################################################################
    
    figs_destination = 'figs/filtered_figs/normal/' + filename.split('_m')[0]

    # Saving the kalman filtered predictions
    Winds              = Xs[9:12]                             # Predicted alpha from KF
    Winds_covariances  = kalman_filter.PP_k1_k1[9:12]         # Predicted alpha covariance from KF
    new_lambdas        = Xs[12:]                              # Predicted lambda from KF
    lambda_covariances = kalman_filter.PP_k1_k1[12:]          # Predicted lambda covariance from KF

    #  some plotting parameters that are useful to save
    colors=['C0','C1','C2','C0','C1','C2']

    # Use 3D scatter plot to visualize the airplane position over time
    fig = plt.figure(num=f'{figs_destination} reconstructed flight path')
    ax = fig.add_subplot(projection='3d')

    ax.scatter(Z[0, ::23], Z[1, ::23], Z[2, ::23], c='r', marker='o', label='Noisey', alpha=0.3, s=1)
    ax.scatter(Xs[0, ::23], Xs[1, ::23], Xs[2, ::23], c='b', marker='o', label='Filtered', alpha=0.4, s=1)
    ax.scatter(xyz[0, ::23], xyz[1, ::23], xyz[2, ::23], c='g', marker='o', label='True', alpha=0.4, s=1)
    plt.xlabel('x (m)', fontsize = 14)
    plt.ylabel('y (m)', fontsize = 14)
    ax.set_zlabel('z (m)', fontsize = 14)
    lgnd = plt.legend(scatterpoints=6, fontsize=10)
    plt.tight_layout()

    if printfigs:
        print(f"\n***************************************\nSaving figures to location: {figs_destination}\n***************************************\n")
        plt.savefig(f'{figs_destination}_xyz.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(Z[9,::25], Z[10,::25], Z[11,::25], label='regenerated', alpha=0.3)
    print(Zs[11])
    ax.scatter(Zs[9,::15], Zs[10,::15], Zs[11,::15], label='predicted')
    ax.scatter(vtas[::10], alphas[::10], betas[::10], label='true', alpha=0.3)
    ax.set_xlabel('v [m/s]')
    ax.set_ylabel('alpha [rad]')
    ax.set_zlabel('beta [rad]')
    # ax.set_zlim(-0.0006, 0.0006)
    ax.legend()
    if printfigs:
        print(f"\n***************************************\nSaving figures to location: {figs_destination}\n***************************************\n")
        plt.savefig(f'{figs_destination}_vtas_alpha_beta.pdf')

    # plt.show()
    x      = dt*np.arange(0, num_samples, 1)

    ys     = {r'raw u': [Z[3], 0.3],
            r'raw v': [Z[4], 0.3],
            r'raw w': [Z[5], 0.3],
            r'kf u':  [Zs[3], 1.0],
            r'kf v':  [Zs[4], 1.0],
            r'kf w':  [Zs[5], 1.0],}
    make_plots(x, [ys], f'{figs_destination} raw and kalman-filtered navigation velocities', r'Time $[s]$', [r'body velocities $[m/s]$'], save=printfigs, colors=colors)

    ys = {r'raw x': [Z[0], 0.3],
        r'raw y': [Z[1], 0.3],
        r'raw z': [Z[2], 0.3],
        r'KF x':  [Xs[0], 1.0],
        r'KF y':  [Xs[1], 1.0],
        r'KF z':  [Xs[2], 1.0]}
    ys2 = {r'$\sigma^2(x)$': [kalman_filter.PP_k1_k1[0], 1.0],
        r'$\sigma^2(y)$': [kalman_filter.PP_k1_k1[1], 1.0],
        r'$\sigma^2(z)$': [kalman_filter.PP_k1_k1[2], 1.0]}
    make_plots(x, [ys, ys2], f'{figs_destination} raw and kalman-filtered positions', r'Time $[s]$', [r'$x [m]$', r'$x^2 [m^2]$'], colors=colors, save=printfigs, log=1)

    ys     = {r'body u':  [Xs[3], 1.0],
            r'body v':  [Xs[4], 1.0],
            r'body w':  [Xs[5], 1.0],}
    ys2 = {r'$\sigma^2(u)$': [kalman_filter.PP_k1_k1[3], 1.0],
        r'$\sigma^2(v)$': [kalman_filter.PP_k1_k1[4], 1.0],
        r'$\sigma^2(w)$': [kalman_filter.PP_k1_k1[5], 1.0]}
    make_plots(x, [ys, ys2], f'{figs_destination} estimated body velocities', r'Time $[s]$', [r'$u [m/s]$', r'$u^2 [m^2/s^2]$'], colors=colors, save=printfigs, log=1)

    ys      = {r'raw $\phi$':   [Z[6], 0.3],
            r'raw $\theta$': [Z[7], 0.3],
            r'raw $\psi$':   [Z[8], 0.3], 
            r'kf $\phi$':    [Xs[6], 1.0],
            r'kf $\theta$':  [Xs[7], 1.0],
            r'kf $\psi$':    [Xs[8], 1.0]}
    ys2     = {r'$\sigma^2(\phi) $': [kalman_filter.PP_k1_k1[6], 1.0],
            r'$\sigma^2(\theta2)$': [kalman_filter.PP_k1_k1[7], 1.0],
            r'$\sigma^2(\psi) $': [kalman_filter.PP_k1_k1[8], 1.0]}
    make_plots(x, [ys, ys2], f'{figs_destination} raw and kalman-filtered angles', r'Time $[s]$', [r'$\phi [rad]$', r'$\phi^2 [rad^2]$'], colors=colors, save=printfigs, log=1)

    ys1 = {r'$W_x$': [Winds[0], 1.0],
        r'$W_y$': [Winds[1], 1.0],
        r'$W_z$': [Winds[2], 1.0]}
    ys2 = {r'$\sigma^2(W_x)$': [Winds_covariances[0], 1.0],
        r'$\sigma^2(W_y)$': [Winds_covariances[1], 1.0],
        r'$\sigma^2(W_z)$': [Winds_covariances[2], 1.0]}
    make_plots(x, [ys1, ys2], f'{figs_destination} Wind over time', r'Time $[s]$', [r'Wind $[m/s]$', r'Variance $[m^2/s^2]$'], save=printfigs, log=1)

    ys1 = {r'$\lambda_{x_r}$': [new_lambdas[0], 1.0],
            r'$\lambda_{y_r}$': [new_lambdas[1], 1.0],
            r'$\lambda_{z_r}$': [new_lambdas[2], 1.0]}
    ys2 = {r'$\sigma^2(\lambda_{x_r})$': [lambda_covariances[0], 1.0],
            r'$\sigma^2(\lambda_{y_r})$': [lambda_covariances[1], 1.0],
            r'$\sigma^2(\lambda_{z_r})$': [lambda_covariances[2], 1.0]}
    make_plots(x, [ys1, ys2], f'{figs_destination} acceleration biases variance over time', r'Time $[s]$', [r'Acceleration Bias $[m/s^2]$', r'$Variance [m^2/s^4]$'], save=printfigs, log=1)

    ys1 = {r'$\lambda_{p_r}$': [new_lambdas[3], 1.0],
            r'$\lambda_{q_r}$': [new_lambdas[4], 1.0],
            r'$\lambda_{r_r}$': [new_lambdas[5], 1.0]}
    ys2 = {r'$\sigma^2(\lambda_{p_r})$': [lambda_covariances[3], 1.0],
            r'$\sigma^2(\lambda_{q_r})$': [lambda_covariances[4], 1.0],
            r'$\sigma^2(\lambda_{r_r})$': [lambda_covariances[5], 1.0]}
    make_plots(x, [ys1, ys2], f'{figs_destination} attitude rate bias variances over time', r'Time $[s]$', [r'Attitude Rate Bias $[rad/s]$', r'$Variance [rad^2/s^2]$'], save=printfigs, log=1)

    ys = {r'$A_x$': [U[0, :], 0.7],
            r'$A_y$': [U[1, :], 0.7],
            r'$A_z$': [U[2, :], 0.7]}
    make_plots(x, [ys], f'{figs_destination} IMU accelerations', r'Time $[s]$', [r'Accelerations $[m/s^2]$'], save=printfigs)

    ys = {r'$A_p$': [U[3, :], 0.7],
            r'$A_q$': [U[4, :], 0.7],  
            r'$A_r$': [U[5, :], 0.7]}  
    make_plots(x, [ys], f'{figs_destination} IMU attitude rates', r'Time $[s]$', [r'Attitude rates $[rad/s]$'], save=printfigs)

    ys = {'Iterations taken by IEKF': [kalman_filter.itr_counts, 1.0]}
    make_plots(x, [ys], f'{figs_destination} Iterations taken by IEKF', r'Time $[s]$', [r'Iterations'], save=printfigs)

    ys = {'innovation': [kalman_filter.innovations[0], 1.0]}
    make_plots(x, [ys], f'{figs_destination} innovation', r'Time $[s]$', [r'Total Innovation'], save=printfigs, log=0)

    if show_plot:
        plt.show()
    
    plt.close('all')

all_results_file.close()
time2 = time.time()
print(f"\nTotal time taken: {round(time2-time1,6)} s")
print("\n\n\n***************************************\nAll filtering finished\n***************************************\n\n\n")