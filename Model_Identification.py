import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *
from matplotlib import cm


########################################################################
## Set script parameters
########################################################################
np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    

save      = False             # enable saving data
show_plot = True            # enable plotting
printfigs = False             # enable saving figures
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

dt = 0.01       # simulation time step

FCs    = None
MCs    = None
Xs     = None
U      = None
Vs     = None
alphas = None
betas  = None


files = ['da3211_2_filtered.csv', 'dadoublet_1_filtered.csv', 'de3211_1_filtered.csv', 'dedoublet_1_filtered.csv', 'dr3211_1_filtered.csv', 'dr3211_2_filtered.csv']
maneuver  = [(19, 40), (10, 15), (19.5, 30), (9.5, 15), (19.5, 29), (19.5, 29)]  # marking the timestamps in each csv files that contain the test manoeuvre, minimize fitting for wrong data
os.chdir("data/filtered/a/")
files = os.listdir(os.getcwd())
os.chdir("../../..")

for i, filename in enumerate(files):
    print(f"\n\nProcessing data for {filename}...\n\n")
    lb ,ub = int(maneuver[i][0]/dt), int(maneuver[i][1]/dt)
    ########################################################################
    ## Data I/O managing
    ########################################################################

    # Change the current working directory to the data folder and process all original csv files
    train_data = genfromtxt("data/filtered/normal/" + filename, delimiter=',').T
    train_data = train_data[:, lb:ub]

    Xs_k       = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
    Winds      = train_data[9:12,:]                 # retreiving wind velocities
    Biases     = train_data[12:18,:]                # retreiving accelerometer and gyro biases
    U_k        = train_data[18:23,:]                # retriving da,de,dr,tc1,tc2
    N = Xs_k.shape[1]
    dt = 0.01

    train_data = genfromtxt("data/regenerated/noise/" + filename.replace("filtered", "measurements"), delimiter=',').T
    train_data = train_data[:, lb:ub]
    IMU          = train_data[15:21]                 # These are the IMU measurements

    ########################################################################
    ## Calculating the values for some needed data
    ########################################################################
    # Calculate the wind vector
    Wind = Winds[:,-25:].mean(axis=1)

    # Calculate the biases
    Bias = Biases[:,-25:].mean(axis=1)
    
    Vs_k     = np.sqrt(Xs_k[3,:]**2 + Xs_k[4,:]**2 + Xs_k[5,:]**2)               # calculating the airspeed from the velocities
    alphas_k = np.arctan2(Xs_k[5,:], Xs_k[3,:])                                  # calculating the angle of attack from the velocities
    betas_k  = np.arctan2(Xs_k[4,:], np.sqrt(Xs_k[3,:]**2 + Xs_k[5,:]**2))       # calculating the sideslip angle from the velocities

    # finding the aircraft angular accelerations by calculating the second derivatives of the angles using finite difference
    ang_rate_X, ang_accel_X = kf_finite_difference(dt, Xs_k[6:9])                # finding derivative of the flight angles

    # finding the aircraft accelerations by calculating the first derivatives of the velocities using finite difference
    vel_rate_X, _ = kf_finite_difference(dt, Xs_k[3:6])                          # finding derivative of the flight velocities

    FCs_k = kf_calc_Fc(mass, rho, S, Vs_k, vel_rate_X)                           # calculating the Fc values
    MCs_k = kf_calc_Mc(rho, b, c, S, I, Vs_k, ang_rate_X, ang_accel_X)           # calculating the Mc values
    ########################################################################
    ## Estimating the parameters for the force and moment models:
    """
    Force models
    FCs[0,:] = CX = CX0 + CX_alpha*alpha + CX_alpha2*alpha**2 + CX_q*qc/Vinf? + CX_delta_e*delta_e + CX_Tc*Tc
    FCs[2,:] = CY = CY0 + CY_beta*beta + CY_p*pb/2Vinf? + CY_r*rb/2Vinf? + CY_delta_a*delta_a + CY_delta_r*delta_r
    FCs[1,:] = CZ = CZ0 + CZ_alpha*alpha + CZ_q*qc/Vinf? + CZ_de*de + CZ_Tc*Tc

    Moment models
    MCs[0,:] = Cl = Cl0 + Cl_beta*beta + Cl_p*pb/2Vinf? + Cl_r*rb/2Vinf? + Cl_delta_a*delta_a + Cl_delta_r*delta_r
    MCs[1,:] = Cm = Cm0 + Cm_alpha*alpha + Cm_q*qc/Vinf? + Cm_delta_e*delta_e + Cm_Tc*Tc
    MCs[2,:] = Cn = Cn0 + Cn_beta*beta + Cn_p*pb/2Vinf? + Cn_r*rb/2Vinf? + Cn_delta_a*delta_a + Cn_delta_r*delta_r
    """
    ########################################################################
    # Construct the datapoints in the solution space
    alphas_k, betas_k = alphas_k, betas_k
    ps_k, qs_k, rs_k  = ang_rate_X[0,:], ang_rate_X[1,:], ang_rate_X[2,:]
    Vinf              = Vs_k[0]
    da, de, dr        = U_k[0,:], U_k[1,:], U_k[2,:]
    Tc                = U_k[4,:]
    consts            = np.ones_like(alphas_k)
    
    CX_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vinf, de, Tc]).T,    name="CX model")
    CX_model.measurements = FCs_k[0].reshape(N,1)

    CY_model = model(np.array([consts, betas_k, ps_k*b/2/Vinf, rs_k*b/2/Vinf, da, dr]).T, name="CY model")
    CY_model.measurements = FCs_k[1].reshape(N,1)
    
    CZ_model = model(np.array([consts, alphas_k, qs_k*c/Vinf, de, Tc]).T,                 name="CY model")
    CZ_model.measurements = FCs_k[2].reshape(N,1)

    Cl_model = model(np.array([consts, betas_k, ps_k*b/2/Vinf, rs_k*b/2/Vinf, da, dr]).T, name="Cl model")
    Cl_model.measurements = MCs_k[0].reshape(N,1)

    Cm_model = model(np.array([consts, alphas_k, qs_k*c/Vinf, de, Tc]).T,                 name="Cm model")
    Cm_model.measurements = MCs_k[1].reshape(N,1)

    Cn_model = model(np.array([consts, betas_k, ps_k*b/2/Vinf, rs_k*b/2/Vinf, da, dr]).T, name="Cn model")
    Cn_model.measurements = MCs_k[2].reshape(N,1)

    models = [CX_model]

    for model_k in models:
        model_k.verbose = True
        model_k.OLS_estimate()
        model_k.MLE_estimate()
        # model_k.RLS_estimate()

        print(f'{model_k.name} OLS params: {model_k.OLS_params} (RMSE: {model_k.OLS_RMSE})')
        print(f'{model_k.name} MLE params: {model_k.MLE_params} (RMSE: {model_k.MLE_RMSE})')
        # print(f'{model_k.name} RLS params: {model_k.RLS_params} (RMSE: {model_k.RLS_RMSE})')

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
    ########################################################################
    ## Plotting some results for visualization
    ########################################################################
    x = dt*np.arange(0, Xs_k.shape[1], 1)

    # ys = {r'$\dot{p}$': [ang_accel_X[0,:], 0.8],
    #       r'$\dot{q}$': [ang_accel_X[1,:], 0.8],
    #       r'$\dot{r}$': [ang_accel_X[2,:], 0.8]}
    # make_plots(x, [ys], f"figs/models/{filename} Angular Accelerations", "Time [s]", [r"Angular Accelerations $[rad/s^2]$"],save=printfigs)
    # ys = {r'$A_x$': [vel_rate_X[0,:], 0.8],
    #       r'$A_y$': [vel_rate_X[1,:], 0.8],
    #       r'$A_z$': [vel_rate_X[2,:], 0.8]}
    # ys2 = {'vx': [Xs_k[3,:], 0.8],
    #       'vy': [Xs_k[4,:], 0.8],
    #       'vz': [Xs_k[5,:], 0.8]}
    # make_plots(x, [ys, ys2], f"figs/models/{filename} Accelerations and velocities", "Time [s]", [r"Acceleration $[m/s^2]$",r"Flight Velocity $[m/s]$"],save=printfigs)
    # ys = {'phi': [Xs_k[6,:], 0.8],
    #       'theta': [Xs_k[7,:], 0.8],
    #       'alpha': [alphas_k, 0.8],
    #       'psi': [Xs_k[8,:], 0.8]}
    # make_plots(x, [ys], f"figs/models/{filename} Flight Angles", "Time [s]", [r"Flight Angle $[rad]$"],save=printfigs)
    ys = {'Vtas': [Vs_k, 0.8]}
    make_plots(x, [ys], f"figs/models/{filename} Airspeed", "Time [s]", [r"Airspeed $[m/s]$"],save=printfigs)
    # ys = {'alpha': [alphas_k, 0.8],
    #       'beta': [betas_k, 0.8]}
    # make_plots(x, [ys], f"figs/models/{filename} Alpha and Beta", "Time [s]", [r"Angle$[rad]$"],save=printfigs)

    # ys = {r'IMU A_x': [IMU[0,:], 0.8],
    #       r'IMU A_y': [IMU[1,:], 0.8],
    #       r'IMU A_z': [IMU[2,:], 0.8]}
    # make_plots(x, [ys], "IMU Accelerations", "Time [s]", [r"Acceleration $[m/s^2]$"])
    # ys = {r'phi': [Xs_k[6,:], 0.8],
    #       r'theta': [Xs_k[7,:], 0.8],
    #       r'psi': [Xs_k[8,:], 0.8]}
    # make_plots(x, [ys], "Flight Angles", "Time [s]", [r"Flight Angle $[rad]$"])
    # plt.show()
    # ys = {'CX': [FCs_k[0,:], 0.8],
    #       'CY': [FCs_k[2,:], 0.8],
    #       'CZ': [FCs_k[1,:], 0.8]}
    # make_plots(x, [ys], f"figs/models/{filename} Force Coefficients", "Time [s]", [r"Force Coefficient $[-]$"],save=printfigs)
    # ys = {'Cl': [MCs_k[0,:], 0.8],
    #         'Cm': [MCs_k[1,:], 0.8],
    #         'Cn': [MCs_k[2,:], 0.8]}
    # make_plots(x, [ys], f"figs/models/{filename} Moment Coefficients", "Time [s]", [r"Moment Coefficient $[-]$"],save=printfigs)
    plt.figure()
    plt.scatter(x, FCs_k[0,:], label='CX', s=0.1)
    plt.scatter(x, FCs_k[1,:], label='CZ', s=0.1)
    plt.scatter(x, FCs_k[2,:], label='CY', s=0.1)
    plt.legend()
    plt.grid()

    # plt.figure()
    # plt.scatter(x, MCs_k[0,:], label='Cl', s=0.1)
    # plt.scatter(x, MCs_k[1,:], label='Cm', s=0.1)
    # plt.scatter(x, MCs_k[2,:], label='Cn', s=0.1)
    # plt.legend()
    # plt.grid()
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
    if show_plot:
        plt.show()

