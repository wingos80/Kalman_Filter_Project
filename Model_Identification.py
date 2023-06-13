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

g = 9.80665

# files = ['da3211_2_filtered.csv', 'dadoublet_1_filtered.csv', 'de3211_1_filtered.csv', 'dedoublet_1_filtered.csv', 'dr3211_1_filtered.csv', 'dr3211_2_filtered.csv']
# maneuver  = [(19, 40), (10, 15), (19.5, 30), (9.5, 15), (19.5, 29), (19.5, 29)]  # marking the timestamps in each csv files that contain the test manoeuvre, minimize fitting for wrong data
os.chdir("data/filtered/normal/")
files = os.listdir(os.getcwd())
os.chdir("../../..")
# files = ["dr3211_1_filtered.csv"]
# maneuver = [ (19.5, 29)]
# files = ["de3211_1_filtered.csv"]
# maneuver = [(19.5, 30)]
files = ["dedoublet_1_filtered.csv"]
maneuver = [(9.5, 15)]
# files = ["da3211_2_filtered.csv"]
# maneuver  = [(19, 40)]
for i, filename in enumerate(files):
    print(f"\n\nProcessing data for {filename}...\n\n")
    lb ,ub = int(maneuver[i][0]/dt), int(maneuver[i][1]/dt)
    ########################################################################
    ## Data I/O managing
    ########################################################################

    # Change the current working directory to the data folder and process all original csv files
    train_data = genfromtxt("data/filtered/normal/" + filename, delimiter=',').T
    # train_data = train_data[:,1:]
    train_data = train_data[:, lb:ub]

    Xs_k       = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
    Winds      = train_data[9:12,:]                 # retreiving wind velocities
    Biases     = train_data[12:18,:]                # retreiving accelerometer and gyro biases
    U_k        = train_data[18:23,:]                # retriving da,de,dr,tc1,tc2
    N = Xs_k.shape[1]
    dt = 0.01

    train_data   = genfromtxt("data/raw/" + filename.replace("_filtered", ""), delimiter=',').T
    # train_data = train_data[:,1:]
    train_data   = train_data[:, lb:ub]
    clean_p      = train_data[4]
    clean_q      = train_data[5]
    clean_r      = train_data[6]
    clean_phi_theta_psi = train_data[1:4]
    actual_clean_alpha  = train_data[9]
    time         = train_data[0]
    IMU_A        = train_data[12:15]                 # These are the IMU acceleration measurements

    # train_data   = genfromtxt("data/regenerated/noise/" + filename.replace("_filtered", "_measurements"), delimiter=',').T
    # # train_data = train_data[:,1:]
    # train_data   = train_data[:, lb:ub]
    clean_u      = train_data[-3]
    clean_v      = train_data[-2]
    clean_w      = train_data[-1]
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
    ang_vels, _ = kf_finite_difference(dt, Xs_k[6:9], step_size=12)  # finding derivative of the flight angles
    sin_phi   = np.sin(Xs_k[6])
    sin_theta = np.sin(Xs_k[7])
    tan_theta = np.tan(Xs_k[7])
    cos_phi   = np.cos(Xs_k[6])
    cos_theta = np.cos(Xs_k[7])
    # sin_phi   = np.sin(clean_phi_theta_psi[0])
    # sin_theta = np.sin(clean_phi_theta_psi[1])
    # tan_theta = np.tan(clean_phi_theta_psi[1])
    # cos_phi   = np.cos(clean_phi_theta_psi[0])
    # cos_theta = np.cos(clean_phi_theta_psi[1])

    pqr = np.zeros_like(ang_vels)
    for i in range(ang_vels.shape[1]):
        matrix_temp = np.array([[1, sin_phi[i]*tan_theta[i], cos_phi[i]*tan_theta[i]],
                                [0, cos_phi[i], -sin_phi[i]],
                                [0, sin_phi[i]/cos_theta[i], cos_phi[i]/cos_theta[i]]])
        inv_matrix  = np.linalg.inv(matrix_temp)   
        pqr[:,i] = (inv_matrix@ang_vels[:,[i]]).flatten()

    # these two are the actual p, q, r and p dot, q dot, r dot
    pqr_dot, _ = kf_finite_difference(dt, pqr, step_size=12)


    uvw = np.array([clean_u,clean_v,clean_w])
    # finding the aircraft accelerations by calculating the first derivatives of the velocities using finite difference
    vel_rate_X, _ = kf_finite_difference(dt, Xs_k[3:6], step_size=12)            # finding derivative of the flight velocities
    # vel_rate_X, _ = kf_finite_difference(dt, uvw, step_size=8)
    uvw = Xs_k[3:6]
    vel_rate_X[0] = vel_rate_X[0] + g*sin_theta - pqr[2]*uvw[1] + pqr[1]*uvw[2]
    vel_rate_X[1] = vel_rate_X[1] - g*cos_theta*sin_phi - pqr[0]*uvw[2] + pqr[2]*uvw[0]
    vel_rate_X[2] = vel_rate_X[2] - g*cos_theta*cos_phi - pqr[1]*uvw[0] + pqr[0]*uvw[1]

    FCs_k = kf_calc_Fc(mass, rho, S, Vs_k, vel_rate_X)                           # calculating the Fc values
    MCs_k = kf_calc_Mc(rho, b, c, S, I, Vs_k, pqr, pqr_dot, )           # calculating the Mc values

    # fig = plt.figure()
    # fig.canvas.manager.set_window_title(f"{filename.replace('_filtered.csv', '')}_q_alpha_envelope") 
    # plt.scatter(actual_clean_alpha[::4], clean_q[::4], label="Simulations",alpha=0.5)
    # plt.scatter(alphas_k[::4], pqr[1,::4], label="Reconstructed",alpha=0.5)
    # plt.xlabel(r"$\alpha$ $[rad]$")
    # plt.ylabel(r"q $[rad/s]$")
    # plt.legend()
    # plt.tight_layout()

    # fig = plt.figure()
    # fig.canvas.manager.set_window_title(f"{filename.replace('_filtered.csv', '')}_p_r_envelope") 
    # plt.scatter(clean_p[::4], clean_r[::4], label="Simulations",alpha=0.5)
    # plt.scatter(pqr[0,::4], pqr[2,::4], label="Reconstructed",alpha=0.5)
    # plt.xlabel(r"p $[rad/s]$")
    # plt.ylabel(r"r $[rad/s]$")
    # plt.legend()
    # plt.tight_layout()
    
    # plt.figure()
    # plt.plot(time, pqr_dot[1,:],label="Finite Difference")
    # plt.xlabel(r"Time $[s]$")
    # plt.ylabel(r"Pitch Acceleration $[rad/s^2]$")
    # plt.legend()
    # plt.tight_layout()
    
    plt.figure()
    plt.plot(time, vel_rate_X[0],label="Finite Difference")
    plt.plot(time, IMU_A[0], label="Simulations")
    plt.xlabel(r"Time $[s]$")
    plt.ylabel(r"$A_X [m/s^2]$")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, vel_rate_X[1],label="Finite Difference")
    plt.plot(time, IMU_A[1], label="Simulations")
    plt.xlabel(r"Time $[s]$")
    plt.ylabel(r"$A_Y [m/s^2]$")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(time, vel_rate_X[2],label="Finite Difference")
    plt.plot(time, IMU_A[2], label="Simulations")
    plt.xlabel(r"Time $[s]$")
    plt.ylabel(r"$A_Z [m/s^2]$")
    plt.legend()
    plt.tight_layout()

    # plt.figure()
    # plt.plot(time, pqr[0,:],label="Finite Difference")
    # plt.plot(time, clean_p, label="Simulations")
    # plt.xlabel(r"Time $[s]$")
    # plt.ylabel(r"Roll Rate $[rad/s^2]$")
    # plt.legend()
    # plt.tight_layout()

    # plt.figure()
    # plt.plot(time, pqr[1,:],label="Finite Difference")
    # plt.plot(time, clean_q, label="Simulations")
    # plt.xlabel(r"Time $[s]$")
    # plt.ylabel(r"Pitch Rate $[rad/s^2]$")
    # plt.legend()
    # plt.tight_layout()

    # plt.figure()
    # plt.plot(time, pqr[2,:],label="Finite Difference")
    # plt.plot(time, clean_r, label="Simulations")
    # plt.xlabel(r"Time $[s]$")
    # plt.ylabel(r"Yaw Rate $[rad/s^2]$")
    # plt.legend()
    # plt.tight_layout()
    plt.show()

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
    # Construct the datapoints in the solution space, and estimate the parameters of all the models

    alphas_k, betas_k = alphas_k, betas_k
    ps_k, qs_k, rs_k  = pqr[0,:], pqr[1,:], pqr[2,:]
    Vinf              = Vs_k[0]
    da, de, dr        = U_k[0,:], U_k[1,:], U_k[2,:]
    Tc                = U_k[4,:]
    consts            = np.ones_like(alphas_k)
    
    CX_model = model(np.array([consts, betas_k, betas_k**2, alphas_k, alphas_k**2, alphas_k**3, qs_k*c/Vs_k, (qs_k*c/Vs_k)**2, de, Tc]).T,    name="CX model bigger")
    CX_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, de, Tc]).T,    name="CX model")
    CX_model.measurements = FCs_k[0].reshape(N,1)

    CY_model = model(np.array([consts, betas_k, betas_k**2, ps_k*b/2/Vs_k, (ps_k*b/2/Vs_k)**2, rs_k*b/2/Vs_k, (rs_k*b/2/Vs_k)**2, da, dr]).T, name="CY model bigger")
    CY_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="CY model")
    CY_model.measurements = FCs_k[1].reshape(N,1)
    
    CZ_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, (qs_k*c/Vs_k)**2, de, Tc]).T,                                      name="CZ model bigger")
    CZ_model = model(np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T,                 name="CZ model")
    CZ_model.measurements = FCs_k[2].reshape(N,1)

    Cl_model = model(np.array([consts, betas_k, betas_k**2, ps_k*b/2/Vs_k, (ps_k*b/2/Vs_k)**2, rs_k*b/2/Vs_k, (rs_k*b/2/Vs_k)**2, da, dr]).T, name="Cl model bigger")
    Cl_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="Cl model")
    Cl_model.measurements = MCs_k[0].reshape(N,1)

    Cm_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, (qs_k*c/Vs_k)**2, de, Tc]).T,                                      name="Cm model bigger")
    Cm_model = model(np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T,                 name="Cm model")
    Cm_model.measurements = MCs_k[1].reshape(N,1)

    Cn_model = model(np.array([consts, betas_k, betas_k**2, ps_k*b/2/Vs_k, (ps_k*b/2/Vs_k)**2, rs_k*b/2/Vs_k, (rs_k*b/2/Vs_k)**2, da, dr]).T, name="Cn model bigger")
    Cn_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="Cn model")
    Cn_model.measurements = MCs_k[2].reshape(N,1)

    # models = [CX_model, CY_model, CZ_model]
    models = [CX_model, CY_model, CZ_model, Cl_model, Cm_model, Cn_model]

    for i, model_k in enumerate(models):
        print(f'Estimating {model_k.name}...')
        model_k.verbose = False
        model_k.OLS_estimate()
        model_k.MLE_estimate(solver='scipy')
        model_k.RLS_estimate()

        print(f'{model_k.name} OLS params: \n{model_k.OLS_params} (RMSE, R2: {model_k.OLS_RMSE}, {model_k.OLS_R2})')
        print(f'{model_k.name} RLS params: \n{model_k.RLS_params} (RMSE, R2: {model_k.RLS_RMSE}, {model_k.RLS_R2})')
        print(f'{model_k.name} MLE params: \n{model_k.MLE_params} (RMSE, R2: {model_k.MLE_RMSE}, {model_k.MLE_R2})')

        ys = {f'{model_k.name} OLS values': [model_k.OLS_y,0.8],
              f'{model_k.name} MLE values': [model_k.MLE_y,0.8],
            #   f'{model_k.name} RLS values': [model_k.RLS_y,0.5],              
              f'{model_k.name} measurements': [model_k.measurements,1.0]}
        make_plots(time, [ys], f'{model_k.name} OLS', 'time [s]', 'value', save=False)
        
        # print(f'{model_k.name} RLS params: {model_k.RLS_params} (RMSE, R2: {model_k.RLS_RMSE}, {model_k.RLS_R2})')
    plt.show()
    ########################################################################
    ## Plotting some verification
    ########################################################################
    
    # verification stuff
    filename = 'dadoublet_1_filtered.csv'
    train_data = genfromtxt("data/filtered/normal/" + filename, delimiter=',').T
    maneuver = [(10, 15)]
    lb ,ub = int(maneuver[i][0]/dt), int(maneuver[i][1]/dt)
    train_data = train_data[:, lb:ub]
    Xs_k       = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
    Vs_k     = np.sqrt(Xs_k[3,:]**2 + Xs_k[4,:]**2 + Xs_k[5,:]**2)               # calculating the airspeed from the velocities

    # finding the aircraft angular accelerations by calculating the second derivatives of the angles using finite difference
    pqr, pqr_dot = kf_finite_difference(dt, Xs_k[6:9], step_size=8)  # finding derivative of the flight angles
    
    # finding the aircraft accelerations by calculating the first derivatives of the velocities using finite difference
    vel_rate_X, _ = kf_finite_difference(dt, Xs_k[3:6], step_size=8)            # finding derivative of the flight velocities

    FCs_k = kf_calc_Fc(mass, rho, S, Vs_k, vel_rate_X)                           # calculating the Fc values
    MCs_k = kf_calc_Mc(rho, b, c, S, I, Vs_k, pqr, pqr_dot)           # calculating the Mc values




    x = dt*np.arange(0, N, 1)

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

