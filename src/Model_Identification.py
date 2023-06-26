########################################################################
# Model identification script 
# 
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################
import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time
from src.KF_Functions import *
from src.Kalman_Filter_Class import *
from src.Plotter import *
from matplotlib import cm


def kf_read_csv(filename, lb, ub):
    
    # Change the current working directory to the data folder and process all original csv files
    train_data = genfromtxt("data/filtered/normal/" + filename, delimiter=',').T
    # train_data = train_data[:,1:]
    train_data = train_data[:, lb:ub]

    Xs_k       = train_data[:9,:]                   # retreiving x,y,z,u,v,w,phi,theta,psi
    Winds      = train_data[9:12,:]                 # retreiving wind velocities
    Biases     = train_data[12:18,:]                # retreiving accelerometer and gyro biases
    U_k        = train_data[18:23,:]                # retriving da,de,dr,tc1,tc2
    dt = 0.01
    time = np.arange(lb*dt, ub*dt, dt)
    
    train_data   = genfromtxt("data/regenerated/noise/" + filename.replace("_filtered", "_measurements"), delimiter=',').T
    train_data   = train_data[:, lb:ub]
    IMU_Accelerations = train_data[15:18,:]

    train_data   = genfromtxt("data/raw/" + filename.replace("_filtered", ""), delimiter=',').T
    # train_data = train_data[:,1:]
    train_data   = train_data[:, lb:ub]
    clean_p      = train_data[4]
    clean_q      = train_data[5]
    clean_r      = train_data[6]
    clean_phi_theta_psi = train_data[1:4]
    actual_clean_alpha  = train_data[9]
    IMU_A_true   = train_data[12:15]                 # These are the IMU acceleration measurements

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
    sin_phi, sin_theta, tan_theta, cos_phi, cos_theta = np.sin(Xs_k[6]), np.sin(Xs_k[7]), np.tan(Xs_k[7]), np.cos(Xs_k[6]), np.cos(Xs_k[7]) 

    pqr = np.zeros_like(ang_vels)
    for i in range(ang_vels.shape[1]):
        matrix_temp = np.array([[1, sin_phi[i]*tan_theta[i], cos_phi[i]*tan_theta[i]],
                                [0, cos_phi[i], -sin_phi[i]],
                                [0, sin_phi[i]/cos_theta[i], cos_phi[i]/cos_theta[i]]])
        inv_matrix  = np.linalg.inv(matrix_temp)   
        pqr[:,i] = (inv_matrix@ang_vels[:,[i]]).flatten()

    # these two are the actual p, q, r and p dot, q dot, r dot
    pqr_dot, _ = kf_finite_difference(dt, pqr, step_size=12)

    # removing the bias and denoising IMU acceleration measurements
    vel_rate_X = IMU_Accelerations - Bias[:3].reshape(-1,1)
    # moving average using the 5 previous accelerations
    vel_rate_X[0,:] = np.convolve(vel_rate_X[0,:], np.ones(12)/12, mode='same')
    vel_rate_X[1,:] = np.convolve(vel_rate_X[1,:], np.ones(12)/12, mode='same')
    vel_rate_X[2,:] = np.convolve(vel_rate_X[2,:], np.ones(12)/12, mode='same')
    # zero order hold for the first and last 5 values
    vel_rate_X[:,:12] = vel_rate_X[:,[12]]
    vel_rate_X[:,-12:] = vel_rate_X[:,[-12]]

    FCs_k = kf_calc_Fc(mass, rho, S, Vs_k, vel_rate_X)                           # calculating the Fc values
    MCs_k = kf_calc_Mc(rho, b, c, S, I, Vs_k, pqr, pqr_dot, )           # calculating the Mc values

    airdata = [alphas_k, betas_k, Vs_k]
    return time, FCs_k, MCs_k, airdata, U_k, pqr, Xs_k


def my_round(value, N):
    exponent = np.ceil(np.log10(abs(value)))
    return 10**exponent*np.round(value*10**(-exponent), N)

########################################################################
## Set script parameters
########################################################################
np.random.seed(7)                    # Set random seed for reproducibility
sns.set(style = "darkgrid")          # Set seaborn style    

save      = False                     # enable saving values or csv data
show_plot = True                     # enable plotting
printfigs = False                     # enable saving figures
model_choice = 1                     # 0 = alternative model, 1 = original model 

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

files = ['da3211_2_filtered.csv', 'de3211_1_filtered.csv', 'dr3211_1_filtered.csv']
maneuver  = [(19, 40), (19.5, 30), (19.5, 29)]  # marking the timestamps in each csv files that contain the test manoeuvre, minimize fitting for wrong data

########################################################################
## Model training loop
########################################################################

for i, filename in enumerate(files):
    print(f"\n\nProcessing data for {filename}...\n\n")
    
    lb ,ub = int(maneuver[i][0]/dt), int(maneuver[i][1]/dt)
    ########################################################################
    ## Data I/O managing
    ########################################################################

    # reading the csv file
    time, FCs_k, MCs_k, airdata, U_k, pqr, Xs_k = kf_read_csv(filename, lb, ub)

    ########################################################################
    ## Estimating the parameters for the models:
    ########################################################################
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

    v = Xs_k[4,:]
    alphas_k, betas_k = airdata[0], airdata[1]
    Vs_k              = airdata[2]
    ps_k, qs_k, rs_k  = pqr[0,:], pqr[1,:], pqr[2,:]
    da, de, dr        = U_k[0,:], U_k[1,:], U_k[2,:]
    Tc                = U_k[4,:]
    consts            = np.ones_like(alphas_k)
    
    if filename=='de3211_1_filtered.csv':
        if model_choice==0: CX_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, de]).T,           name="CX model bigger")
        elif model_choice==1: CX_model = model(np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, de, Tc]).T,    name="CX model")
        CX_model.measurements = FCs_k[0].reshape(-1,1)

        if model_choice==0: CZ_model = model(np.array([consts, alphas_k, alphas_k**3, qs_k*c/Vs_k, de]).T,  name="CZ model bigger")
        elif model_choice==1: CZ_model = model(np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T,                 name="CZ model")
        CZ_model.measurements = FCs_k[2].reshape(-1,1)
            
        if model_choice==0: Cm_model = model(np.array([consts, alphas_k, alphas_k**3, qs_k*c/Vs_k, de]).T,  name="Cm model bigger")
        elif model_choice==1: Cm_model = model(np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T,                 name="Cm model")
        Cm_model.measurements = MCs_k[1].reshape(-1,1)

        models   = [CX_model, CZ_model, Cm_model]

    elif filename=='da3211_2_filtered.csv':
        if model_choice==0: Cl_model = model(np.array([consts, betas_k*ps_k*b/2/Vs_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da]).T, name="Cl model bigger")
        elif model_choice==1: Cl_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="Cl model")
        Cl_model.measurements = MCs_k[0].reshape(-1,1)

        models   = [Cl_model]

    elif filename=='dr3211_1_filtered.csv':
        if model_choice==0: CY_model = model(np.array([consts, betas_k, betas_k**3, rs_k*b/2/Vs_k, dr]).T, name="CY model bigger")
        elif model_choice==1: CY_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="CY model")
        CY_model.measurements = FCs_k[1].reshape(-1,1)
        
        if model_choice==0: Cn_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, dr]).T, name="Cn model bigger")
        elif model_choice==1: Cn_model = model(np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T, name="Cn model")
        Cn_model.measurements = MCs_k[2].reshape(-1,1)

        models   = [CY_model, Cn_model]

    else:
        assert False, f'the filename {filename} is not a training file'

    for i, model_k in enumerate(models):
        name = model_k.name
        print('\n----------------------------------------------------------------')
        print(f'Estimating {name}...')
        model_k.verbose = False
        model_k.OLS_estimate()
        RLS_params = model_k.OLS_params.copy()
        RLS_params[0] = np.random.rand()
        print(f'constant changed from {model_k.OLS_params[0]} to {RLS_params[0]}')
        model_k.RLS_estimate(RLS_params=RLS_params)
        OLS_params = model_k.OLS_params
        RLS_params = model_k.RLS_params
        print(f'\n{model_k.name} OLS params:')
        for j, param in enumerate(OLS_params.flatten()):
            if abs(param) > 10:
                print(f'{param:e}')
            else:
                print(f'{round(param,4)}')

        print(f'\n{model_k.name} RLS params:')
        for j, param in enumerate(RLS_params.flatten()):
            if abs(param) > 10:
                print(f'{param:e}')
            else:
                print(f'{round(param,4)}')


        print(f'\nOLS parameter variances:\n{my_round(model_k.OLS_P.diagonal(),3)}')
        print(f'\nRLS parameter variances:\n{my_round(model_k.RLS_P.diagonal(),2)}')
        ys = {f'{model_k.name} OLS values': [model_k.OLS_y,1.0],
              f'{model_k.name} measurements': [model_k.measurements,0.5]}
        make_plots(time, [ys], f'figs/models/train {model_k.name} OLS {filename.replace("_filtered.csv","")}', 'time [s]', ['Coefficient'], save=printfigs, colors=['C1','C0'])

        ys = {f'{model_k.name} RLS values': [model_k.RLS_y,1.0],
              f'{model_k.name} measurements': [model_k.measurements,0.5]}
        make_plots(time, [ys], f'figs/models/train {model_k.name} RLS {filename.replace("_filtered.csv","")}', 'time [s]', ['Coefficient'], save=printfigs, colors=['C2','C0'])

if show_plot:    
    plt.show()
        
plt.close('all')
########################################################################
## Model validation loop
########################################################################

files = ['dadoublet_1_filtered.csv', 'dedoublet_1_filtered.csv', 'dr3211_2_filtered.csv']
maneuver  = [(10, 15), (9.5, 15), (19.5, 29)]  # marking the timestamps in each csv files that contain the test manoeuvre, minimize fitting for wrong data

for i, filename in enumerate(files):
    print(f"\n\nProcessing data for {filename}...\n\n")
    
    lb ,ub = int(maneuver[i][0]/dt), int(maneuver[i][1]/dt)
    ########################################################################
    ## Data I/O managing
    ########################################################################

    # reading the csv file
    time, FCs_k, MCs_k, airdata, U_k, pqr, Xs_k = kf_read_csv(filename, lb, ub)

    ########################################################################
    ## Estimating the parameters for the models:
    ########################################################################
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
    v = Xs_k[4,:]
    alphas_k, betas_k = airdata[0], airdata[1]
    Vs_k              = airdata[2]
    ps_k, qs_k, rs_k  = pqr[0,:], pqr[1,:], pqr[2,:]
    da, de, dr        = U_k[0,:], U_k[1,:], U_k[2,:]
    Tc                = U_k[4,:]
    consts            = np.ones_like(alphas_k)
    
    if filename=='dedoublet_1_filtered.csv':
        if model_choice==0: CX_model.regression_matrix = np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, de]).T
        elif model_choice==1: CX_model.regression_matrix = np.array([consts, alphas_k, alphas_k**2, qs_k*c/Vs_k, de, Tc]).T
        CX_model.measurements      = FCs_k[0].reshape(-1,1)

        if model_choice==0: CZ_model.regression_matrix = np.array([consts, alphas_k, alphas_k**3, qs_k*c/Vs_k, de]).T
        elif model_choice==1: CZ_model.regression_matrix = np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T
        CZ_model.measurements      = FCs_k[2].reshape(-1,1)
            
        if model_choice==0: Cm_model.regression_matrix = np.array([consts, alphas_k, alphas_k**3, qs_k*c/Vs_k, de]).T
        elif model_choice==1: Cm_model.regression_matrix = np.array([consts, alphas_k, qs_k*c/Vs_k, de, Tc]).T
        Cm_model.measurements      = MCs_k[1].reshape(-1,1)

        models                     = [CX_model, CZ_model, Cm_model]

    elif filename=='dadoublet_1_filtered.csv':
        if model_choice==0: Cl_model.regression_matrix = np.array([consts, betas_k*ps_k*b/2/Vs_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da]).T
        elif model_choice==1: Cl_model.regression_matrix = np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T
        Cl_model.measurements      = MCs_k[0].reshape(-1,1)

        models                     = [Cl_model]

    elif filename=='dr3211_2_filtered.csv':
        if model_choice==0: CY_model.regression_matrix = np.array([consts, betas_k, betas_k**3, rs_k*b/2/Vs_k, dr]).T
        elif model_choice==1: CY_model.regression_matrix = np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T 
        CY_model.measurements      = FCs_k[1].reshape(-1,1)
        
        if model_choice==0: Cn_model.regression_matrix = np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, dr]).T
        elif model_choice==1: Cn_model.regression_matrix = np.array([consts, betas_k, ps_k*b/2/Vs_k, rs_k*b/2/Vs_k, da, dr]).T 
        Cn_model.measurements      = MCs_k[2].reshape(-1,1)

        models                     = [CY_model, Cn_model]
    else:
        assert False, f'the filename {filename} is not a validating file'

    
    for i, model_k in enumerate(models):
        name = model_k.name
        print('\n----------------------------------------------------------------')
        print(f'Validating {name}...')

        model_k.OLS_estimate(validate=True)
        model_k.RLS_estimate(validate=True)

        ys = {f'{model_k.name} OLS values': [model_k.OLS_y,1.0],
              f'{model_k.name} measurements': [model_k.measurements,0.5]}
        make_plots(time, [ys], f'figs/models/validate {model_k.name} OLS {filename.replace("_filtered.csv","")}', 'time [s]', ['Coefficient'], save=printfigs, colors=['C1','C0'])

        ys = {f'{model_k.name} RLS values': [model_k.RLS_y,1.0],
              f'{model_k.name} measurements': [model_k.measurements,0.5]}
        make_plots(time, [ys], f'figs/models/validate {model_k.name} RLS {filename.replace("_filtered.csv","")}', 'time [s]', ['Coefficient'], save=printfigs, colors=['C2','C0'])


if show_plot:    
    plt.show()


plt.close('all')