# time, phi, theta, psi, p, q, r, vtas, Mach, alpha, beta, gamma, Ax, Ay, Az, da, de, dr, dta, dte, dtr, flaps, gear, Tc1, Tc2, u_n, v_n, w_n = 1
# "time", "phi", "theta", "psi", "p", "q", "r", "vtas", "Mach", "alpha", "beta", "gamma", "Ax", "Ay", "Az", "da", "de", "dr", "dta", "dte", "dtr", "flaps", "gear", "Tc1", "Tc2", "u_n", "v_n", "w_n"
########################################################################
# Data preprocessing, trying to regenerate all positions, and GPS, IMU, airdata measurements 
# for the given flight
# 
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 07-04-2023
########################################################################
import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
# import time, sys, os, control.matlab
np.random.seed(7)                            # Set random seed for reproducibility
sns.set(style = "darkgrid")                  # Set seaborn style    


def rk4(fn, xin, uin, t):
    """
    4th order Runge-Kutta method for solving ODEs
    
    Parameters
    ----------
    fn : function
        function handle for the derivatives of the state vector

    xin : numpy.ndarray
        initial state vector
    
    uin : numpy.ndarray
        input vector

    t : numpy.ndarray
        time vector

    Returns
    -------
    t : numpy.ndarray
        time vector (same as input)

    xout : numpy.ndarray
        state vector values at next time step
        """
    
    a   = t[0]
    b   = t[1]
    x   = xin
    N   = 2
    h   = (b - a)/N
    t   = a

    for j in range(1, N+1):
        K1  = h*fn(t, x, uin)
        K2  = h*fn(t+h/2, x+K1/2, uin)
        K3  = h*fn(t+h/2, x+K2/2, uin)
        K4  = h*fn(t+h, x+K3, uin)
        
        x   = x + (K1 + 2*K2 + 2*K3 +K4)/6
        t   = a+j*h
    xout = x
    return t, xout


# x = [x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz]
#  u = [Ax, Ay, Az, p, q, r]
def kf_calc_f(t, X, U):
    """
    Calculates the system dynamics equation f(X,U,t)
    
    Parameters
    ----------
    t : float
    
    X : numpy.ndarray (n,1)
        state vector
        
    U : numpy.ndarray
        input vector
        
    Returns
    -------
    Xdot : numpy.ndarray (n,1)
        time derivative of the state vector, system dynamics
    """
    
    n       = x.size
    Xdot    = np.zeros([n,1])
    g = 9.80665             # gravitational acceleration [m/s^2]

    # saving the individual state and input names to make the code more readable
    x, y, z, u, v, w, phi, theta, psi, Wx, Wy, Wz = X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11]
    Ax, Ay, Az, p, q, r = U[0], U[1], U[2], U[3], U[4], U[5]


    # system dynamics go here
    A = u*np.cos(theta) + (v*np.sin(phi) + w*np.cos(phi))*np.sin(theta)  # saving some big terms to make expressions more readable
    B = (v*np.cos(phi) - w*np.sin(phi))                                    # saving some big terms to make expressions more readable

    Xdot[0] = A*np.cos(psi) - B*np.sin(psi) + Wx
    Xdot[1] = A*np.sin(psi) + B*np.cos(psi) + Wy
    Xdot[2] = -u*np.sin(theta) + (v*np.sin(phi) + w*np.cos(phi))*np.cos(theta) + Wz
    Xdot[3] = Ax - g*np.sin(theta) + r*v - q*w
    Xdot[4] = Ay + g*np.cos(theta)*np.sin(phi) + p*w - r*u
    Xdot[5] = Az + g*np.cos(theta)*np.cos(phi) + q*u - p*v
    Xdot[6] = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    Xdot[7] = q*np.cos(phi) - r*np.sin(phi)
    Xdot[8] = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
    Xdot[9] = 0
    Xdot[10] = 0
    Xdot[11] = 0
    return Xdot

########################################################################
## Data I/O managing
########################################################################

# time, phi, theta, psi, u_n, v_n, w_n = None
Wx, Wy, Wz = 2, -8, 1

filename   = 'data/de3211_1.csv'
train_data = genfromtxt(filename, delimiter=',').T
train_data = train_data[:, 1:]

time       = train_data[0]
phi        = train_data[1]*np.pi/180
theta      = train_data[2]*np.pi/180
psi        = train_data[3]*np.pi/180
u_n        = train_data[25]
v_n        = train_data[26]
w_n        = train_data[27]

result_file = open(f"data/F16traindata_CMabV_2023_kf.csv", "w")
result_file.write(f"Cm, alpha_m, beta_m, V_t, alpha_m_kf, beta_m_kf, V_t_kf, alpha_t_kf, C_a_up\n")

# Initial state vector, 12x1
X0 = np.array([[0],[0],[0], [u_n[0]], [v_n[0]], [w_n[0]], [phi[0]], [theta[0]], [psi[0]], [Wx], [Wy], [Wz]])
X1 = np.zeros_like(X0)
U  = np.zeros([6,1])

xyz = np.zeros([3, len(time)])
for k in range(len(time)-1):
    t_vector = [time[k], time[k+1]]
    t, X1 = rk4(kf_calc_f, X0, U, t_vector)

    #  only pick out the new xyz states and store them
    xyz[:, k+1] = X1[0:3,0]
    
    X1[3:] = np.array([[u_n[k+1]], [v_n[k+1]], [w_n[k+1]], [phi[k+1]], [theta[k+1]], [psi[k+1]], [Wx], [Wy], [Wz]])
    X0 = X1

#  use 3D scatter plot to visualize the xyz positions over time
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o', label='Measured', s=1)
plt.title('Reconstructed flight path', fontsize = 18)
plt.xlabel('x (m)', fontsize = 14)
plt.ylabel('y (m)', fontsize = 14)
ax.set_zlabel('z (m)', fontsize = 14)
plt.show()