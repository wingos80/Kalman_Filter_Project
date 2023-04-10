########################################################################
# File to plot the datapoints from the F16 flight data, filtered and unfiltered
# 
#   Author: Wing Chan
#   Email: wingyc80@gmail.com
#   Date: 09-04-2023
########################################################################
import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *
from matplotlib import cm

sns.set(style = "darkgrid")                   # Set seaborn style

########################################################################
## Data I/O managing
########################################################################

filename = 'data/F16traindata_CMabV_2023_kf.csv'
all_data = genfromtxt(filename, delimiter=',')
all_data = all_data[1:,:]

vali_data = []                        # Validation data
train_data = []                       # Training data

for i in range(len(all_data)):
    if i%3 == 0:
        train_data.append(all_data[i])
    else:
        vali_data.append(all_data[i])

# convert validation and train lsits to np.array
vali_data = np.array(vali_data).T
train_data = np.array(train_data).T

C_m       , C_m_v         = train_data[0], vali_data[0]                    # Measured C_m (training, validating)

alpha, alpha_v  = train_data[7], vali_data[7]                    # KF estimated true alpha (training, validating)
beta , beta_v   = train_data[5], vali_data[5]                    # KF estimated beta (training, validating)
V    , V_v      = train_data[6], vali_data[6]                    # KF estimated velocity (training, validating)
dt = 0.01
N = len(C_m)


########################################################################
## Creating the OLS cubic polynomial model
## model: f(x, y) = a + bx + cy + dxx + eyy + fxy + gxxx + hxxy + ixyy + jyyy
########################################################################

A = np.zeros((N, 10))   # regression matrix
A[:,0] = 1
A[:,1] = alpha
A[:,2] = beta
A[:,3] = alpha**2
A[:,4] = beta**2
A[:,5] = alpha*beta
A[:,6] = alpha**3
A[:,7] = alpha**2*beta
A[:,8] = alpha*beta**2
A[:,9] = beta**3

# OLS model
theta = np.linalg.inv(A.T@A)@A.T@C_m   # theta = (A.T*A)^-1*A.T*C_m, parameters for the cubic polynomial

# Plot the OLS model over the alpha beta domain spanned by the training data
min_beta, max_beta = np.min(beta), np.max(beta)
min_alpha, max_alpha = np.min(alpha), np.max(alpha)

n = 100
X = np.linspace(min_alpha, max_alpha, n)
Y = np.linspace(min_beta, max_beta, n)
XX, YY = np.meshgrid(X, Y)


Test_matrix = np.zeros((n*n, 10))
Test_matrix[:,0] = 1
Test_matrix[:,1] = XX.flatten()
Test_matrix[:,2] = YY.flatten()
Test_matrix[:,3] = XX.flatten()**2
Test_matrix[:,4] = YY.flatten()**2
Test_matrix[:,5] = XX.flatten()*YY.flatten()
Test_matrix[:,6] = XX.flatten()**3
Test_matrix[:,7] = XX.flatten()**2*YY.flatten()
Test_matrix[:,8] = XX.flatten()*YY.flatten()**2
Test_matrix[:,9] = YY.flatten()**3

Z = Test_matrix@theta

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(alpha, beta, C_m, c='b', marker='o', alpha = 0.5, label='OLS training data', s=0.2)
ax.plot_surface(XX, YY, Z.reshape(n,n), cmap=cm.coolwarm,
                       linewidth=0, label='OLS model')

#  finding the covariance matrix of the ols parameters
theta_cov = np.linalg.inv(A.T@A)   # covariance matrix of the parameters
min_var = np.min(np.diag(theta_cov))
theta_cov = theta_cov/min_var
print(f'OLS model parameters: {theta}\n')
print(f'OLS model parameters variances: {np.diag(theta_cov)}\n')

fig, ax = plt.subplots()
z = ax.imshow(theta_cov, cmap='RdBu')
fig.colorbar(z)

plt.show()