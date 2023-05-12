import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *


x = np.arange(0,2,0.1)
y = 10-3*x + 5*x**2
n = 20
OLS_RMSE = 0
MLE_RMSE = 0
for i in range(n):
    y_noise = y+np.random.normal(0,1,y.size)

    f_model = model(np.array([np.ones_like(x), x, x**2]).T)
    f_model.measurements = y_noise.reshape(y_noise.size,1)
    f_model.OLS_estimate()
    f_model.MLE_estimate()

    a = (f_model.OLS_y.flatten()-y)
    a = a**2
    OLS_RMSE += np.sqrt(np.sum((f_model.OLS_y.flatten()-y)**2))
    MLE_RMSE += np.sqrt(np.sum((f_model.MLE_y.flatten()-y)**2))

print(f'average OLS RMSE: {OLS_RMSE/n}')
print(f'average MLE RMSE: {MLE_RMSE/n}')

# prob = f_model.log_likelihood2(np.array([[10,-3]]))

# print('\n\nperfect fitness', prob)
# print('\n\nmle best fitness', f_model.MLE_best)
# print('\n\n',f_model.MLE_params)

# plt.figure()
# plt.plot(x,y, label='original function',alpha=0.4)
# plt.plot(x,y_noise, label='noisey function',alpha=0.4)
# plt.plot(x,f_model.OLS_y, label='OLS estimate')
# plt.plot(x,f_model.MLE_y, label='MLE estimate')
# plt.scatter(x,y)
# plt.grid()
# plt.legend()


# # plt.figure()
# # plt.plot(volumes, label='volumes')
# # plt.grid()
# # plt.legend()
# plt.show()