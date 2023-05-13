import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *


x = np.arange(-2,7,0.01)
y = 10 - 3*x + 5*x**2 - x**3
# y = 1 + 2*x - 0.3*x**2 -x**3 + 0.25*x**4
n = 1
OLS_RMSE = 0
MLE_RMSE = 0
RLS_RMSE = 0

for i in range(n):
    y_noise = y+np.random.normal(0,4,y.size)

    f_model = model(np.array([np.ones_like(x), x, x**2,x**3]).T,verbose=True)
    f_model.measurements = y_noise.reshape(y_noise.size,1)
    f_model.OLS_estimate()
    f_model.MLE_estimate(solver="scipy")
    f_model.RLS_estimate()



    OLS_RMSE += np.sqrt(np.sum((f_model.OLS_y.flatten()-y)**2))
    MLE_RMSE += np.sqrt(np.sum((f_model.MLE_y.flatten()-y)**2))
    RLS_RMSE += np.sqrt(np.sum((f_model.RLS_y.flatten()-y)**2))

print(f'average OLS RMSE: {OLS_RMSE/n}')
print(f'average MLE RMSE: {MLE_RMSE/n}')
print(f'average RLS RMSE: {RLS_RMSE/n}')

# prob = f_model.log_likelihood2(np.array([[10,-3]]))

# print('\n\nperfect fitness', prob)
print('\n\nmle best fitness', f_model.MLE_best)
print('\n\nOLS params: ',f_model.OLS_params.flatten())
print('\n\nMLE params: ',f_model.MLE_params.flatten())
print('\n\nRLS params: ',f_model.RLS_params.flatten())

plt.figure()
plt.plot(x,y, label='original function',alpha=0.4)
plt.plot(x,y_noise, label='noisey function',alpha=0.4)
plt.plot(x,f_model.OLS_y, label='OLS estimate')
plt.plot(x,f_model.MLE_y, label='MLE estimate');
plt.plot(x,f_model.RLS_y, label='RLS estimate');
# plt.ylim(-2.5,6)
plt.grid()
plt.scatter(x,y)
plt.legend()


# plt.figure()
# plt.plot(volumes, label='volumes')
# plt.grid()
# plt.legend()
plt.show()