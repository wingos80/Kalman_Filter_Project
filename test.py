import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *


x = np.arange(-2,7,0.1)
y = -1 +8*x + 5*x**2 - 2*x**3 + 0.15*x**4 + np.random.normal(0,0.1,len(x))
# y = 1 + 2*x - 0.3*x**2 -x**3 + 0.25*x**4
n = 1
OLS_RMSE = 0
MLE_RMSE = 0
RLS_RMSE = 0
MLE_ES = 0
MLE_SCI = 0

plt.figure()
for i in range(n):
    y_noise = y+3*np.random.randn(y.size)

    f_model = model(np.array([np.ones_like(x), x, x**2, x**3, x**4]).T,verbose=True)
    f_model.measurements = y_noise.reshape(y_noise.size,1)
    f_model.OLS_estimate()
    # f_model.MLE_estimate(solver="scipy")
    # print(f'scipy mle params: {f_model.MLE_params}')
    # plt.plot(x,f_model.MLE_y, label='MLE estimate scipy')
    # MLE_SCI += np.sqrt(np.sum((f_model.MLE_y.flatten()-y)**2))

    f_model.RLS_estimate(RLS_params=f_model.OLS_params)
    plt.show()
    f_model.MLE_estimate(solver="scipy")
    # print(f'ES mle params: {f_model.MLE_params}')
    # plt.plot(x,f_model.MLE_y, label='MLE estimate ES')
    MLE_ES += np.sqrt(np.sum((f_model.MLE_y.flatten()-y)**2))


    # OLS_RMSE += np.sqrt(np.sum((f_model.OLS_y.flatten()-y)**2))
    # MLE_RMSE += np.sqrt(np.sum((f_model.MLE_y.flatten()-y)**2))
    # RLS_RMSE += np.sqrt(np.sum((f_model.RLS_y.flatten()-y)**2))

# print(f'average OLS RMSE, R2: {OLS_RMSE/n}, {f_model.OLS_R2}')
# # print(f'average MLE RMSE, R2: {MLE_RMSE/n}, {f_model.MLE_R2}')
# # print(f'average RLS RMSE, R2: {RLS_RMSE/n}, {f_model.RLS_R2}')
# # print(f'average MLE ES RMSE: {MLE_ES/n}')
# print(f'average MLE SCI RMSE: {MLE_SCI/n}')

# prob = f_model.log_likelihood2(np.array([[10,-3]]))

# print('\n\nperfect fitness', prob)
print('\n\nmle best fitness', f_model.MLE_best)
print('\n\nOLS params: ',f_model.OLS_params.flatten())
print('\n\nMLE params: ',f_model.MLE_params.flatten())

print(f'\n\nOLS r2: {f_model.OLS_R2}')
print(f'\n\nMLE r2: {f_model.MLE_R2}')
# print('\n\nRLS params: ',f_model.RLS_params.flatten())

print(f'\n\nOLS covariance: {f_model.OLS_cov}')

print(f'ols rmse_rel, eps_max, r2: {f_model.OLS_RMSE_rel}, {f_model.OLS_eps_max}, {f_model.OLS_R2}')
print(f'mle rmse_rel, eps_max, r2: {f_model.MLE_RMSE_rel}, {f_model.MLE_eps_max}, {f_model.MLE_R2}')
print(f'rls rmse_rel, eps_max, r2: {f_model.RLS_RMSE_rel}, {f_model.RLS_eps_max}, {f_model.RLS_R2}')
# plt.plot(x,y, label='original function',alpha=0.4)
plt.plot(x,f_model.OLS_y, label='OLS estimate', alpha=0.7)
plt.plot(x,f_model.MLE_y, label='MLE estimate', alpha=0.7)
plt.plot(x,f_model.RLS_y, label='RLS estimate')
# plt.scatter(x,y,s=1.5,label='data points',marker='o')
plt.scatter(x,y_noise, label='noisey function',alpha=0.4)
# plt.ylim(-2.5,6)
plt.grid()
plt.legend()


# plt.figure()
# plt.plot(volumes, label='volumes')
# plt.grid()
# plt.legend()
plt.show()