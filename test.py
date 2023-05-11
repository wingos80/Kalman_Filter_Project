import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *


x = np.arange(0,11,0.1)
y = 1+2*x+0.2*x**2
y_noise = y+np.random.normal(0,2,y.size)

f_model = model(parameters={'const':np.ones_like(x), 'x':x, 'x^2':x**2})
f_model.measurements = y_noise
f_model.OLS_estimate()


plt.plot(x,y, label='original function')
plt.plot(x,y_noise, label='noisey function')
plt.plot(x,f_model.OLS_y, label='OLS estimate')
plt.scatter(x,y)
plt.grid()
plt.legend()
plt.show()