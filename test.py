import numpy as np
import seaborn as sns
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *

a = np.arange(10)
b = np.array([a, a, a]).T
c = np.array([1,2,4])
b = b*c
print(b)   
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(1 * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins)**2 / (2 * 1**2) ),
#          linewidth=2, color='r')
# plt.show()