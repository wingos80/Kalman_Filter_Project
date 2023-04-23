import numpy as np
import matplotlib.pyplot as plt
from KF_Functions import *

dt = 0.5
x = np.arange(0,10,dt)
y = x**3+0.5*x**2+np.sin(x)+np.exp(-x)
#  testing some finite dfference stuff
Xs = np.zeros((10,y.size))
Xs[:] = y

# b = (np.diff(a, n=1)[:-1] + np.diff(a, n=1)[1:])/(2*dx)
# c = (np.diff(b, n=1)[:-1] + np.diff(b, n=1)[1:])/(2*dx)
# d = 2/(dx**2)*(np.diff(a[1:], n=1)-b*dx)

# Note, the second deriviatives of the angles will only be calculated for the values from the indices [1:-1] of the angle arrays!!!
rate_dots_X = np.zeros_like(Xs[6:9,2:-2])                                                     # angular accceleations in body frame [rad/s^2]

_, rate_dots_X = finite_difference(dt, Xs[6:9])                                       # second derivative of the flight angles, from finite difference


first = 3*x[1:-1]**2
second_a = 6*x[2:-2]
second_b = 6*x+1-np.sin(x)+np.exp(-x)

# print(f'error in the second derivative: {rate_dots_X[0]-second_b[1:-1]}')

# plot the errors on a log graph
fig, ax = plt.subplots()
# ax.plot(x, np.abs(rate_dots_X[0]-second_b), label='error in second derivative')
# ax.set_yscale('log')
ax.plot(x, second_b, label='second derivative')
ax.plot(x, rate_dots_X[0], label='second derivative from finite difference')
ax.legend()
ax.grid()
plt.show()



# # plot a and b together
# fig, ax = plt.subplots()
# ax.plot(x, a, label='a')
# ax.plot(x[1:-1],b, label='b')
# ax.plot(x[2:-2],c, label='c')
# ax.plot(x[1:-1],d, label='d')
# ax.plot(x[1:-1],3*x[1:-1]**2, label='analytical derivative of x cubed')
# ax.plot(x[2:-2],6*x[2:-2], label='analytical derivative of 3x squared')
# ax.legend()
# ax.grid()
# plt.show()

