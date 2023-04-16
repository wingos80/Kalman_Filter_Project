import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x

t = 0
dt = 1
x0 = 1
x0_si = x0


N = 5
x = []
x_exact = []
x_si = []

x.append(x0)
x_exact.append(np.exp(t))
x_si.append(x0_si)
for i in range(N):
    t+=dt
    x_exact.append(np.exp(t))
    
    x0 = x0+ dt*f(x0)
    x0_si_temp = x0_si + dt*f(x0_si)
    x0_si = x0_si + dt*f(x0_si_temp)

    
    x.append(x0)
    x_si.append(x0_si)


plt.plot(x, label='forward euler simple')
plt.plot(x_exact, label='exact')
plt.plot(x_si, label='si euler')
plt.grid()
plt.legend()
plt.show()
