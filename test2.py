import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from KF_Functions import *
sns.set(style = "darkgrid")                  # Set seaborn style    
dx = 0.02
x = np.arange(0,5,dx)
y = x**3/3 + np.random.normal(0,0.001,len(x))

# find the derivatives of y
dy, ddy = kf_finite_difference(dx,y,step_size=25)

actual_dy = x**2
actual_ddy = 2*x

plt.figure()
plt.plot(x, y, label=r'f(x) = $x^2 + \mathcal{N} (0, 10^{-6})$')
plt.plot(x, dy, label='numerical f\'(x)',color='C2')
plt.plot(x, ddy, label='numerical f\"(x)',color='C3')

plt.plot(x, actual_dy, label='actual f\'(x)', alpha=0.5,color='C2')
plt.plot(x, actual_ddy, label='actual f\"(x)', alpha=0.5,color='C3')
# remove the y and x axis ticks
plt.xlabel('time')
plt.ylabel('[-]')
plt.tight_layout()
plt.legend()
plt.show()
