import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "darkgrid")                  # Set seaborn style    
x = np.arange(0.1,10,0.1)
y = np.exp(-x)

plt.figure()
plt.plot(x,y)
# remove the y and x axis ticks
plt.xticks([])
plt.yticks([])
plt.xlabel('time')
plt.ylabel('variance')
plt.tight_layout()
plt.show()
