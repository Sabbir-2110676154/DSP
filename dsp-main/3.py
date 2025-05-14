import numpy as np
import matplotlib.pyplot as plt
omega =[0,np.pi/4,np.pi/2]

n=np.arange(0,40)
plt.figure(figsize=(10,6))
for i in range (len(omega)):
  x=np.cos(omega[i]*n)
  plt.subplot(2,2,i+1)
  plt.stem(x)

or

import numpy as np
import matplotlib.pyplot as plt

# Sample indices
n = np.arange(0, 11, 1) 

# Frequencies
omega1 = 0.5 * np.pi
omega2 = np.pi

# Signals
x1 = np.cos(omega1 * n)
x2 = np.cos(omega2 * n)

# Plotting
plt.figure(figsize=(10, 4))

# ω = 0.5π
plt.subplot(1, 2, 1)
plt.stem(n, x1,label='low Ossillation at w=0.5*pi')
plt.title('ω = 0.5*pi')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)
plt.legend(loc='best')

# ω = π
plt.subplot(1, 2, 2)
plt.stem(n, x2,label='high Ossillation at w=pi')
plt.title('ω = pi')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.legend(loc='best')
plt.tight_layout()
plt.show()
