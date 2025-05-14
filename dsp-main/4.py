import numpy as np
import matplotlib.pyplot as plt


     

t = np.arange(0, 1, 0.001)
x = 3*np.cos(2 * np.pi * 50 * t)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('x(t) = 3cos(2π.50t)')



fs = 200
n = np.arange(0, 1, 1/fs)
x = 3*np.cos(2 * np.pi * 50 * n)

plt.subplot(2, 2, 2)
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('200hz)')
fs = 75
n = np.arange(0, 1, 1/fs)
x_n = np.sin(2 * np.pi * 25 * n)

plt.subplot(2, 2, 3)
plt.stem(n, x_n)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('fs=75')


plt.tight_layout()
plt.show()

or,

import numpy as np
import matplotlib.pyplot as plt

# Analog signal parameters
A = 3
f_analog = 50  # Because 100π = 2πf → f = 50 Hz
t_max = 0.2  # Time duration for visualization

# Sampling rates
fs1 = 200  # Hz
fs2 = 75  # Hz

# Time vectors for sampled signals
n1 = np.arange(0, t_max, 1/fs1)
n2 = np.arange(0, t_max, 1/fs2)

# Sampled signals
x1 = A * np.cos(2 * np.pi * f_analog * n1)  # 200 Hz sampling
x2 = A * np.cos(2 * np.pi * f_analog * n2)  # 75 Hz sampling

# Plotting
plt.figure(figsize=(10, 5))

# 200 Hz plot
plt.subplot(2, 1, 1)
plt.stem(n1, x1)
plt.title("Sampled Signal at 200 Hz (No Aliasing)")
plt.ylabel("x[n]")
plt.grid(True)

# 75 Hz plot
plt.subplot(2, 1, 2)
plt.stem(n2, x2)
plt.title("Sampled Signal at 75 Hz (Aliasing Occurs)")
plt.xlabel("Time [s]")
plt.ylabel("x[n]")
plt.grid(True)

plt.tight_layout()
plt.show()
