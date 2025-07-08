import numpy as np
import matplotlib.pyplot as plt

# Parameter for continuous signal
fs = 500 
# frequency 10 Hz
n = np.arange(0,1,1/fs)

# time vector for continuous time signal
x = np.sin(2 * np.pi * 5 * n) + 0.5 * np.cos(2*np.pi *50*n) # continuous signal

# Plotting
plt.figure(figsize=(16 , 6))

# Demostrating for High Sampling frequency
plt.subplot(1,2,1)
plt.plot(n, x )

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.grid(True)
#alpha = np.linspace(0.2, 1, 3)

alpha = np.array([0.05])
plt.subplot(1, 2, 2)
for a in alpha:
    y = [x[0]]
    for n in range(1, len(x)):
        cal = (1 - a) * y[n - 1] + a * x[n]
        y.append(cal)
    
    plt.plot(y)
plt.plot(x)

plt.grid(True)

plt.tight_layout()

plt.show()
