import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Define signals
a = np.sin(2 * np.pi * 10 * t)               # 10 Hz sine wave
b = np.sign(np.sin(2 * np.pi * 10 * t))      # 10 Hz square wave
c = np.sin(2 * np.pi * 20 * t)               # 20 Hz sine wave

# Auto-correlation of a
auto_a = correlate(a, a, mode='full')

# Cross-correlation between a and b
cross_ab = correlate(a, b, mode='full')

# Cross-correlation between a and c
cross_ac = correlate(a, c, mode='full')

lags = np.arange(-len(t)+1, len(t))

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(lags, auto_a)
plt.title('Auto-correlation of a')

plt.subplot(1, 3, 2)
plt.plot(lags, cross_ab)
plt.title('Cross-correlation: a & b')

plt.subplot(1, 3, 3)
plt.plot(lags, cross_ac)
plt.title('Cross-correlation: a & c')

plt.tight_layout()
plt.show()
