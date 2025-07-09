import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100  # Sampling frequency
f = 10    # Signal frequency
t = np.arange(0, 1, 1/fs)  # Time vector

# Corrected six-point moving average filter
def six_point(x):
    len_x = len(x)
    y = np.zeros_like(x, dtype=float)
    for n in range(len_x):
        sum_val = 0
        count = 0
        for i in range(6):
            if n - i >= 0:
                sum_val += x[n - i]
                count += 1
        y[n] = sum_val / count  # Avoid division by 0
    return y

# Generate clean sine wave
signal = np.sin(2 * np.pi * f * t)

# Add white noise
np.random.seed(0)  # For reproducibility
noise = np.random.normal(0, 0.5, len(t))  # mean=0, std=0.5
noisy_signal = signal + noise

# Apply six-point filter
signal_filter = six_point(noisy_signal)

# Plot all in one figure using subplots
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, signal, label="Original Clean Sine")
plt.title("Original Sine Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, noise, label="Noise", color='orange')
plt.title("Noise")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, noisy_signal, label="Noisy Signal", color='gray', alpha=0.7)
plt.title("Noisy Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, signal_filter, label="Filtered Signal (6-point avg)", color='green', linewidth=2)
plt.title("Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
