import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100  # Sampling frequency (Hz)
f = 5     # Sine wave frequency (Hz)
t = np.arange(0, 2, 1/fs)  # Time vector (2 seconds duration)
signal = np.sin(2 * np.pi * f * t)  # Original sine wave

# Delay Signal
delay_samples = 10
delayed_signal = np.zeros_like(signal)
delayed_signal[delay_samples:] = signal[:-delay_samples]  # Shift by 10 samples

# Cross-correlation
correlation = np.correlate(delayed_signal, signal, mode='full')
lags = np.arange(-len(signal) + 1, len(signal))  # Lags corresponding to correlation

# Detect delay (peak in correlation)
peak_index = np.argmax(correlation)
estimated_delay = lags[peak_index]

print(f"Estimated Delay (in samples): {estimated_delay}")

# Plot
plt.figure(figsize=(12, 10))

# Plot Original and Delayed Signal
plt.subplot(3, 1, 1)
plt.plot(t, signal, label="Original Signal")
plt.plot(t, delayed_signal, '--', label=f"Delayed Signal ({delay_samples} samples)")
plt.title("Original and Delayed Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Plot Cross-Correlation
plt.subplot(3, 1, 2)
plt.plot(lags, correlation)
plt.title("Cross-Correlation")
plt.xlabel("Lag [samples]")
plt.ylabel("Correlation")
plt.grid(True)

# Plot Zoomed Cross-Correlation Near Peak
plt.subplot(3, 1, 3)
plt.plot(lags, correlation)
plt.title("Cross-Correlation (Zoom near Peak)")
plt.xlim(estimated_delay - 20, estimated_delay + 20)
plt.axvline(estimated_delay, color='r', linestyle='--', label=f'Estimated Delay: {estimated_delay}')
plt.xlabel("Lag [samples]")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
