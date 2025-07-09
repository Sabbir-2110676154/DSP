import numpy as np
import matplotlib.pyplot as plt

fs = 100
f = 10  
t1 = np.arange(0, 1, 1/fs)  
x1 = np.sin(2 * np.pi * f * t1)

plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 1)
plt.plot(t1, x1, marker='o')
plt.title("10 Hz Sine Wave (1 second, sampled at 100 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

X1 = np.fft.fft(x1)
freq = np.fft.fftfreq(len(x1), 1/fs)

plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 2)
plt.stem(freq[:len(freq)//2], np.abs(X1)[:len(freq)//2], basefmt=" ")
plt.title("DFT Magnitude Spectrum (1 second)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

t2 = np.arange(0, 0.95, 1/fs)  
x2 = np.sin(2 * np.pi * f * t2)

plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 3)
plt.plot(t2, x2, marker='o')
plt.title("10 Hz Sine Wave (0.95 second, non-integer cycles)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

X2 = np.fft.fft(x2)
freq2 = np.fft.fftfreq(len(x2), 1/fs)

plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 4)
plt.stem(freq2[:len(freq2)//2], np.abs(X2)[:len(freq2)//2], basefmt=" ")
plt.title("DFT Magnitude Spectrum (0.95 second)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

window = np.hamming(len(x2))
x2_win = x2 * window

plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 5)
plt.plot(t2, x2_win, marker='o')
plt.title("Windowed 10 Hz Sine Wave (Hamming Window, 0.95 second)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

X2_win = np.fft.fft(x2_win)

plt.figure(figsize=(10, 4))
plt.stem(freq2[:len(freq2)//2], np.abs(X2_win)[:len(freq2)//2], basefmt=" ")
plt.title("DFT Magnitude Spectrum (Hamming Window Applied)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
