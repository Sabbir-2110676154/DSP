import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
def  sin(a,f,v):
    return a*np.sin(2 * np.pi * f * v)
def  cos(a,f,v):
    return a*np.cos(2 * np.pi * f * v)
def arrange(v,fs):
    return np.arange(0,v,1/fs)


def signal(t):
    x1=cos(3,100,t)
    x2=sin(5,300,t)
    x3=cos(10,600,t)
    return x1+x2+x3



#add title labels

t=arrange(0.008,100000)
xt=signal(t)
plt.figure(figsize=(12,6))
plt.subplot(3,2,1)
plt.plot(t,xt)

fs=1200#......nyquist
n=arrange(0.008,fs)

xn=signal(n)

plt.subplot(3,2,2)
plt.plot(n,xn)
fs=900 #......undersample
n=arrange(0.008,fs)
xn=signal(n)
plt.subplot(3,2,3)
plt.plot(n,xn)

fs=1400 #......oversample
n=arrange(0.008,fs)
xn=signal(n)
plt.subplot(3,2,4)
plt.plot(n,xn)



plt.tight_layout()
plt.show()


or,


import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
A1, A2, A3 = 3, 5, 10
f1, f2, f3 = 1000, 3000, 6000  # Frequencies in Hz
t_max = 0.005  # short time window

# Define sampling rates
sampling_rates = [16000, 12000, 8000]  # >Nyquist, Nyquist, <Nyquist

# High-resolution time vector (for original signal)
t_cont = np.linspace(0, t_max, 10000)
x_cont = A1 * np.cos(2*np.pi*f1*t_cont) + A2 * np.sin(2*np.pi*f2*t_cont) + A3 * np.cos(2*np.pi*f3*t_cont)

# Plotting
plt.figure(figsize=(12, 8))

# Plot original continuous signal
plt.subplot(4, 1, 1)
plt.plot(t_cont, x_cont)
plt.title('Original Analog Signal')
plt.ylabel('Amplitude')
plt.grid(True)


high_t = np.linspace(0, t_max, int(16000*t_max))
high_sampled = A1*np.cos(2*np.pi*f1*high_t)+ A2 * np.sin(2*np.pi*f2*high_t) + A3 * np.cos(2*np.pi*f3*high_t)

plt.subplot(4,1,2)
plt.stem(high_t,high_sampled)

high_t = np.linspace(0, t_max, int(16000*t_max))
high_sampled = A1*np.cos(2*np.pi*f1*high_t)+ A2 * np.sin(2*np.pi*f2*high_t) + A3 * np.cos(2*np.pi*f3*high_t)




# Sample and plot for each sampling rate
# for i, fs in enumerate(sampling_rates):
#     n = np.arange(0, t_max, 1/fs)  # sample indices
#     x_sampled = A1 * np.cos(2*np.pi*f1*n) + A2 * np.sin(2*np.pi*f2*n) + A3 * np.cos(2*np.pi*f3*n)

#     plt.subplot(len(sampling_rates)+1, 1, i+2)
#     plt.stem(n, x_sampled)
#     plt.title(f'Sampled Signal at {fs} Hz')
#     plt.ylabel('Amplitude')
#     plt.grid(True)

# plt.xlabel('Time [s]')
# plt.tight_layout()
plt.show()

