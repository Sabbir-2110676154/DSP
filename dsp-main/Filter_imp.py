import numpy as np
import matplotlib.pyplot as plt




def convolve(x, h):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1
    y = [0] * len_y

    for i in range(len_y):
        for k in range(len_x):
            if(i - k >= 0 and i - k < len_h):
                y[i] += x[k] * h[i - k]

    return y


     

fs = 500
n = np.arange(0, 1, 1/fs)

x = np.sin(2 * np.pi * 10 * n) + 0.5 * np.sin(2 * np.pi * 100 * n)
fc = 20

# Number of filter coefficients
N = 21
fc_norm=fc/(fs/2)

n1 = np.arange(-(N-1)//2,(N-1)//2)
h=fc_norm*np.sinc(fc_norm*n1)
# list of coefficients




plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(n, x)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('x(t) = sin(2π.50t + 0.5sin(2π.200t))')
plt.subplot(2, 2, 3)
plt.stem(n1,h)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('h')
y = convolve( x,h)

len_x = len(x)
len_h = len(h)
len_y = len_x + len_h - 1
n=np.arange(0,len_y,1)
plt.subplot(2, 2, 2)
plt.plot(n,y)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('y(n) = fir_filter(x(n))')



plt.tight_layout()
plt.show()
