import numpy as np
import matplotlib.pyplot as plt

def unit_step(n):
    return np.where(n >= 0 , 1 , 0)

def sine(f1 , t):
    return np.sin(2 * np.pi * f1 * t)

def coss(f1 , t):
    return np.cos(2 *np.pi * f1 * t)

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


     

n = np.arange(0,10,1)
#define parameters
x=unit_step(n)
h=unit_step(n)-unit_step(n-5)


plt.figure(figsize=(12,8))

plt.subplot(2, 2, 1)
plt.stem(n,x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('x(n)')

plt.subplot(2, 2, 2)
plt.stem(n,h)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('h(n)')

# Convolution
y = convolve(x, h)

plt.subplot(2, 2, 3)
plt.stem(y)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('y(n) = x(n) * h(n)')
plt.tight_layout()
plt.show()

or, 

## 🐍 Python Code with Documentation
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the unit step function
def unit_step(n):
    """Generates unit step signal u(n)"""
    return np.array([1 if i >= 0 else 0 for i in n])

# Step 2: Define the range of n
n = np.arange(0, 11)  # We compute for 20 samples to see output behavior

# Step 3: Define x[n] = u(n)
x = unit_step(n)

# Step 4: Define h[n] = u(n) - u(n-5) => rectangular pulse of 5 samples
h = unit_step(n) - unit_step(n - 5)

# Step 5: Perform convolution: y[n] = x[n] * h[n]
y = np.convolve(x, h)

# Step 6: Define time index for output
n_y = np.arange(0, len(y))

# Step 7: Plot x[n], h[n], and y[n]
plt.figure(figsize=(14, 6))

plt.subplot(3, 1, 1)
plt.stem(n, x, basefmt=" ")
plt.title("Input x[n] = u(n)")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n, h, basefmt=" ")
plt.title("Impulse Response h[n] = u(n) - u(n-5)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(n_y, y)
plt.title("Output y[n] = x[n] * h[n]")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)

plt.tight_layout()
plt.show()

