import numpy as np
import matplotlib.pyplot as plt

def _DFT(x , N):
    result = np.zeros(N , dtype=complex)

    for m in range(N):
        for n in range(N):
            result[m] += x[n] * np.exp(-1j * 2 * np.pi * m * n / N)

    return result









f1 = 30
f2 = 75
f3 = 70 
fs = 100
T = 1/fs


N = 4 #number of samples

n = np.arange(N)

x = np.sin(2 * np.pi * f3/fs * n) 
x2= np.cos(2 * np.pi * f2/fs *n )


dft = _DFT(xn, N)


magnitude = np.abs(dft)
phase = np.angle(dft , deg=True)
real_part = np.real(dft)
imaginary_part = np.imag(dft)





# Plotting
plt.figure(figsize=(12,8))

#plotting the original values of x(n)
plt.subplot(3,2,1)
plt.plot(n,xn)
plt.stem(n , xn , linefmt="green")
plt.title("Values of x(n)")
plt.xlabel("n")
plt.ylabel("x(n)")

#plotting the magnitude of dft
plt.subplot(3,2,2)
plt.stem(n , magnitude)
plt.title("Magnitude of DFT")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

#plotting the phase of dft
plt.subplot(3,2,3)
plt.stem(n , phase)
plt.title("Phase of DFT")
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid(True)

#plotting the realpart
plt.subplot(3,2,4)
plt.stem(n , real_part)
plt.title("Real part")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)

#plotting the imaginary part
plt.subplot(3,2,5)
plt.stem(n , imaginary_part)
plt.title("imaginary part")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)

#plotting the original values of x(n)


plt.tight_layout()
plt.show()
