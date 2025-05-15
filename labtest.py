import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 2, 10, 2, 2, 1])

def compute_y(x, a):
    y = np.zeros_like(x, dtype=float)
    for n in range(1, len(x)):
        y[n] = (1 - a) * y[n - 1] + a * x[n]
    return y


alphas = [0.1, 0.5, 0.9]
plt.figure(figsize=(10, 6))

for a in alphas:
    y = compute_y(x, a)
    plt.plot(y, marker='o', label=f'a = {a}')

plt.plot(x, 'k--', label='x(n)', marker='x')  # Original input signal
plt.title('Output y(n) for different values of a')
plt.xlabel('n')
plt.ylabel('y(n)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
