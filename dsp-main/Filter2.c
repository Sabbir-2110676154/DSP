import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
x = np.array([1, 2, 2, 10, 2, 2, 1])
#alpha = np.linspace(0, 1, 5)
alpha = np.array([0.2])
for a in alpha:
    y = [x[0]]
    for n in range(1, len(x)):
        cal = (1 - a) * y[n - 1] + a * x[n]
        y.append(cal)
    
    plt.plot(y,label='y[n]',color='blue')
plt.plot(x, label='x[n]', color='gray')
plt.title('Effect of Alpha')
plt.grid(True)
plt.xlabel('n and X[i]')
plt.ylabel('X[n] and Y[i]')
plt.legend()
plt.show()
