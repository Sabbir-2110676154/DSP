import numpy as np
import matplotlib.pyplot as plt

def unit_step_signal(n):
    return np.where(n >= 0 , 1 , 0)

def ramp_signal(n):
    return np.where(n >= 0 , n , 0)

def exponential_signal(a,n):
    return pow(a,n)

def sine_signal(f1 , t):
    return np.sin(2 * np.pi * f1 * t)

def cosine_signal(f1 , t):
    return np.cos(2 *np.pi * f1 * t)

#define parameters
n = np.arange(-10,10,1)

step_signal = unit_step_signal(n)
ramp = ramp_signal(n)
exponential = exponential_signal(1.5,n)


plt.figure(figsize=(12,8))

plt.subplot(3,2,1)
plt.stem(n,step_signal)
plt.xlabel("n")
plt.ylabel("x(n)")
plt.title("Unit Step Signal")

plt.subplot(3,2,2)
plt.stem(n,ramp)
plt.xlabel("n")
plt.ylabel("x(n)")
plt.title("Unit Ramp Signal")

plt.subplot(3, 2, 3)
plt.stem(n, exponential)
plt.title("Exponential Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

or,

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42) #FIXED random value
# Generate Unit Step Sequence
n = np.arange(-10, 11, 1)  # Range from -10 to 10 increment 1
u_n = np.where(n >= 0, 1, 0)  # Unit step: 1 if n >= 0, else 0
print(n)
print(u_n)

#plt.plot(n,u_n)
plt.stem(n,u_n,label="Unit Step Seq")
plt.xlabel("n",color='red')
plt.ylabel("u_n")
plt.legend()
plt.grid()

plt.show()



n=np.arange(-10,11,1)
ramp=np.where(n<0,0,n)

plt.scatter(n,ramp)
plt.stem(n,ramp,label="Ramp Sequence")
plt.xlabel("n")
plt.ylabel("r(n)")
plt.legend(loc='best')
plt.grid(True)
plt.show()



# Parameters
A = 1      # Amplitude
b = 0.5    # Growth rate (decay for negative values)
t = np.linspace(0, 10, 50)  # Time from 0 to 10, 500 points
# Exponential function
y = A * np.exp(b * t)

# Plotting
#plt.plot(t, y)
plt.stem(t,y)
plt.title('Exponential Growth')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()

x=np.linspace(0,2*np.pi,1000) # 1000 points between 0 and 2π
A=4

y1=np.sin(x) #Sine Wave
y2=np.cos(x) #Cos Wave

plt.figure(figsize=(15,4)) #Figure Size

#ax1=fig.add_subplot(1,2,1) #
#ax1.plot(x,y1,color='green' , label='Sine wave')

plt.subplot(2,2,1)
plt.plot(x,y1,color='green' , label='Sine wave')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(loc='best')
plt.axhline(y=0,color='black') #horizontal line
plt.axvline(x=0,color='red') #vertical line
plt.title("Sine Wave")


#ax2=fig.add_subplot(1,2,2)
#ax2.plot(x,y2,color='green' , label="Cos Wave")
plt.subplot(2,2,4)
plt.plot(x,y2,color='green' , label="Cos Wave")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(loc='best')
plt.axhline(y=0,color='green') #horizontal line
plt.axvline(x=0,color='red') #vertical line
plt.title("Cos Wave")

plt.show() #Render Plot
