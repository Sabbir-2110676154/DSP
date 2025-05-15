import numpy as np
import matplotlib.pyplot as plt



    
    
x=[1,2,2,10,2,2,1]


plt.figure(figsize=(12,6))
plt.subplot(4,2,1)
plt.stem(x)
plt.title("input original signal")
y=[0]*7
a=0.7
y[0]=a*x[0]

for i in range (1,len(x)):
    y[i]=y[i-1]*(1-a)+a*x[i]
plt.subplot(4,2,2)
plt.plot(y)
plt.title("alpha 0.7")
y=[0]*7
a=0.5
y[0]=a*x[0]

for i in range (1,len(x)):
    y[i]=y[i-1]*(1-a)+a*x[i]
plt.subplot(4,2,3)
plt.plot(y)
plt.title("alpha 0.5")
y=[0]*7
a=0.3
y[0]=a*x[0]

for i in range (1,len(x)):
    y[i]=y[i-1]*(1-a)+a*x[i]
plt.subplot(4,2,4)
plt.plot(y)
plt.title("alpha 0.3")


w=np.pi/8
wn=7*w
n=np.arange(0,40)
x4=np.cos(w*n)
x5=np.cos(w*n)
plt.subplot(4,2,5)
plt.plot(x4)
plt.title("low cosine")
plt.subplot(4,2,6)
plt.plot(x5)
plt.title("low cosine")
yn2=[0]*40
a=0.7
yn2[0]=a*x5[0]

for i in range (1,len(x5)):
    yn2[i]=yn2[i-1]*(1-a)+a*x5[i]
plt.subplot(4,2,8)
plt.plot(yn2)
plt.title("high a=0.7")
yn=[0]*40
a=0.3
yn[0]=a*x4[0]

for i in range (1,len(x4)):
    yn[i]=yn[i-1]*(1-a)+a*x4[i]
plt.subplot(4,2,7)
plt.plot(yn)
plt.title("low a=0.3")
plt.tight_layout()
plt.show()
