import numpy as np
import matplotlib.pyplot as plt
from math import pi as p

# amplitude sampling
t= np.arange(0,0.5,0.0005)
Am = eval(input("enter the message signal amplitude: "))
Ac = eval(input("enter the carrier signal amplitude: "))
          
fm = eval(input("enter the message signal frequency: "))
# carrier frequency
fs = eval(input("enter the carrier signal frequency: "))

#t1= np.arange(0,0.5,1/fs)
#print(t1)
#x = np.cos(2*p*fm*t)
s = Ac*np.cos(2*p*fs*t)* Am*np.cos(2*p*fm*t)
plt.plot(t,s,"g")
#plt.stem(t1,s,"r")
plt.grid(axis='both')
plt.show()
