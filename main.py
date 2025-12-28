import numpy as np
import matplotlib.pyplot as plt

from signals import (n_vector, t_vector, cosine, sine, cmplx_exp, unit_step)

N=100
fs=100

n=n_vector(N)
t=t_vector(N,fs)

x1=cosine(1.0,5.0,fs,N)
x2=sine(1.0,5.0,fs,N)

w0=0.25*np.pi
z=cmplx_exp(1.0, w0, N)

u1=unit_step(-50, N)
u2=unit_step(20,N)
plt.legend()

w=u1-u2-u2

y1=x1*w+w
y2=x2*w+w

plt.plot(t, x1, label="cos")
plt.plot(t,x2, label="sen")
plt.xlabel("t[s]")
plt.title("Sen/Cos")
plt.legend()
plt.grid(True)

plt.figure()
plt.stem(n,x2)
plt.xlabel("n")
plt.title("Discrete sine")
plt.grid(True)

plt.figure()
plt.stem(n,x1)
plt.xlabel("n")
plt.title("Discrete cosine")
plt.grid(True)

plt.figure()
plt.stem(n, u1)
plt.title("unit step 1")
plt.grid(True)

plt.figure()
plt.stem(n, u2)
plt.title("unit step 2")
plt.grid(True)

plt.figure()
plt.plot(t, y1)
plt.xlabel("t [s]")
plt.title("Coseno recortado por ventana")
plt.grid(True)

plt.figure()
plt.plot(t,y2)
plt.xlabel("t[s]")
plt.title("Seno recortado por ventana")
plt.grid(True)

plt.show()



