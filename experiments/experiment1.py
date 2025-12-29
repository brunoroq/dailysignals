import numpy as np
"""
generators for discrete-time signals
"""

def n_range(n_min, n_max):
    return np.arange(n_min, n_max+1)

def unit_step(n, n0=0):
    return (n>=n0).astype(float)

def exp_decay(n, a=0.9, n0=0):
    u = unit_step(n, n0)
    return (a**(n - n0)) * u

"""
Convolution of discrete-time signals.
x: input signal.
n[x]: time index of input signal.
h: impulse response.
n[h]: time index of impulse response.

It returns:
y: output signal, where y[n] = x[n] * h[n] (convolution).
"""

def convolution(x, nx, h, nh):
    ny=np.arange(nx[0]+nh[0], nx[-1]+nh[-1]+1)
    y=np.zeros_like(ny, dtype=float)
    #Diccionarios para mapear.
    x_map={n: xv for n, xv in zip(nx, x)}
    h_map={n:hv for n, hv in zip(nh, h)}
    
    for i, n in enumerate(ny):
        s=0.0
        for k in nx:
            #y[n]+=x[k]h[n-k]
            s+=x_map.get(k,0.0)*h_map.get(n-k, 0.0)
        y[i]=s
    return y, ny

