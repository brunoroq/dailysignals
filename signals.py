import numpy as np

def n_vector(N):
    return np.arange(N)

def t_vector(N,fs):
    n=n_vector(N)
    return n/fs

def cosine(A, f, fs, N, phase=0.0):
    t=t_vector(N,fs)
    return A*np.cos(2*np.pi*f*t+phase)

def sine(A, f, fs, N, phase=0.0):
   t= t_vector(N,fs)
   return A*np.sin(2*np.pi*f*t+phase)

def cmplx_exp(A, w0, N):
    n=np.arange(N)
    return A* np.exp(1j*w0*n)

def unit_step(n0, N):
    n=np.arange(N)
    return (n>=n0).astype(float)
