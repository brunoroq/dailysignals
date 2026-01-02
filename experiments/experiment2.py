import os
import numpy as np
import wave

#-------------------------
#WAV I/O
#-------------------------
def read_wav_mono(path: str):
    """
    Lee el WAV PCM (8/16/24/32-bit int). retorna (x, fs)
    x: float32 mono en [-1,1]
    si es stereo, mezcla a mono promediando los inputs.
    """
    with wave.open(path, "rb") as wf:
        fs=wf.getframerate()
        n_channels=wf.getnchannels()
        sampwidth=wf.getsampwidth() #bytes
        n_frames=wf.getnframes()
        raw=wf.readframes(n_frames)
        
        if sampwidth==1:
                #unsigned 8-bit
                x=np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                x=(x-128.0)/128.0
        elif sampwidth==2:
                x=np.frombuffer(raw,dtype=np.init16).astype(np.float32)
                x=x/32768.0
        elif sampwidth==3:
                #24-bit PCM: parse manual
                b=np.frombuffer(raw, dtype=np.uint8).reshape(-1,3)
                x=(b[:,0].astype(np.int32) |
                (b[:,1].astype(np.int32)<<8) |
                (b[:,2].astype(np.int32)<<16))
                sign=(x & 0x80000)!=0
                x=x-(sign.astype(np.int32)<<24)
                x=x.astype(np.float32)/8388608.0
        elif sampwidth==4:
                x=np.frombuffer(raw, dtype=np.init32).astype(np.float32)
                x=x/2147483648.0
        else:
                raise ValueError(f"unsupported sample witdh: {sampwidth} bytes")
        if n_channels>1:
                x=x.reshape(-1, n_channels).mean(axis=1)
        return x.astype(np.float32),fs
    
def write_wav_int16_mono(path:str, x: np.ndarray, fs:int):
    """
    Escribe WAV mono 16-bit PCM. x float en [-1,1](Se recorta).
    """        
    x=np.asarray(x, dtype=np.float32)
    x=np.clip(x,-1.0,1.0)
    y=(x*32767.0).astype(np.int16)
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(y.tobytes())
        
def normalize_peak(x: np.ndarray, peak=0.98):
    m=float(np.max(np.abs(x))) if x.size else 0.0
    if m<1e-12:
        return x
    return (peak/m)*x

#-----------------
#convolucion
#-----------------

def conv(x: np.ndarray, h:np.ndarray):
    """
    Convolucion lineal directa: y[n]=sum_k x[n]h[n-k]
    Complejidad O(NM). perfecta si h es corta (eco/cola corta).
    """
    x=np.asarray(x, dtype=np.float32)
    h=np.asarray(h, dtype=np.float32)
    N, M = x.size, h.size
    y=np.zeros(N+M -1, dtype=np.float32)
    
    for n in range(N+M-1):
            s=0.0
            kmin=max(0,n-(M-1))
            kmax=min(n,N-1)
            for k in range (kmin,kmax+1):
                s+=x[k]*h[n-k]
                y[n]=s
    return y

def ir_echo(fs:int, delay_ms=120.0, gain=0.6):
    """
    h[n]=delta[n]+gain*delta[n-D]
    """
    D=int(round((delay_ms/1000.0)*fs))
    h=np.zeros(D+1, dtype=np.float32)
    h[0]=1.0
    h[D]=float(gain)
    return h

def ir_exp_decay(fs:int, T=0.20, a=0.995):
        """
        h[n]=a^n u[n], truncada a T segundos.
        puede subir mucho el volumen, normalizamos por suma.
        """
        M=int(round(T*fs))
        n=np.arange(M,dtype=np.float32)
        h=(a**n).astype(np.float32)
        #Normalizacion simple.
        h=h/max(np.sum(h),1e-12)
        return h
    

    