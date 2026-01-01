import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Importa utilidades de experiment2 (te sugiero ponerlas ahí)
from experiments.experiment2 import (
    read_wav_mono, write_wav_int16_mono, normalize_peak,
    ir_echo, ir_exp_decay,
    conv_direct
)

# ----------------------------
# Parámetros
# ----------------------------
AUDIO_REL = os.path.join("wavs", "1.wav")

# Para que la animación sea razonable (no 44k frames):
N_VIS = 2500          # cantidad de muestras visibles del x (≈ 56 ms a 44.1k)
MAX_FRAMES = 400      # cuantos n animar (si quieres animar todo: pon None)
INTERVAL_MS = 30      # velocidad

USE_ECHO = True

# ----------------------------
# Carga WAV
# ----------------------------
base_dir = os.path.dirname(__file__)  # carpeta signals/
wav_in = os.path.join(base_dir, AUDIO_REL)

x_full, fs = read_wav_mono(wav_in)

# Recorte para visualización (animación)
x = x_full[:N_VIS].astype(np.float32)
nx = np.arange(len(x), dtype=int)

# IR corta (importante para conv directa)
if USE_ECHO:
    h = ir_echo(fs, delay_ms=25.0, gain=0.6)   # eco corto y claro
else:
    h = ir_exp_decay(fs, T=0.05, a=0.995)      # cola corta

nh = np.arange(len(h), dtype=int)

# Convolución "real" (para comparar y para guardar audio convolved completo)
y_true_full = conv_direct(x_full, h)
y_true_full = normalize_peak(y_true_full, 0.98)

# Convolución del trozo para animar
y_true, ny = conv_direct(x, h), np.arange(len(x) + len(h) - 1)
y = np.zeros_like(y_true, dtype=np.float32)

ymax = 1.05 * float(np.max(np.abs(y_true))) if y_true.size else 1.0

# Define cuántos frames animar
n_frames = len(ny) if MAX_FRAMES is None else min(MAX_FRAMES, len(ny))

# ----------------------------
# Figura (columna)
# ----------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
ax_x, ax_h, ax_prod, ax_y = axs
for ax in axs:
    ax.grid(True)

# x[k] fijo
ax_x.stem(nx, x, basefmt='b-', markerfmt='bo')
ax_x.set_title("x[k] (trozo del WAV)")
ax_x.set_xlim(nx[0], nx[-1])

def update(frame):
    n = ny[frame]

    # k_h: índices en k donde cae h[n-k]
    k_h = n - nh
    h_shift_vals = h  # valores no cambian, solo se mueven los índices

    # producto sobre eje nx (k = nx)
    prod = np.zeros_like(nx, dtype=np.float32)

    # mapa k -> h[n-k]
    h_map = dict(zip(k_h, h_shift_vals))

    for i, k in enumerate(nx):
        prod[i] = x[i] * h_map.get(k, 0.0)

    y[frame] = float(np.sum(prod))

    # limpiar ejes variables
    ax_h.cla(); ax_prod.cla(); ax_y.cla()
    for ax in (ax_h, ax_prod, ax_y):
        ax.grid(True)

    # h[n-k]
    ax_h.stem(k_h, h_shift_vals, basefmt='C1-', markerfmt='C1o')
    ax_h.set_title(f"h[n-k], n={n}")
    ax_h.set_xlim(nx[0], nx[-1])

    # producto
    ax_prod.stem(nx, prod, basefmt='g-', markerfmt='go')
    ax_prod.set_title("x[k] · h[n-k]")
    ax_prod.set_xlim(nx[0], nx[-1])
    ax_prod.set_ylim(-1.05*np.max(np.abs(x)), 1.05*np.max(np.abs(x)) + 1e-6)

    # y acumulado
    ax_y.stem(ny[:frame+1], y[:frame+1], basefmt='r-', markerfmt='ro')
    ax_y.set_title(f"y[n] acumulado, n={n}")
    ax_y.set_xlim(ny[0], ny[min(len(ny)-1, n_frames-1)])
    ax_y.set_ylim(-ymax, ymax)

    return ()

ani = FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=INTERVAL_MS,
    repeat=False
)

plt.tight_layout()
plt.show()

# ----------------------------
# Guardar audio convolved completo
# ----------------------------
out_name = "1_conv_echo.wav" if USE_ECHO else "1_conv_decay.wav"
wav_out = os.path.join(base_dir, "wavs", out_name)
write_wav_int16_mono(wav_out, y_true_full, fs)
print("Escrito:", wav_out)
