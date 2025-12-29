import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from experiments.experiment1 import(n_range, unit_step, exp_decay, convolution)

def update(frame):
    n = ny[frame]

    k_h = n - nh
    h_shift_vals = h.copy()

    prod = np.zeros_like(nx, dtype=float)
    h_map = dict(zip(k_h, h_shift_vals))

    for i, k in enumerate(nx):
        prod[i] = x[i] * h_map.get(k, 0.0)

    y[frame] = np.sum(prod)

    ax_h.cla(); ax_prod.cla(); ax_y.cla()

    for ax in (ax_h, ax_prod, ax_y):
        ax.grid(True)

    ax_h.stem(k_h, h_shift_vals, basefmt='C1-', markerfmt='C1o')
    ax_h.set_title(f"h[n-k], n={n}")
    ax_h.set_xlim(nx[0], nx[-1])

    ax_prod.stem(nx, prod, basefmt='g-', markerfmt='go')
    ax_prod.set_title("x[k] · h[n-k]")
    ax_prod.set_xlim(nx[0], nx[-1])

    ax_y.stem(ny[:frame+1], y[:frame+1], basefmt='r-', markerfmt='ro')
    ax_y.set_title(f"y[n] acumulado, n={n}")
    ax_y.set_xlim(ny[0], ny[-1])
    ax_y.set_ylim(0, ymax)   # si todo es positivo

    return ()


            
nx=n_range(-20,80)
nh=n_range(-20,80)

a=0.9
x=exp_decay(nx, a=a, n0=0) #a^n u[n]
h=unit_step(nh,n0=0) # u[n]

y, ny = convolution(x, nx, h, nh)

fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
ax_x, ax_h, ax_prod, ax_y = axs


for ax in axs.flat:
    ax.grid(True)
    
# x[k]
stem_x=ax_x.stem(nx, x, basefmt='b-', markerfmt='bo')
ax_x.set_title("x[k]")
# h[n-k]
stem_h=ax_h.stem(nx, np.zeros_like(nx), basefmt='C1', markerfmt='C1o')
ax_h.set_title("h[n-k]")
#Product
stem_p=ax_prod.stem(nx, np.zeros_like(nx), basefmt='g-', markerfmt='go')
ax_prod.set_title("x[k]· h[n-k]")
#y[n]
stem_y=ax_y.stem(ny, y, basefmt='r-', markerfmt='ro')
ax_y.set_title("y[n]=x*h")

y_true, ny = convolution(x, nx, h, nh)
y = np.zeros_like(y_true)
ymax = 1.05 * np.max(np.abs(y_true))


ani= FuncAnimation(
    fig,
    update,
    frames=len(ny),
    interval=100,
    repeat=False
    )
plt.show()