import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

STEP = 200
MULTI = 34
RADIUS = 1
DIM = 12

COLOR1 = "#%06x" % randint(0, 0xFFFFFF)
COLOR2 = "#%06x" % randint(0, 0xFFFFFF)

theta = np.arange(0.0, 2*np.pi, 2*np.pi/STEP, dtype=np.float16)
coords = np.array([(RADIUS*np.cos(t), RADIUS*np.sin(t)) for t in theta])

fig, ax = plt.subplots(figsize=(DIM, DIM))
ax.scatter(*zip(*coords), c=COLOR1, lw=2)

def calc_idx(idx):
    if idx > len(coords) - 1:
        idx %= STEP
    return int(idx)

def animate(idx):
    i0 = calc_idx(idx + 1)
    i1 = calc_idx((idx + 1) * MULTI)
    
    x0, y0 = coords[i0]
    x1, y1 = coords[i1]
    x = [x0, x1]
    y = [y0, y1]
    
    ax.plot(x, y, c=COLOR2)

anim = FuncAnimation(fig, animate, frames=STEP*5, repeat=False, interval=0.0)

ax.set_aspect(1)
lim = 1.10*RADIUS
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.tick_params(axis='both', colors='black')
plt.show()
