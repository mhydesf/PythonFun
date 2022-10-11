import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

STEP = 200.0
MULTI = 2
RADIUS = 1
DIM = 12

th = np.arange(0.0, 2*np.pi, 0.01)
po = np.array([(RADIUS*np.cos(t), RADIUS*np.sin(t)) for t in th])
theta = np.arange(0.0, 2*np.pi, 2*np.pi/STEP, dtype=np.float16)
coords = np.array([(RADIUS*np.cos(t), RADIUS*np.sin(t)) for t in theta])

fig, ax = plt.subplots(figsize=(DIM, DIM))
ax.scatter(*zip(*po), color='w', lw=1)
ax.scatter(*zip(*coords), c='r', lw=2)

def calc_idx(idx):
    if idx > len(coords) - 1:
        idx %= STEP
    return int(idx)

def animate(idx):
    i0 = calc_idx(idx + 1)
    i1 = calc_idx((idx + 1) * MULTI)
    
    x = [coords[i0][0], coords[i1][0]]
    y = [coords[i0][1], coords[i1][1]]
    
    ax.plot(x, y)

anim = FuncAnimation(fig, animate, frames=1000, repeat=True, interval=0.0)

ax.set_aspect(1)
lim = 1.10*RADIUS
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.tick_params(axis='both', colors='black')
plt.show()
