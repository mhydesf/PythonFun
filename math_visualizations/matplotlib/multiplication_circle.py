from ast import arg
import numpy as np
import argparse
from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

DESCRIPTION = """Visualization of multiplication tables along points on
the circumferance of a circle. See this link for more detail:
https://www.youtube.com/watch?v=qhbuKbxJsk8&ab_channel=Mathologer"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-s', '--step', type=int, default=200,
                    help="Number of points to define on circular boundary (higher is prettier)")
parser.add_argument('-m', '--multiplication', type=int, default=2,
                    help="Factor to multiply by - which times table are you visualizing?")
parser.add_argument('-d', '--dim', type=int, default=12,
                    help="Window Dimension")
args = parser.parse_args()

STEP = args.step
MULTI = args.multiplication
RADIUS = 1
DIM = args.dim

COLOR1 = "#%06x" % randint(0, 0xFFFFFF)
COLOR2 = "#%06x" % randint(0, 0xFFFFFF)

theta = np.arange(0.0, 2*np.pi, 2*np.pi/STEP, dtype=np.float16)
coords = np.array([(RADIUS*np.cos(t), RADIUS*np.sin(t)) for t in theta])

fig, ax = plt.subplots(figsize=(DIM, DIM))
ax.scatter(*zip(*coords), c=COLOR1, lw=2)

def calc_idx(idx) -> int:
    if idx > len(coords) - 1:
        idx %= STEP
    return int(idx)

def animate(idx) -> None:
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
