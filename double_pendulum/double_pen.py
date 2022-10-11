import argparse
import random
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
plt.style.use('dark_background')

DESCRIPTION = "Visualization of a double pendulum path"

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--length_1', type=float, default=1.0,
                    help="Length of first pendulum arm")
parser.add_argument('--length_2', type=float, default=1.0,
                    help="Length of second penulum arm")
parser.add_argument('--mass_1', type=float, default=1.0,
                    help="Mass of pendulum 1")
parser.add_argument('--mass_2', type=float, default=1.0,
                    help="Mass of pendulum 2")
parser.add_argument('--theta_1', type=float, default=None,
                    help="Initial angle of pendulum 1 (random by default)")
parser.add_argument('--theta_2', type=float, default=None,
                    help="Initial angle of pendulum 2 (random by default)")
args = parser.parse_args()

G = 9.81
L1 = args.length_1
L2 = args.length_2
L = L1 + L2
M1 = args.mass_1
M2 = args.mass_2

if args.theta_1 is None:
    th1 = random.randint(-90, 90)
else:
    th1 = args.theta_1

if args.theta_2 is None:
    th2 = random.randint(-120, 120)
else:
    th2 = args.theta_2

w1 = 0.0
w2 = 0.0
t_stop = 60
dt = 0.01
history_len = int(t_stop/dt)


def derivs(state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
t = np.arange(0, t_stop, dt)

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(y[i - 1]) * dt

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
ax.set_aspect('equal')

line, = ax.plot([], [], '#A3BE8C', lw=3)
mass1 = ax.scatter([x1], [y1], c='#A3BE8C', lw=5)
mass2 = ax.scatter([x2], [y2], c='#A3BE8C', lw=5)
trace, = ax.plot([], [], '#FFFFFF', linestyle='-.', lw=2, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.96, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)
ax.text(0.05, 0.94, f"length 1: {L1}", transform=ax.transAxes)
ax.text(0.05, 0.92, f"length 2: {L2}", transform=ax.transAxes)
ax.text(0.05, 0.90, f"mass 1: {M1}", transform=ax.transAxes)
ax.text(0.05, 0.88, f"mass 2: {M2}", transform=ax.transAxes)
ax.text(0.05, 0.86, f"angle 1: {th1}", transform=ax.transAxes)
ax.text(0.05, 0.84, f"angle 2: {th2}", transform=ax.transAxes)
ax.tick_params(axis='both', colors='black')
ax.set_title("Double Pendulum Animation", fontsize=20)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    mass1.set_offsets([x1[i], y1[i]])
    mass2.set_offsets([x2[i], y2[i]])
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, mass1, mass2, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()