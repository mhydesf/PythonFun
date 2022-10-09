from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation

# CONSTANTS
DPI = 96                            # MONITOR DPI
DIM1 = 800/DPI                      # X DIMENSION (Pixels)
DIM2 = 800/DPI                      # Y DIMENSION (Pixels)
MAIN_LINE_LENGTH = int(DIM1 / 2)    # Main Revolute Radius
SUB_LINE_LENGTH = int(DIM1 / 2.5)  # Secondary Revolute Radii
ANGLE1_INIT = 2/3 * np.pi           # Secondary Revolute Starting Angle
ANGLE2_INIT = 2*np.pi               # Secondary Revolute Starting Angle
ANGLE3_INIT = 300/90*np.pi          # Secondary Revolute Starting Angle
REVOLUTION = 2*np.pi                # One Revolution (2pi Radians)
ITERATIONS = 4                      # Number of Revolutions for Animation

# Line width and color definitions
WIDTH1 = 3
WIDTH2 = 2.5
COLOR1 = "#333333"
COLOR2 = "#FFFFFF"

# Plt Configuration
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(DIM2, DIM1))


class Animation:
    """
    Animation class for tracking lines and point tracing.
    
    Coordinate system is Cartesian.
    
    A single line revolves at radius MAIN_LINE_LENGTH around
    (0, 0) and three secondary lines revolve around the tip
    of the main segment. The paths of the 3 secondary line
    endpoints are traced to form cool patters. 
    """
    def __init__(self, ax: Axes):
        self.ax = ax
        self.main_link_size = MAIN_LINE_LENGTH
        self.sub_link_size = SUB_LINE_LENGTH
        
        x1 = self.main_link_size + self.sub_link_size * np.cos(ANGLE1_INIT)
        x2 = self.main_link_size + self.sub_link_size * np.cos(ANGLE2_INIT)
        x3 = self.main_link_size + self.sub_link_size * np.cos(ANGLE3_INIT)
        
        y1 = self.sub_link_size * np.sin(ANGLE1_INIT)
        y2 = self.sub_link_size * np.sin(ANGLE2_INIT)
        y3 = self.sub_link_size * np.sin(ANGLE3_INIT)
        
        main_x = [0, self.main_link_size]
        sub1_x = [self.main_link_size, x1]
        sub2_x = [self.main_link_size, x2]
        sub3_x = [self.main_link_size, x3]
        
        main_y = [0, 0]
        sub1_y = [0, y1]
        sub2_y = [0, y2]
        sub3_y = [0, y3]
        
        self.trace1_x = []
        self.trace1_y = []
        self.trace2_x = []
        self.trace2_y = []
        self.trace3_x = []
        self.trace3_y = []
        
        self.main_link = self.ax.plot(main_x, main_y, lw=WIDTH1, c=COLOR1)[0]
        self.sub1 = self.ax.plot(sub1_x, sub1_y, lw=WIDTH1, c=COLOR1)[0]
        self.sub2 = self.ax.plot(sub2_x, sub2_y, lw=WIDTH1, c=COLOR1)[0]
        self.sub3 = self.ax.plot(sub3_x, sub3_y, lw=WIDTH1, c=COLOR1)[0]
        self.trace1 = self.ax.plot(self.trace1_x, self.trace1_y, lw=WIDTH2, c=COLOR2)[0]
        self.trace2 = self.ax.plot(self.trace2_x, self.trace2_y, lw=WIDTH2, c=COLOR2)[0]
        self.trace3 = self.ax.plot(self.trace3_x, self.trace3_y, lw=WIDTH2, c=COLOR2)[0]
        
    def __call__(self, thetas: Tuple[float, float]):
        theta1, theta2 = thetas
        
        main_x = self.main_link_size * np.cos(theta1)
        main_y = self.main_link_size * np.sin(theta1)

        x1 = main_x + self.sub_link_size * np.cos(ANGLE1_INIT + theta2)
        x2 = main_x + self.sub_link_size * np.cos(ANGLE2_INIT + theta2)
        x3 = main_x + self.sub_link_size * np.cos(ANGLE3_INIT + theta2)
        
        y1 = main_y + self.sub_link_size * np.sin(ANGLE1_INIT + theta2)
        y2 = main_y + self.sub_link_size * np.sin(ANGLE2_INIT + theta2)
        y3 = main_y + self.sub_link_size * np.sin(ANGLE3_INIT + theta2)
        
        sub1_x = [main_x, x1]
        sub2_x = [main_x, x2]
        sub3_x = [main_x, x3]
        
        sub1_y = [main_y, y1]
        sub2_y = [main_y, y2]
        sub3_y = [main_y, y3]
        
        self.trace1_x.append(x1)
        self.trace1_y.append(y1)
        self.trace2_x.append(x2)
        self.trace2_y.append(y2)
        self.trace3_x.append(x3)
        self.trace3_y.append(y3)
        
        self.main_link.set_data([0, main_x], [0, main_y])
        self.sub1.set_data(sub1_x, sub1_y)
        self.sub2.set_data(sub2_x, sub2_y)
        self.sub3.set_data(sub3_x, sub3_y)
        self.trace1.set_data(self.trace1_x, self.trace1_y)
        self.trace2.set_data(self.trace2_x, self.trace2_y)
        self.trace3.set_data(self.trace3_x, self.trace3_y)
        return self.main_link, self.sub1, self.sub2, self.sub3, self.trace1, self.trace2, self.trace3

# Two sets of angles for animation
# Theta 1 - Main Line Angles
# Theta 2 - Secondary Line Angles
#       Irrational multiples result in out of phase
#       rotation (which is way cooler)
theta1 = np.linspace(0, ITERATIONS*REVOLUTION, 360*ITERATIONS)
theta2 = np.linspace(0, ITERATIONS*REVOLUTION*3.25, 360*ITERATIONS)
thetas = [(t1, t2) for t1, t2 in zip(theta1, theta2)]

a = Animation(ax)
anim = FuncAnimation(fig, a, frames=thetas, repeat=True, interval=0.1)

major_ticks = np.arange(-DIM2, DIM2, DIM2/5)
minor_ticks = np.arange(-DIM2, DIM2, DIM2/20)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.tick_params(axis='both', colors='black')
ax.grid(which='both')
ax.grid(which='minor', alpha=0.075)
ax.grid(which='major', alpha=0.3)
ax.set_xlim(-DIM2, DIM2)
ax.set_ylim(-DIM1, DIM1)
plt.show()