from typing import Tuple, List
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from random import randint


class Node:
    
    def __init__(self,
                 coord: Tuple[int, int],
                 cost: float,
                 prev_node: Tuple[int, int]):
        self.coord = coord
        self.cost = cost
        self.prev_node = prev_node


class AStar:
    
    def __init__(self,
                 world: np.ndarray,
                 start: Tuple[int, int],
                 end: Tuple[int, int]):
        self.world = world
        self.start = Node(start, 0.0, -1)
        self.end = Node(end, 0.0, -1)
        
        self.motion = [[1, 0, 1],
                       [0, 1, 1],
                       [-1, 0, 1],
                       [0, -1, 1],
                       [-1, -1, math.sqrt(2)],
                       [-1, 1, math.sqrt(2)],
                       [1, -1, math.sqrt(2)],
                       [1, 1, math.sqrt(2)]]
        
        self.open_set = {self.start.coord: self.start}
        self.closed_set = {}
        
        self.fig, self.ax = plt.subplots(figsize=(12,12))
        
        ob_x, ob_y = grid_to_obstacle_list(self.world)
        circ1 = plt.Circle(start, 1, color='r')
        circ2 = plt.Circle(end, 1, color='r')
        self.ax.scatter(ob_x, ob_y, c='k', linestyle=':')
        self.ax.add_patch(circ1)
        self.ax.add_patch(circ2)
        
        X_MAX, Y_MAX = self.world.shape
        self.ax.set_xlim(0, X_MAX)
        self.ax.set_ylim(0, Y_MAX)
        self.ax.tick_params(axis='both', colors='white')
        
    def run(self) -> np.ndarray:
        while self.open_set:
            curr_idx = min(self.open_set,
                        key=lambda o: self.open_set[o].cost +\
                            self.calc_heuristic(self.end, self.open_set[o]))
            curr_node = self.open_set[curr_idx]

            self.ax.plot(curr_node.coord[0], curr_node.coord[1], "xg")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if len(self.closed_set) % 10 == 0:
                plt.pause(0.000001)

            if curr_node.coord == self.end.coord:
                return self.calculate_final_path(curr_node=curr_node)
              
            del self.open_set[curr_idx]
            self.closed_set[curr_idx] = curr_node
            
            for neighbor in self.motion:
                x0, y0 = curr_node.coord
                dx, dy, cost = neighbor
                neighbor = Node((x0 + dx, y0 + dy), curr_node.cost + cost, curr_node)
                x, y = neighbor.coord
                
                if (x, y) in self.closed_set:
                    continue
                
                if not self.verify_node(neighbor):
                    continue

                if (x, y) not in self.open_set:
                    self.open_set[(x, y)] = neighbor
                else:
                    if self.open_set[(x, y)].cost > neighbor.cost:
                        self.open_set[(x, y)] = neighbor

        print(f"Could not find path from {self.start.coord} to {self.end.coord}")
        return None
            
    
    def calculate_final_path(self,
                             curr_node: Node) -> np.ndarray:
        total_path = [curr_node.coord]
        while curr_node.prev_node != -1:
            curr_node = curr_node.prev_node
            total_path.append(curr_node.coord)
            
        return total_path

    def verify_node(self, node: Node) -> bool:
        x, y = node.coord
        
        if x < 0:
            return False
        if x > self.world.shape[0] - 1:
            return False
        if y < 0:
            return False
        if y > self.world.shape[1] - 1:
            return False
        
        if self.world[node.coord] == 1:
            return False
        
        return True
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        x1, y1 = n1.coord
        x2, y2 = n2.coord
        d = w * math.hypot(x1 - x2, y1 - y2)
        return d


def grid(shape: Tuple[int, int],
         obstacles: float) -> np.ndarray:
    grid = np.zeros(shape=shape, dtype=np.int8)
    
    for _ in range(obstacles):
        obs = np.array([1]*randint(0, 5))
        obs_x = randint(0, shape[0])
        obs_y = randint(0, shape[1])
        obs_dir = randint(0, 1)

        for block in obs:
            try:
                grid[obs_x, obs_y] = block
                if obs_dir == 0:
                    obs_x += 1
                if obs_dir == 1:
                    obs_y += 1
            except IndexError:
                break
    
    return grid

def grid_to_obstacle_list(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    x_coord = []
    y_coord = []
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            if val == 1:
                x_coord.append(i)
                y_coord.append(j)
    return x_coord, y_coord


def main():
    
    DESCRIPTION = "A Star Algorithm Visualization Program"
    
    # Load script arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-x', '--x_dim', type=int, default=64,
                        help="X-Dimension of Simulation Grid")
    parser.add_argument('-y', '--y_dim', type=int, default=64,
                        help="Y-Dimension of Simulation Grid")
    parser.add_argument('-s', '--start', type=int, default=None, nargs='+',
                        help="Starting Coordinate of Simulation")
    parser.add_argument('-e', '--end', type=int, default=None, nargs='+',
                        help="End or Goal Coordinate of Simulation")
    parser.add_argument('-o', '--obstacles', type=float, default=0.05,
                        help="Percentage of Grid Area to be Obstructed (Randomly)")
    args = parser.parse_args()
    
    # Declare constants
    X_DIM = int(args.x_dim)
    Y_DIM = int(args.y_dim)
    OBS_PERC = float(args.obstacles)
    OBSTACLES = int(OBS_PERC*(X_DIM*Y_DIM))
    if args.start:
        P0 = tuple(args.start)
    else:
       P0 = (randint(0, X_DIM), randint(0, Y_DIM))
    if args.end:
        Pn = tuple(args.end)
    else:
        Pn = (randint(0, X_DIM), randint(0, Y_DIM))
    
    if OBS_PERC > 1.0:
        raise ValueError(f"Obstacle percentage {OBS_PERC} too high. Must be below 1.0")

    # Generate world (grid)
    world = grid((X_DIM, Y_DIM), obstacles=OBSTACLES)
    
    # UNCOMMENT BELOW FOR A CUSTOM (NON_RANDOM)
    # WORLD. MODIFY AS FIT
    #P0 = (2, 2)
    #Pn = (62, 62)
    #world = np.zeros(shape=(X_DIM, Y_DIM), dtype=np.int8)
    #world[0:48, 20] = 1
    #world[23:X_DIM, 38] = 1
    #world[48, 44:Y_DIM] = 1
    
    a_star = AStar(world=world, start=P0, end=Pn)
    result = a_star.run()
    
    if result is not None:
        a_star.ax.plot([idx[0] for idx in result], [idx[1] for idx in result], "-r")
        plt.show()

if __name__ == "__main__":
    main()