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
        
        self.open_set = {self.start.coord: self.start}
        self.closed_set = {}
        
    def run(self) -> np.ndarray:
        while self.open_set:
            n_id = min(self.open_set,
                        key=lambda o: self.open_set[o].cost +\
                            self.calc_heuristic(self.end, self.open_set[o]))
            curr_node = self.open_set[n_id]

            if curr_node.coord == self.end.coord:
                return self.draw_path_on_world(curr_node=curr_node)
              
            del self.open_set[n_id]
            self.closed_set[n_id] = curr_node
            
            for neighbor in self.find_neighbors(curr_node=curr_node):
                x, y, cost = neighbor
                neighbor = Node((x, y), cost, curr_node)
                
                if (x, y) in self.closed_set:
                    continue
                
                if not self.verify_node(neighbor):
                    continue

                if (x, y) not in self.open_set:
                    self.open_set[(x, y)] = neighbor
                else:
                    if self.open_set[(x, y)].cost >= cost:
                        self.open_set[(x, y)] = neighbor

        print(f"Could not find path from {self.start.coord} to {self.end.coord}")
        return None
            
    
    def draw_path_on_world(self,
                           curr_node: Node) -> np.ndarray:
        total_path = [curr_node.coord]
        while curr_node.prev_node != -1:
            curr_node = curr_node.prev_node
            total_path.append(curr_node.coord)
            
        for coord in total_path:
            self.world[coord] = 2
        
        return self.world

    def verify_node(self, node: Node) -> bool:
        if self.world[node.coord] == 1:
            return False
        return True
    
    def find_neighbors(self, curr_node: Node) -> List[Tuple[int]]:
        x, y = curr_node.coord
        neighbors = []
        
        if x > 0:    
            neighbors.append((x-1, y, 1))
        if x < self.world.shape[0] - 1:
            neighbors.append((x+1, y, 1))
        if y < self.world.shape[1] - 1:
            neighbors.append((x, y+1, 1))
        if y > 0:
            neighbors.append((x, y-1, 1))
        
        return neighbors
    
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


def main():
    
    DESCRIPTION = "A Star Algorithm Visualization Program"
    
    # Load script arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-x', '--x_dim', type=int, default=64,
                        help="X-Dimension of Simulation Grid")
    parser.add_argument('-y', '--y_dim', type=int, default=64,
                        help="Y-Dimension of Simulation Grid")
    parser.add_argument('-s', '--start', type=int, default=(0, 0), nargs='+',
                        help="Starting Coordinate of Simulation")
    parser.add_argument('-e', '--end', type=int, default=(63, 63), nargs='+',
                        help="End or Goal Coordinate of Simulation")
    parser.add_argument('-o', '--obstacles', type=float, default=0.05,
                        help="Percentage of Grid Area to be Obstructed (Randomly)")
    args = parser.parse_args()
    
    # Declare constants
    X_DIM = int(args.x_dim)
    Y_DIM = int(args.y_dim)
    OBS_PERC = float(args.obstacles)
    OBSTACLES = int(OBS_PERC*(X_DIM*Y_DIM))
    P0 = tuple(args.start)
    Pn = tuple(args.end)
    
    if OBS_PERC > 1.0:
        raise ValueError(f"Obstacle percentage {OBS_PERC} too high. Must be below 1.0")

    # Generate world (grid)
    #world = grid((X_DIM, Y_DIM), obstacles=OBSTACLES)
    
    world = np.zeros(shape=(X_DIM, Y_DIM), dtype=np.int8)
    world[0:48, 20] = 1
    world[23:X_DIM, 38] = 1
    world[48, 44:Y_DIM] = 1
    
    a_star = AStar(world=world, start=P0, end=Pn)
    result = a_star.run()

    if result is not None:
        # Plot world and solution
        cmap1 = colors.ListedColormap(['white', 'black', 'green'])
        
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(result, cmap=cmap1)
        
        #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(0, X_DIM, 1))
        ax.set_yticks(np.arange(0, Y_DIM, 1))
        ax.tick_params(axis='both', colors='white')
        
        plt.xlim(0, X_DIM)
        plt.ylim(0, Y_DIM)
        plt.show()

if __name__ == "__main__":
    main()