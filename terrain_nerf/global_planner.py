"""Global planners

"""

import numpy as np
from terrain_nerf.astar import AStar


class GlobalPlanner(AStar):

    def __init__(self, costmap):
        self.costmap = costmap
        self.width = costmap.shape[0]
        self.height = costmap.shape[1]

    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] 
                if 0 <= nx < self.width and 0 <= ny < self.height]

    def distance_between(self, node1, node2):
        return self.costmap[node2]
    
    def heuristic_cost_estimate(self, node1, node2):
        """Straight line distance"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    

    def plan(self, start, goal):
        """Find a path from start to goal"""
        path = list(self.astar(start, goal))
        dt = np.dtype('int32','int32')
        path = np.array(path, dtype=dt)
        return path
    


class CostMap:

    def __init__(self, costmap, center, scale):
        self.costmap = costmap
        self.width = costmap.shape[0]
        self.height = costmap.shape[1]
    