"""
AutoNav rover path planning
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from nerfnav.autonav_utils import arc, local_to_global, depth_to_points


class AutoNav:

    def __init__(self, goal, arc_duration=5.0):
        """
        """
        # Candidate arc parameters
        dt = 0.1
        self.arc_duration = arc_duration  # seconds
        self.N = int(self.arc_duration / dt)
        N_arcs = 15
        speed = 3.5  # m/s
        max_omega = 0.3  # rad/s
        self.omegas = np.linspace(-max_omega, max_omega, N_arcs)
        self.candidate_arcs = [arc(np.zeros(3), [speed, w], self.N, dt) for w in self.omegas]

        # Costmap
        self.cmap_resolution = 2  # m
        self.max_depth = 25  # m
        W = int(self.max_depth / self.cmap_resolution)
        self.cmap_dims = [W + 1, 2*W + 1]  # m
        self.cmap_center = [W, W]
        self.costmap = np.zeros(self.cmap_dims) 

        self.goal = goal
        self.opt_idx = None
        self.max_costval = 0

        self.cam_params = {'w': 800,
                            'h': 600,
                            'cx': 400, 
                            'cy': 300, 
                            'fx': 400, 
                            'fy': 300}


    def update_goal(self, goal):
        """
        """
        self.goal = goal


    def update_costmap(self, pose, depth):
        """
        Parameters
        ----------
        pose : np.array 
            [x, y, theta]
        """
        points = depth_to_points(depth, self.cam_params, depth_thresh=self.max_depth, patch_size=1)

        # Bin points into grid
        bins = {}
        scale = self.cmap_resolution
        for x, y, z in points[:,:3]:
            x_idx = self.cmap_center[0] - int(x / scale)
            y_idx = self.cmap_center[1] + int(y / scale)
            if (x_idx, y_idx) not in bins:
                bins[(x_idx, y_idx)] = [z]
            else:
                bins[(x_idx, y_idx)].append(z)

        cost_vals = []                           # TODO: use a longer max_depth for cost_vals 
        for k, v in bins.items():
            cost = 500 * np.var(v)
            self.costmap[k] = cost  # update costmap
            self.max_costval = max(self.max_costval, cost)

            # Convert local coordinates to global coordinates
            local_x = self.cmap_center[0] - k[0]
            local_y = k[1] - self.cmap_center[1]
            global_x = local_x * np.cos(pose[2]) - local_y * np.sin(pose[2]) + pose[0]
            global_y = local_x * np.sin(pose[2]) + local_y * np.cos(pose[2]) + pose[1]
            cost_vals.append([global_x, global_y, cost])

        print("  MAX LOCAL COST: ", self.max_costval)
        return cost_vals
        

    def costmap_val(self, x, y):
        """
        """
        cx, cy = self.cmap_center
        # if x is an array
        if isinstance(x, np.ndarray):
            return self.costmap[(-x + cx + 0.5).astype(int), (-y + cy + 0.5).astype(int)]
        # if x is a scalar
        else:
            return self.costmap[int(-x + cx + 0.5), int(-y + cy + 0.5)]

    
    def replan(self, pose):
        """
        
        """
        costs = np.zeros(len(self.omegas))
        costmap_costs = np.zeros(len(self.omegas))
        global_costs = np.zeros(len(self.omegas))
        for i, w in enumerate(self.omegas):
            arc = self.candidate_arcs[i]
            # Steering cost
            costs[i] += np.abs(w)
            # Costmap cost
            cmap_cost = np.sum(self.costmap_val(arc[:,0], arc[:,1])) / self.N
            costs[i] += cmap_cost
            costmap_costs[i] = cmap_cost
            # Global cost
            arc_global = local_to_global(pose, arc)
            global_cost = np.linalg.norm(self.goal - arc_global[-1,:2])
            costs[i] += global_cost
            global_costs[i] = global_cost

        # Print steering angles and costs
        # print("costmap costs: ", costmap_costs)
        # print("global costs: ", global_costs)
        idx = np.argmin(costs)
        self.opt_idx = idx

        # print("optimal cost: ", costs[idx])

        return self.candidate_arcs[idx], costs[idx], self.omegas[idx]

    
    def plot_costmap(self, ax, show_arcs=False):
        """
        """
        im = ax.imshow(self.costmap, alpha=0.75, cmap='viridis',
                       extent=[self.max_depth, -self.max_depth,
                               0, self.max_depth],
                       vmin=0, vmax=50)
        ax.set_xlabel('y (m)')
        ax.set_ylabel('x (m)')
        if show_arcs:
            # Plot arcs in costmap
            for i, arc in enumerate(self.candidate_arcs):
                if i == self.opt_idx:
                    ax.plot(arc[:,1], arc[:,0], 'r', linewidth=3.0)
                else:
                    ax.plot(arc[:,1], arc[:,0], 'c--', alpha=0.5, linewidth=2.0)
        ax.set_xticks(np.arange(-self.max_depth, self.max_depth, 5))
        ax.set_yticks(np.arange(0, self.max_depth, 5))
        # Invert x axis
        #ax.invert_xaxis()
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        return im