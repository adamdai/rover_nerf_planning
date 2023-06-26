"""
AutoNav rover path planning
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from terrain_nerf.utils import wrap_angle


def arc(x0, u, N, dt):
    """
    """
    def dx(v, w, t, eps=1e-15):
        return v * np.sin((w + eps) * t) / (w + eps)
    def dy(v, w, t, eps=1e-15):
        return v * (1 - np.cos((w + eps) * t)) / (w + eps)

    traj = np.zeros((N, 3))
    traj[:,0] = [x0[0] + dx(u[0], u[1], i * dt) for i in range(N)]
    traj[:,1] = [x0[1] + dy(u[0], u[1], i * dt) for i in range(N)]
    traj[:,2] = x0[2] + u[1] * np.arange(N) * dt

    return traj


def local_to_global(pose, arc):
    """
    Parameters
    ----------
    pose : np.array 
        [x, y, theta]
    arc : np.array
        [x, y, theta]
    
    """
    x = arc[:,0] * np.cos(pose[2]) - arc[:,1] * np.sin(pose[2]) + pose[0]
    y = arc[:,0] * np.sin(pose[2]) + arc[:,1] * np.cos(pose[2]) + pose[1]
    theta = wrap_angle(arc[:,2] + pose[2])

    return np.vstack((x, y, theta)).T


class AutoNav:

    def __init__(self, goal):
        """
        """
        # Candidate arc parameters
        dt = 0.1
        self.arc_duration = 5.0  # seconds
        self.N = int(self.arc_duration / dt)
        N_arcs = 11
        speed = 2.5  # m/s
        max_omega = 0.25  # rad/s
        self.omegas = np.linspace(-max_omega, max_omega, N_arcs)
        self.candidate_arcs = [arc(np.zeros(3), [speed, w], self.N, dt) for w in self.omegas]

        # Costmap
        self.cmap_resolution = 1  # m
        self.cmap_dims = [41, 41]  # m
        self.cmap_center = [20, 20]
        self.costmap = np.zeros(self.cmap_dims)  # costmap is aligned with rover orientation, 40m x 40m

        self.goal = goal
        self.opt_idx = None


    def update_goal(self, goal):
        """
        """
        self.goal = goal


    def update_costmap(self, image):
        """
        """
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = gray / np.max(gray)
        # Downsample to 41x41
        gray = cv.resize(gray, self.cmap_dims)
        # Invert
        gray = 1.0 - gray
        # Automatically set values in center to 0
        cx, cy = self.cmap_center
        gray[cx-2:cx+2, cy-2:cy+2] = np.min(gray)
        self.costmap = gray
        #self.costmap = np.random.rand(self.cmap_dims[0], self.cmap_dims[1])
        #self.costmap = np.zeros(self.cmap_dims)
        

    def costmap_val(self, x, y):
        """
        """
        cx, cy = self.cmap_center
        # if x is an array
        if isinstance(x, np.ndarray):
            return self.costmap[(-x + cx + 0.5).astype(int), (y + cy + 0.5).astype(int)]
        # if x is a scalar
        else:
            return self.costmap[int(x + cx + 0.5), int(y + cy + 0.5)]

    
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
            cmap_cost = 100 * np.sum(self.costmap_val(arc[:,0], arc[:,1])) / self.N
            costs[i] += cmap_cost
            costmap_costs[i] = cmap_cost
            # Global cost
            arc_global = local_to_global(pose, arc)
            global_cost = np.linalg.norm(self.goal - arc_global[-1,:2])
            costs[i] += global_cost
            global_costs[i] = global_cost

        # Print steering angles and costs
        print("costmap costs: ", costmap_costs)
        print("global costs: ", global_costs)
        idx = np.argmin(costs)
        self.opt_idx = idx

        print("optimal cost: ", costs[idx])

        return self.candidate_arcs[idx], costs[idx], self.omegas[idx]

    
    def plot_costmap(self, ax, show_arcs=False):
        """
        """
        im = ax.imshow(self.costmap, alpha=0.5, cmap='viridis_r',
                       extent=[-self.cmap_dims[0]/2, self.cmap_dims[0]/2,
                               -self.cmap_dims[1]/2, self.cmap_dims[1]/2])
        ax.set_xlabel('y (m)')
        ax.set_ylabel('x (m)')
        if show_arcs:
            # Plot arcs in costmap
            cx, cy = self.cmap_center
            for i, arc in enumerate(self.candidate_arcs):
                if i == self.opt_idx:
                    ax.plot(arc[:,1], arc[:,0], 'r')
                else:
                    ax.plot(arc[:,1], arc[:,0], 'b')
        return im