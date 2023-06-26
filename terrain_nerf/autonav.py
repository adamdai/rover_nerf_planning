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
        N = int(self.arc_duration / dt)
        N_arcs = 11
        speed = 2.5  # m/s
        max_omega = 0.25  # rad/s
        self.omegas = np.linspace(-max_omega, max_omega, N_arcs)
        self.candidate_arcs = [arc(np.zeros(3), [speed, w], N, dt) for w in self.omegas]

        # Costmap
        self.cmap_resolution = 1  # m
        self.cmap_dims = [41, 41]  # m
        self.cmap_center = [20, 20]
        self.costmap = np.zeros(self.cmap_dims)  # costmap is aligned with rover orientation, 40m x 40m

        self.goal = goal

        self.visualization = False
        self.image = None

        if self.visualization:
            f, (self.ax1, self.ax2) = plt.subplots(1, 2)
            plt.ion()
            plt.show()


    def update_goal(self, goal):
        """
        """
        self.goal = goal


    def update_costmap(self, image):
        """
        """
        self.image = image
        #self.costmap = np.random.rand(self.cmap_dims[0], self.cmap_dims[1])
        self.costmap = np.zeros(self.cmap_dims)


    def costmap_val(self, x, y):
        """
        """
        # if x is an array
        if isinstance(x, np.ndarray):
            return self.costmap[(x + self.cmap_center[0] + 0.5).astype(int), (-y + self.cmap_center[1] + 0.5).astype(int)]
        # if x is a scalar
        else:
            return self.costmap[int(x + self.cmap_center[0] + 0.5), int(-y + self.cmap_center[1] + 0.5)]

    
    def replan(self, pose):
        """
        
        """
        costs = np.zeros(len(self.omegas))
        for i, w in enumerate(self.omegas):
            arc = self.candidate_arcs[i]
            # Steering cost
            costs[i] += np.abs(w)
            # Costmap cost
            costs[i] += np.sum(self.costmap_val(arc[:,0], arc[:,1]))
            # Global cost
            arc_global = local_to_global(pose, arc)
            costs[i] += np.linalg.norm(self.goal - arc_global[-1,:2])

        # Print steering angles and costs
        print("steering angles: ", self.omegas)
        print("costs: ", costs)
        idx = np.argmin(costs)

        if self.visualization:
            self.ax2.clear()
            self.ax1.imshow(self.image)
            self.ax2.imshow(self.costmap, alpha=0.2)
            self.ax2.plot(self.candidate_arcs[idx][:,0] + self.cmap_center[0] + 0.5, -self.candidate_arcs[idx][:,1] + self.cmap_center[1] + 0.5)
            plt.draw()
            plt.pause(.001)

        print("optimal cost: ", costs[idx])

        return self.candidate_arcs[idx], costs[idx], self.omegas[idx]

    
    def plot_arcs(self, ax):
        """
        """
        for arc in self.candidate_arcs:
            ax.plot(arc[:,0], arc[:,1])