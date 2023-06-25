"""
AutoNav rover path planning
"""


import numpy as np



def arc(x0, u, N, dt):
    """
    """
    def dx(v, w, t, eps=1e-15):
        return v * np.sin((w + eps) * t) / (w + eps)
    def dy(v, w, t, eps=1e-15):
        return v * (1 - np.cos((w + eps) * t)) / (w + eps)

    traj = np.zeros((N, 3))
    traj[:,1] = [x0[0] + dx(u[0], u[1], i * dt) for i in range(N)]
    traj[:,0] = [x0[1] + dy(u[0], u[1], i * dt) for i in range(N)]
    traj[:,2] = x0[2] + u[1] * np.arange(N) * dt

    return traj


class AutoNav:

    def __init__(self):
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

        self.visualization = True


    def update_costmap(self, image):
        """
        """
        self.costmap = np.random.rand(self.cmap_dims[0], self.cmap_dims[1])


    def costmap_val(self, x, y):
        """
        """
        # if x is an array
        if isinstance(x, np.ndarray):
            return self.costmap[(x + self.cmap_center[0] + 0.5).astype(int), (-y + self.cmap_center[1] + 0.5).astype(int)]
        # if x is a scalar
        else:
            return self.costmap[int(x + self.cmap_center[0] + 0.5), int(-y + self.cmap_center[1] + 0.5)]

    
    def replan(self):
        """
        
        """
        # Compute costmap cost for each arc
        costs = np.zeros(len(self.candidate_arcs))
        for i, arc in enumerate(self.candidate_arcs):
            costs[i] = np.sum(self.costmap_val(arc[:,0], arc[:,1]))
        idx = np.argmin(costs)
        return self.candidate_arcs[idx], costs[idx], self.omegas[idx]

        