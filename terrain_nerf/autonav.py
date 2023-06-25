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
    traj[:,0] = [x0[0] + dx(u[0], u[1], i * dt) for i in range(N)]
    traj[:,1] = [x0[1] + dy(u[0], u[1], i * dt) for i in range(N)]
    traj[:,2] = x0[2] + u[1] * np.arange(N) * dt

    return traj


class AutoNav:

    def __init__(self):
        """
        """

        # Candidate arc parameters
        dt = 0.1
        N = 50
        speed = 1.0  # m/s
        self.omegas = np.linspace(-np.pi/2, np.pi/2, 10)
        self.candidate_arcs = [arc(np.zeros(3), [speed, w], N, dt) for w in self.omegas]
        self.costmap = np.zeros((21, 21))  # costmap is aligned with rover orientation, 20m x 20m

        self.visualization = True


    def update_costmap(self, image):
        """
        """
        self.costmap = np.random.rand(21, 21)


    def costmap_val(self, x, y):
        """
        """
        # if x is an array
        if isinstance(x, np.ndarray):
            return self.costmap[(x + 10 + 0.5).astype(int), (-y + 10 + 0.5).astype(int)]
        # if x is a scalar
        else:
            return self.costmap[int(x + 10 + 0.5), int(-y + 10 + 0.5)]

    
    def get_next_arc(self):
        """
        
        """
        # Compute costmap cost for each arc
        costs = np.zeros(len(self.candidate_arcs))
        for i, arc in enumerate(self.candidate_arcs):
            costs[i] = np.sum(self.costmap_val(arc[:,0], arc[:,1]))
        idx = np.argmin(costs)
        return self.candidate_arcs[idx], costs[idx], self.omegas[idx]

        