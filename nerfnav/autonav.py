"""
AutoNav rover path planning
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.linear_model import RANSACRegressor

from collections import defaultdict


from nerfnav.autonav_utils import arc, local_to_global, depth_to_points, estimate_hessian_trace, hessian_grid, compute_slope_and_roughness


class AutoNav:

    def __init__(self, goal, arc_duration=5.0):
        """
        """
        # Candidate arc parameters
        dt = 0.1
        self.arc_duration = arc_duration  # seconds
        self.N = int(self.arc_duration / dt)
        N_arcs = 15
        self.speed = 4.0  # m/s
        max_omega = 0.3  # rad/s
        self.omegas = np.linspace(-max_omega, max_omega, N_arcs)
        self.candidate_arcs = [arc(np.zeros(3), [self.speed, w], self.N, dt) for w in self.omegas]

        # Costmap
        self.cmap_resolution = 2  # m
        self.max_depth = 50  # m
        W = int(self.max_depth / self.cmap_resolution)
        self.cmap_dims = [W + 1, 2*W + 1]  # m
        self.cmap_center = [W, W]
        self.costmap = np.zeros(self.cmap_dims) 

        self.goal = goal
        self.opt_idx = None
        self.max_costval = 0

        self.ransac = RANSACRegressor(max_trials=10, residual_threshold=0.01)

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


    def calc_throttle(self, curr_speed):
        """
        """
        return 0.1 * (curr_speed - 2.0)


    # def update_costmap(self, pose, depth):
    #     """
    #     Parameters
    #     ----------
    #     pose : np.array 
    #         [x, y, theta]
    #     """
    #     points = depth_to_points(depth, self.cam_params, depth_thresh=self.max_depth, patch_size=1)

    #     # Bin points into grid
    #     bins = {}
    #     scale = self.cmap_resolution
    #     for x, y, z in points[:,:3]:
    #         x_idx = self.cmap_center[0] - int(x / scale)
    #         y_idx = self.cmap_center[1] + int(y / scale)
    #         if (x_idx, y_idx) not in bins:
    #             bins[(x_idx, y_idx)] = [z]
    #         else:
    #             bins[(x_idx, y_idx)].append(z)

    #     cost_vals = []                           # TODO: use a longer max_depth for cost_vals 
    #     for k, v in bins.items():
    #         cost = 500 * np.var(v)
    #         self.costmap[k] = cost  # update costmap
    #         self.max_costval = max(self.max_costval, cost)

    #         # Convert local coordinates to global coordinates
    #         local_x = self.cmap_center[0] - k[0]
    #         local_y = k[1] - self.cmap_center[1]
    #         global_x = local_x * np.cos(pose[2]) - local_y * np.sin(pose[2]) + pose[0]
    #         global_y = local_x * np.sin(pose[2]) + local_y * np.cos(pose[2]) + pose[1]
    #         cost_vals.append([global_x, global_y, cost])

    #     print("  MAX LOCAL COST: ", self.max_costval)
    #     return cost_vals

    def update_costmap(self, pose, depth):
        """
        Parameters
        ----------
        pose : np.array 
            [x, y, theta]
        """
        points = depth_to_points(depth, self.cam_params, depth_thresh=self.max_depth, patch_size=1)
        centroid = np.mean(points[:,:3], axis=0)

        bins = defaultdict(list)
        scale = self.cmap_resolution

        x_indices = self.cmap_center[0] - (points[:, 0] / scale).astype(int)
        y_indices = self.cmap_center[1] + (points[:, 1] / scale).astype(int)

        adjusted_points = points[:,:3] - centroid

        for x_idx, y_idx, point in zip(x_indices, y_indices, adjusted_points):
            bins[(x_idx, y_idx)].append(tuple(point))

        cost_vals = []                           # TODO: use a longer max_depth for cost_vals 
        for k, v in bins.items():

            bin_pts = np.array(v)
            
            if len(bin_pts) > 10:
                # print("k: ", k)
                # print("bin_pts: ", len(bin_pts))
                #try:
                    # c = estimate_hessian_trace(bin_pts)
                    # print("c: ", c)
                    # slope, roughness = compute_slope_and_roughness(bin_pts)
                    # self.costmap[k] = slope

                dem = {}
                dem_resolution = 0.1
                min_x, min_y = np.inf, np.inf
                max_x, max_y = -np.inf, -np.inf
                for i, (x, y, z) in enumerate(bin_pts):
                    x = int(x / dem_resolution)
                    y = int(y / dem_resolution)
                    if (x, y) not in dem:
                        dem[(x, y)] = z
                    else:
                        dem[(x, y)] = max((dem[(x, y)], z))

                xy_vals = np.array(list(dem.keys()))
                z_vals = np.array(list(dem.values()))
                #print("xy_vals: ", xy_vals.shape)

                self.ransac = RANSACRegressor(max_trials=10, residual_threshold=0.01)
                self.ransac.fit(xy_vals, z_vals)
                a, b = self.ransac.estimator_.coef_
                # a, b = 0, 0

                z_pred = self.ransac.estimator_.predict(xy_vals)
                # z_pred = np.zeros(len(z_vals))
                loss_vals = z_pred - z_vals

                    # dem = {}
                    # dem = np.zeros((20, 20))

                    # dem_resolution = 0.1
                    # min_x, min_y = np.inf, np.inf
                    # max_x, max_y = -np.inf, -np.inf
                    # for i, (x, y, z) in enumerate(bin_pts):
                    #     x = int(x / dem_resolution)
                    #     y = int(y / dem_resolution)
                    #     min_x = min(min_x, x)
                    #     min_y = min(min_y, y)
                    #     max_x = max(max_x, x)
                    #     max_y = max(max_y, y)
                    #     if (x, y) not in dem:
                    #         dem[(x, y)] = loss[i]
                    #     else:
                    #         dem[(x, y)] = max((dem[(x, y)], loss[i]))

                    # xy_vals = np.array(list(dem.keys()))
                    # loss_vals = np.array(list(dem.values()))

                    # dem_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
                    # dem_grid[xy_vals[:, 0] - min_x, xy_vals[:, 1] - min_y] = loss_vals

                roughness = estimate_hessian_trace(np.hstack((xy_vals, loss_vals[:, None])))

                    #roughness = hessian_grid(dem_grid)




                    # roughness = np.var(z_pred - bin_pts[:, 2])
                    # #print("loss: ", np.abs(z_pred - bin_pts[:, 2]))

                    # #outlier_frac = np.count_nonzero(~self.ransac.inlier_mask_) / len(self.ransac.inlier_mask_)
                n = np.array([a, b, -1])
                n = n / np.linalg.norm(n)
                if n[2] < 0:
                    n = -n
                slope = np.abs(np.arccos(np.dot(n, np.array([0, 0, 1]))))
                cost = 1.0 * slope + 10.0 * roughness
                # except ValueError as e:
                #     print(e)
                #     return bin_pts

                # cost = 500 * np.var(v)
                self.costmap[k] = cost  # update costmap
                self.max_costval = max(self.max_costval, cost)

                # Convert local coordinates to global coordinates
                local_x = self.cmap_center[0] - k[0]
                local_y = k[1] - self.cmap_center[1]
                global_x = local_x * np.cos(pose[2]) - local_y * np.sin(pose[2]) + pose[0]
                global_y = local_x * np.sin(pose[2]) + local_y * np.cos(pose[2]) + pose[1]
                cost_vals.append([global_x, global_y, cost])

        return np.array(cost_vals)
        

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
                       vmin=0, vmax=30)
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
