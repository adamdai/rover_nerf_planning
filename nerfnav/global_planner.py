"""Global planners

"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import LinAlgError
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from nerfnav.astar import AStar
from collections import defaultdict


SPATIAL_NORM_CONST = 500.0


class GlobalPlanner(AStar):

    def __init__(self, costmap, feat_map, goal, goal_tolerance=30, interp_method='rbf', interp_features='spatial_rgb'):
        """
        Parameters
        ----------
        costmap : CostMap

        interp_method : str
            'avg', 'linear', 'kde', 'rbf', 'krr'
        interp_features : str
            'spatial', 'rgb', 'spatial_rgb'
        
        """
        self.interp_method = interp_method
        self.interp_features = interp_features


        self.feat_map = feat_map
        self.costmap = costmap
        self.width = self.costmap.mat.shape[0]
        self.height = self.costmap.mat.shape[1]

        self.path = None
        self.goal_px = tuple(goal)     # goal in pixel coordinates
        self.goal_tolerance = goal_tolerance
   
        self.local_samples = []  # dictionary indexed by x,y and stores cost
        for k in range(self.costmap.num_clusters):
            self.local_samples.append(defaultdict(list))

        self.max_costval = 0
        self.cmap_resolution = 2  # m

        self.cluster_costs = costmap.num_clusters * [None]
        for k in range(costmap.num_clusters):
            self.cluster_costs[k] = []

        # For each cluster, pre-compute features
        self.cluster_features = []

        for k in range(self.costmap.num_clusters):
            idxs = self.costmap.cluster_idxs[k]
            xy = np.stack(self.feat_map.img_to_global(idxs[:,0], idxs[:,1])).T
            rgb = self.feat_map.img[idxs[:,0], idxs[:,1], :]

            if self.interp_features == 'spatial':
                self.cluster_features.append(xy/SPATIAL_NORM_CONST)
            elif self.interp_features == 'rgb':
                self.cluster_features.append(rgb/255.0)
            elif self.interp_features == 'spatial_rgb':
                self.cluster_features.append(np.hstack((xy/SPATIAL_NORM_CONST, rgb/255.0)))


    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] 
                if 0 <= nx < self.width and 0 <= ny < self.height]

    def distance_between(self, node1, node2):
        return 5 * self.costmap.mat[node2]
    
    def heuristic_cost_estimate(self, node1, node2):
        """Straight line distance"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    

    def plan(self, start, goal):
        """Find a path from start to goal
        
        Start and goal are in image coordinates
        
        """
        path = list(self.astar(start, goal))
        dt = np.dtype('int32','int32')
        path = np.array(path, dtype=dt)
        return path
    

    def replan(self, cur_pose):
        """Replan"""
        x, y = self.feat_map.global_to_img(cur_pose[0], cur_pose[1])
        path = self.plan((x, y), self.goal_px)
        num_waypts = int(len(path) / 30)
        path = path[np.linspace(0, len(path)-1, num_waypts, dtype=int)]
        path_x, path_y = self.feat_map.img_to_global(path[:,0], path[:,1])
        path = np.vstack((path_x, path_y)).T
        self.path = path
        return path
    

    def plot(self, ax):
        """Plot costmap and path"""
        xmin, xmax, ymin, ymax = self.feat_map.bounds
        im = ax.imshow(self.costmap.mat, cmap='viridis', extent=self.feat_map.bounds, vmax=5.0)
        if self.path is not None:
            ax.plot(self.path[:,0], self.path[:,1], 'r')
            ax.scatter(self.path[:,0], self.path[:,1], c='r', s=10, marker='*')
        # ax.set_xticks(np.arange(xmin, xmax, 50))
        # ax.set_yticks(np.arange(ymin, ymax, 50))
        # Invert x axis
        #ax.invert_xaxis()
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        return im
    


    def naive_update_costmap(self, cost_vals):
        """Update costmap with new cost values

        Parameters
        ----------
        cost_vals : list
            List of cost values (x, y, cost)
        
        """
        for x, y, c in cost_vals:
            if not self.feat_map.in_bounds(x, y):
                continue
            i, j = self.feat_map.global_to_img(x, y)
            self.costmap.mat[i, j] = c

    
    def update_costmap(self, cost_vals):
        """
        Parameters
        ----------
        pose : np.array (3,)
            [x, y, theta]
        """
        in_bound_mask = self.feat_map.in_bounds(cost_vals[:, 0], cost_vals[:, 1])
        valid_cost_vals = cost_vals[in_bound_mask]
        #i, j = self.feat_map.global_to_img(valid_cost_vals[:, 0], valid_cost_vals[:, 1])
        coords = self.feat_map.get_img_coords(valid_cost_vals[:,:2])
        i = coords[:,0]
        j = coords[:,1]
        k_values = self.costmap.cluster_labels[i, j].astype(int)

        scale = self.cmap_resolution
        
        x_indices = (valid_cost_vals[:, 0] / scale).astype(int)
        y_indices = (valid_cost_vals[:, 1] / scale).astype(int)
        valid_costs = valid_cost_vals[:, 2]

        # Update local samples
        for k, x, y, c in zip(k_values, x_indices, y_indices, valid_costs):
            self.local_samples[k][(x, y)] = c
            self.cluster_costs[k].append(c)

        # Interpolate for each cluster
        #for k, cluster_samples in enumerate(self.local_samples):
        for k in np.unique(k_values):
            cluster_samples = self.local_samples[k]   
            if len(cluster_samples) == 0:
                continue

            xy_vals = np.array(list(cluster_samples.keys()))
            costs = np.array(list(cluster_samples.values()))

            # Assemble features
            rgb_features = self.feat_map.get_features(xy_vals)
            
            if self.interp_features == 'spatial':
                sample_features = xy_vals/SPATIAL_NORM_CONST
            elif self.interp_features == 'rgb':
                sample_features = rgb_features/255.0
            elif self.interp_features == 'spatial_rgb':
                sample_features = np.hstack((xy_vals/SPATIAL_NORM_CONST, rgb_features/255.0))

            # Interpolate
            if self.interp_method == 'linear':
                lreg = LinearRegression().fit(sample_features, costs)
                costs_fit = lreg.predict(self.cluster_features[k])
            elif self.interp_method == 'avg':
                costs_fit = np.mean(costs)
            elif self.interp_method == 'kde':
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample_features)
                costs_fit = kde.score_samples(self.cluster_features[k])
            elif self.interp_method == 'rbf':
                try:
                    rbf = RBFInterpolator(sample_features, costs)
                    costs_fit = rbf(self.cluster_features[k])
                except LinAlgError:
                    print("LinAlgError")
                    continue
            elif self.interp_method == 'krr':
                krr = KernelRidge().fit(sample_features, costs)
                costs_fit = krr.predict(self.cluster_features[k])
            
            # ## Averaging (TODO: update)
            # avg_cost = np.mean(ls[:,2])
            # self.costmap.mat[self.costmap.cluster_masks[k]] = avg_cost

            ## KDE
            # rgb_features = self.feat_map.get_features(ls[:,:2])
            # #sample_features = np.hstack((ls[:,:2]/SPATIAL_NORM_CONST, rgb_features/255.0))
            # sample_features = rgb_features/255.0

            # Spatial + feature
            # lreg = LinearRegression().fit(sample_features, costs)
            # costs_fit = lreg.predict(self.cluster_features[k])
            # krr = KernelRidge().fit(sample_features, costs)
            # costs_fit = krr.predict(self.cluster_features[k])
            
            # # Feature only
            # lreg = LinearRegression().fit(rgb_features/255.0, costs)
            # costs_fit = lreg.predict(self.cluster_features[k])

            # print(self.cluster_features[k].shape, self.costmap.mat[self.costmap.cluster_masks[k]].shape)

            #self.costmap.mat[self.costmap.cluster_masks[k]] = np.abs(costs_fit)


            # Update costmap
            i = self.costmap.cluster_idxs[k][:,0]
            j = self.costmap.cluster_idxs[k][:,1]

            #self.costmap.mat[i,j] = np.maximum(np.abs(costs_fit), self.costmap.mat[i,j])
            self.costmap.mat[i,j] = np.abs(costs_fit)
        
            # Cluster-specific metrics
            # cluster_size = len(cluster_samples)
            # mean_cost = np.mean(costs_fit)
            # median_cost = np.median(costs_fit)
            # std_cost = np.std(costs_fit)
            # min_cost = np.min(costs_fit)
            # max_cost = np.max(costs_fit)
    
            # print(f"Cluster {k} Metrics:")
            # print(f"  Size: {cluster_size}")
            # print(f"  Mean Cost: {mean_cost}")
            # print(f"  Median Cost: {median_cost}")
            # print(f"  Standard Deviation: {std_cost}")
            # print(f"  Min Cost: {min_cost}")
            # print(f"  Max Cost: {max_cost}\n")
    

    def update(self, cur_pose):
        """Check if goal has been reached, if so, get next waypoint

        """
        pass
