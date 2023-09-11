"""Global planners

"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import LinAlgError
from sklearn.neighbors import KernelDensity

from nerfnav.astar import AStar


SPATIAL_NORM_CONST = 200.0


class GlobalPlanner(AStar):

    def __init__(self, costmap, feat_map, goal, goal_tolerance=30):
        """
        Parameters
        ----------
        costmap : CostMap
        
        """
        self.feat_map = feat_map
        self.costmap = costmap
        self.width = self.costmap.mat.shape[0]
        self.height = self.costmap.mat.shape[1]

        self.path = None
        self.goal_px = tuple(goal)     # goal in pixel coordinates
        self.goal_tolerance = goal_tolerance

        self.local_samples = costmap.num_clusters * [None]  # local cost observations (x,y,c) for each cluster

        self.max_costval = 0


        self.cluster_costs = costmap.num_clusters * [None]
        for k in range(costmap.num_clusters):
            self.cluster_costs[k] = []

        # For each cluster, pre-compute features
        self.cluster_features = []

        print(self.costmap.cluster_labels.shape)
        #print(self.costmap.cluster_idxs.shape)
        print(self.feat_map.img.shape)

        for k in range(self.costmap.num_clusters):
            idxs = self.costmap.cluster_idxs[k]
            xy = np.stack(self.feat_map.img_to_global(idxs[:,0], idxs[:,1])).T
            rgb = self.feat_map.img[idxs[:,0], idxs[:,1], :]
            #self.cluster_features.append(np.hstack((xy/SPATIAL_NORM_CONST, rgb/255.0)))
            self.cluster_features.append(rgb/255.0)


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
        im = ax.imshow(self.costmap.mat, cmap='viridis', extent=self.feat_map.bounds)
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
        # Reset local samples
        self.local_samples = self.costmap.num_clusters * [None]

        # Update local samples
        for x, y, c in cost_vals:
            if not self.feat_map.in_bounds(x, y):
                continue
            i, j = self.feat_map.global_to_img(x, y)
            k = int(self.costmap.cluster_labels[i, j])
            if self.local_samples[k] is None:
                self.local_samples[k] = []
            self.local_samples[k].append((x, y, c))

            self.cluster_costs[k].append(c)

        # Interpolate for each cluster
        for k in range(self.costmap.num_clusters):
            if self.local_samples[k] is None:
                continue

            print("  cluster ", k)
            print("     costs: ", np.sort(np.array(self.local_samples[k])[:,2])[-10:])
            ls = np.array(self.local_samples[k])

            # ## Averaging
            # avg_cost = np.mean(ls[:,2])
            # self.costmap.mat[self.costmap.cluster_masks[k]] = avg_cost

            ## KDE
            rgb_features = self.feat_map.get_features(ls[:,:2])
            #sample_features = np.hstack((ls[:,:2]/SPATIAL_NORM_CONST, rgb_features/255.0))
            sample_features = rgb_features/255.0

            # Spatial + feature
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(sample_features, ls[:,2])
            costs_fit = np.exp(kde.score_samples(self.cluster_features[k]))
            # # Feature only
            # kde = KernelDensity(kernel='linear', bandwidth=0.5).fit(rgb_features/255.0, ls[:,2])
            # costs_fit = np.exp(kde.score_samples(self.cluster_features[k]))

            print(self.cluster_features[k].shape, self.costmap.mat[self.costmap.cluster_masks[k]].shape)

            #self.costmap.mat[self.costmap.cluster_masks[k]] = np.abs(costs_fit)
            i = self.costmap.cluster_idxs[k][:,0]
            j = self.costmap.cluster_idxs[k][:,1]
            #self.costmap.mat[i,j] = np.maximum(np.abs(costs_fit), self.costmap.mat[i,j])
            self.costmap.mat[i,j] = np.abs(costs_fit)
        
        self.max_costval = max(np.max(self.costmap.mat), self.max_costval)
        print("  MAX GLOBAL COST: ", self.max_costval)
        print("  Min GLOBAL COST: ", np.min(self.costmap.mat))
    

    def update(self, cur_pose):
        """Check if goal has been reached, if so, get next waypoint

        """
        pass
