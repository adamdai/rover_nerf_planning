"""
Utilities for AutoNav
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from sklearn.linear_model import RANSACRegressor

from nerfnav.utils import wrap_angle


def arc(x0, u, N, dt):
    """
    """
    def dx(v, w, t, eps=1e-15):
        return v * np.sin((w + eps) * t) / (w + eps)
    def dy(v, w, t, eps=1e-15):
        return -v * (1 - np.cos((w + eps) * t)) / (w + eps)

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


def depth_to_points(depth, cam_params, depth_thresh=50.0, patch_size=20):
    """Project depth image to local point cloud
    
    Parameters
    ----------
    depth : np.array (w, h)
        Depth image
    cam_pose : np.array (4, 3)
        Camera pose in global frame (R, t)
    cam_params : dict
        Camera parameters
    depth_thresh : float
        Maximum depth value
    patch_size : int
        Patch size for sampling depth image

    Returns
    -------
    G : np.array (N, 5)
        Local points and associated pixel coordinates in image
    
    """
    w, h = depth.shape

    I, J = np.mgrid[0:w:patch_size, 0:h:patch_size]

    I = I.flatten()
    J = J.flatten()
    D = depth[0:w:patch_size, 0:h:patch_size].flatten()
    I = I[D < depth_thresh]
    J = J[D < depth_thresh]
    D = D[D < depth_thresh]

    G = np.zeros((D.shape[0], 5))
    G[:,0] = D                                                # x
    G[:,1] = (J - cam_params['cx']) * D / cam_params['fx']    # y
    G[:,2] = -(I - cam_params['cy']) * D / cam_params['fy']   # z
    G[:,3] = I
    G[:,4] = J

    return G



def estimate_hessian_trace(samples, lamb=1.0):
    x, y, z = samples[:,0], samples[:,1], samples[:,2]
    rbf = Rbf(x, y, z, function='multiquadric')

    # First order derivative (to set regularization constraint on smooth surface)
    def partial_derivatives(params):
        dx, dy = params
        fx = (rbf(x+dx, y) - rbf(x-dx, y))/ (2*dx)
        fy = (rbf(x, y+dy) - rbf(x, y-dy))/ (2*dy)
        # return np.sum(fx**2) + np.sum(fy**2) + lamb * (dx**2 + dy**2)
        return fx, fy

    # # Optimize
    # out = minimize(partial_derivatives, [0.01, 0.01], bounds=[(1e-6, 1), (1e-6, 1)])
    # dx_opt, dy_opt = out.x
    dx_opt, dy_opt = 0.1, 0.1

    # Second order derivative
    fxx = (rbf(x+dx_opt, y) - 2*rbf(x, y) + rbf(x-dx_opt, y))/ (dx_opt**2)
    fyy = (rbf(x, y+dy_opt) - 2*rbf(x, y) + rbf(x, y-dy_opt))/ (dy_opt**2)

    # print(partial_derivatives((1e-6, 1e-6)))

    # Estimate max of Hessian
    return np.max(fxx) + np.max(fyy)


def hessian_grid(grid):
    max_dxx, max_dyy = 0, 0

    for ds in [1, 2, 4]:
        ds_grid = grid[::ds, ::ds]
        padded_grid = np.pad(ds_grid, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        dxx, dyy = np.gradient(padded_grid, edge_order=2)
        max_dxx = max(max_dxx, np.max(np.abs(dxx)))
        max_dyy = max(max_dyy, np.max(np.abs(dyy)))
    return max_dxx + max_dyy




def compute_slope_and_roughness(points):
    """Compute slope and roughness of 3d points"""
    u, s, vh = np.linalg.svd(points)
    v1, v2, v3 = vh
    center = np.mean(points, axis=0)
    slope = np.arccos(np.dot(v3, np.array([0, 0, 1]))/np.linalg.norm(v3))
    roughness = np.var(np.abs(np.dot(points - center, v3)))
    return slope, roughness



def ransac_plane_fit(points):
    pass



def points_to_cost(points, resolution=2, center=[40, 40]):
    """Bin points into grid

    Parameters
    ----------
    points : np.array (N, 3)
        Local points (x, y, z)
    
    Returns
    -------
    bins : dict
        Dictionary of bins with key (x_idx, y_idx) and value list of z values
    
    """
    bins = {}
    scale = self.cmap_resolution
    for x, y, z in G[:,:3]:
        x_idx = self.cmap_center[0] - int(x / scale)
        y_idx = self.cmap_center[1] + int(y / scale)
        if (x_idx, y_idx) not in bins:
            bins[(x_idx, y_idx)] = [z]
        else:
            bins[(x_idx, y_idx)].append(z)

    cost_vals = []                           # TODO: use a longer max_depth for cost_vals 
    for k, v in bins.items():
        cost = 500 * np.var(v)
        self.costmap[k] = cost
        self.max_costval = max(self.max_costval, cost)
        local_x = self.cmap_center[0] - k[0]
        local_y = k[1] - self.cmap_center[1]
        global_x = local_x * np.cos(pose[2]) - local_y * np.sin(pose[2]) + pose[0]
        global_y = local_x * np.sin(pose[2]) + local_y * np.cos(pose[2]) + pose[1]
        cost_vals.append([global_x, global_y, cost])

    print("  MAX LOCAL COST: ", self.max_costval)
    return cost_vals


def local_to_global_cost(bins, pose):
    """Convert local cost grid to global cost samples
    
    """
    cost_vals = []                           # TODO: use a longer max_depth for cost_vals 
    for k, v in bins.items():
        cost = 500 * np.var(v)
        self.costmap[k] = cost
        self.max_costval = max(self.max_costval, cost)
        local_x = self.cmap_center[0] - k[0]
        local_y = k[1] - self.cmap_center[1]
        global_x = local_x * np.cos(pose[2]) - local_y * np.sin(pose[2]) + pose[0]
        global_y = local_x * np.sin(pose[2]) + local_y * np.cos(pose[2]) + pose[1]
        cost_vals.append([global_x, global_y, cost])
    
    return cost_vals