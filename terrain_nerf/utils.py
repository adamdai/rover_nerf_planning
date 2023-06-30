"""
General utilities

"""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


def wrap_angle(a):
    """
    Wrap angle to [-pi, pi)

    """
    return (a + np.pi) % (2 * np.pi) - np.pi


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def quat_to_R(quat):
    """Convert quaternion to 3D rotation matrix 

    Parameters
    ----------
    quat : np.array (1 x 4)
        Quaternion in scalar-last (x, y, z, w) format

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix

    """
    r = R.from_quat(quat)
    return r.as_matrix()


def euler_to_R(roll, pitch, yaw):
    """Convert Euler angles to 3D rotation matrix

    Parameters
    ----------
    roll : float
        Roll angle (radians)
    pitch : float
        Pitch angle (radians)
    yaw : float
        Yaw angle (radians)

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix

    """
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_matrix()


def pc_plot_trace(P, color=None, size=2):
    """Generate plotly plot trace for point cloud

    Parameters
    ----------
    P : np.array (N x 3)
        Point cloud

    Returns
    -------
    go.Scatter3d
        Scatter plot trace

    """
    return go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', marker=dict(size=size, color=color))


def trajectory_plot_trace(Rs, ts, color="red", scale=1.0):
    """Generate plotly plot trace for a 3D trajectory of poses
    TODO: plot xyz arrows as RGB

    Parameters
    ----------
    Rs : np.array (3 x 3 x N)
        Sequence of orientations
    ts : np.array (N x 3)
        Sequence of positions
    
    Returns
    -------
    list
        List containing traces for plotting 3D trajectory
    
    """
    points = go.Scatter3d(x=[ts[:,0]], y=[ts[:,1]], z=[ts[:,2]], showlegend=False)#, mode='markers', marker=dict(size=5))
    xs = []; ys = []; zs = []
    for i in range(len(ts)):
        for j in range(3):
            xs += [ts[i,0], ts[i,0] + scale*Rs[0,j,i], None]
            ys += [ts[i,1], ts[i,1] + scale*Rs[1,j,i], None]
            zs += [ts[i,2], ts[i,2] + scale*Rs[2,j,i], None]
    lines = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color), showlegend=False)
    return [points, lines]