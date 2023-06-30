"""
Utilities for AirSim API

"""

import numpy as np
import airsim

from terrain_nerf.utils import quat_to_R


def get_pose2D(client):
    """
    """
    pose = client.simGetVehiclePose()
    x = pose.position.x_val
    y = pose.position.y_val
    yaw = airsim.utils.to_eularian_angles(pose.orientation)[2]
    return np.array([x, y, yaw])


def get_pose3D(client):
    """
    """
    pose = client.simGetVehiclePose()
    x = pose.position.x_val
    y = pose.position.y_val
    z = pose.position.z_val
    roll, pitch, yaw = airsim.utils.to_eularian_angles(pose.orientation)
    return np.array([x, y, z, roll, pitch, yaw])


def airsim_pose_to_Rt(pose):
    """Convert AirSim pose to (R, t)

    """
    x = pose.position.x_val
    y = pose.position.y_val
    z = pose.position.z_val
    q = np.array([pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val])
    R = quat_to_R(q)
    return R, np.array([x, y, z])