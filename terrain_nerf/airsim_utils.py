"""
Utilities for AirSim API

"""

import numpy as np
import airsim


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