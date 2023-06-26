"""Print current vehicle pose

"""

import airsim
from terrain_nerf.airsim_utils import get_pose2D

client = airsim.CarClient()
client.confirmConnection()

pose = client.simGetVehiclePose()

# print pose
print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))

pose2D = get_pose2D(client)
print("theta={}".format(pose2D[2]))