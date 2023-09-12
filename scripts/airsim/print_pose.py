"""Print current vehicle pose

"""

import airsim
from nerfnav.airsim_utils import get_pose2D, get_pose3D

# Rover pose
client = airsim.CarClient()
client.confirmConnection()

pose = client.simGetVehiclePose()

# print pose
#print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))

pose2D = get_pose2D(client)
print(pose2D)

# Camera pose
# client = airsim.VehicleClient()
# client.confirmConnection()

# info = client.simGetCameraInfo("0")
# pose = info.pose
# print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
# print(pose.orientation)
# pose3d = get_pose3D(client)
# print(pose3d)