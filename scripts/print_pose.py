"""Print current vehicle pose

"""

import airsim

client = airsim.CarClient()
client.confirmConnection()

pose = client.simGetVehiclePose()

# print pose
print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))