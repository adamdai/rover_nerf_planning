"""Log vehicle pose

"""

import numpy as np
import airsim
import time

client = airsim.CarClient()
client.confirmConnection()

dt = 0.1
poses = []

try:
    while True:
        pose = client.simGetVehiclePose()
        print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
        pose_vals = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        poses.append(pose_vals)
        time.sleep(dt)

except KeyboardInterrupt:
    print("saving poses")
    np.savez('poses.npz', poses=poses)
    print("Interrupted by user, shutting down")
    client.enableApiControl(False)
    exit()