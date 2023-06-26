"""Log vehicle pose

"""

import numpy as np
import airsim
import time

client = airsim.CarClient()
client.confirmConnection()

dt = 0.1
states = []

try:
    while True:
        pose = client.simGetVehiclePose()
        car_state = client.getCarState()
        x = car_state.kinematics_estimated.position.x_val
        y = car_state.kinematics_estimated.position.y_val
        z = car_state.kinematics_estimated.position.z_val
        speed = car_state.speed
        yaw = airsim.utils.to_eularian_angles(pose.orientation)[2]
        print(f"x = {x:.2f}, y = {y:.2f}, z = {z:.2f}, speed = {speed:.2f}, yaw = {yaw:.2f}")
        state_vals = np.array([x, y, z, speed])
        states.append(state_vals)
        time.sleep(dt)

except KeyboardInterrupt:
    print("Saving states")
    np.savez('states.npz', states=states)
    print("Interrupted by user, shutting down")
    client.enableApiControl(False)
    exit()