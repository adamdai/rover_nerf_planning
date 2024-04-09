"""Log vehicle pose

"""

import numpy as np
import airsim
import time

from rover_nerf.airsim_utils import get_pose2D, get_pose3D

client = airsim.CarClient()
client.confirmConnection()

dt = 0.1
states = []
collision_count = 0

try:
    while True:
        collision_info = client.simGetCollisionInfo("Rover")  # FIXME: collisions not detected in Moon environment
        if collision_info.has_collided:
            collision_count += 1
            print(f"==> Collision detected! Count = {collision_count}")

        # pose = client.simGetVehiclePose()
        # x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val
        pose2D = get_pose2D(client)
        print(f"x = {pose2D[0]:.2f}, y = {pose2D[1]:.2f}, theta = {pose2D[2]:.2f}")
        states.append(pose2D)
        # theta = np.rad2deg(pose2D[2])
        # print(f"x = {x:.2f}, y = {y:.2f}, z = {z:.2f}, theta = {theta:.2f}")


        # car_state = client.getCarState()
        # x = car_state.kinematics_estimated.position.x_val
        # y = car_state.kinematics_estimated.position.y_val
        # z = car_state.kinematics_estimated.position.z_val
        # speed = car_state.speed
        # yaw = airsim.utils.to_eularian_angles(pose.orientation)[2]
        # print(f"x = {x:.2f}, y = {y:.2f}, z = {z:.2f}, speed = {speed:.2f}, yaw = {yaw:.2f}")
        # state_vals = np.array([x, y, z, speed, yaw])
        # states.append(state_vals)
        time.sleep(dt)

except KeyboardInterrupt:
    print("Saving states")
    timestamp = time.strftime("%Y%m%d-%H%M")
    np.savez(f'../../data/airsim/logs/states_{timestamp}.npz', states=states, collision_count=collision_count)
    print("Interrupted by user, shutting down")
    client.enableApiControl(False)
    exit()