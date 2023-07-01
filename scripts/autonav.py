"""Leader and follow cars taking images of each other
   
   - Car 2 is leader, car 1 is follower
   - Car 1 tracks a goal pose that is a fixed distance behind car 2

Notes
-----
    Positive steer turns right, negative turns left

Settings: 'settings_two_car.json'

"""

#import airsim_data_collection.common.setup_path
import airsim
import os
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2 as cv
import plotly.express as px

from terrain_nerf.autonav import AutoNav
from terrain_nerf.airsim_utils import get_pose2D
from terrain_nerf.global_planner import GlobalPlanner

## -------------------------- PARAMS ------------------------ ##
# Unreal environment
# (in unreal units, 100 unreal units = 1 meter)
UNREAL_PLAYER_START = np.array([-117252.054688, 264463.03125, 25148.908203])
UNREAL_GOAL = np.array([-83250.0, 258070.0, 24860.0])

GOAL_POS = (UNREAL_GOAL - UNREAL_PLAYER_START)[:2] / 100.0
print("GOAL_POS: ", GOAL_POS)

VISUALIZE = False

global_img = cv.imread('../data/airsim/images/test_scenario.png')
#global_planner = GlobalPlanner()
goal = GOAL_POS

## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    path_idx = 0
    current_pose = get_pose2D(client)

    # Initialize AutoNav class
    autonav = AutoNav(goal)

    # Visualization
    if VISUALIZE:
        f, (ax1, ax2) = plt.subplots(1, 2)
        # Set figure size
        f.set_figwidth(15)
        f.set_figheight(7)
        plt.ion()
        plt.show()

    # Parameters
    throttle = 0.5
    N_iters = 1e5
    idx = 0

    # brake the car
    car_controls.brake = 1
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    # wait until car is stopped
    time.sleep(1)

    car_controls.brake = 0
    print(autonav.costmap.shape)

    try:
        while idx < N_iters:
            start_time = time.time()
            current_pose = get_pose2D(client)
            print("current_pose: ", current_pose)
            print("goal: ", goal)
            if np.linalg.norm(current_pose[:2] - autonav.goal) < 10:
                print("Reached goal!")
                break

            # Get image
            png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
            img_decoded = cv.imdecode(np.frombuffer(png_image, np.uint8), -1)

            autonav.update_costmap(img_decoded)
            arc, cost, w = autonav.replan(current_pose)

            car_controls.steering = w / 1.6
            car_controls.throttle = throttle
            client.setCarControls(car_controls)
            print("steering: ", car_controls.steering, "throttle: ", car_controls.throttle)
            print("planning time: ", time.time() - start_time)
            print("--------------------------------------------------------------------------------")

            if VISUALIZE:
                ax2.clear()
                ax1.imshow(img_decoded)
                im = autonav.plot_costmap(ax2, show_arcs=True)
                #plt.colorbar(im)  # FIXME: makes a new colorbar every time
                plt.draw()
                plt.pause(.001)

            time.sleep(autonav.arc_duration)
            idx += 1
    
    except KeyboardInterrupt:
        # Restore to original state
        client.reset()
        client.enableApiControl(False)


    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



