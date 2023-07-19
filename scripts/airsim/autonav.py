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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 as cv
from PIL import Image

from terrain_nerf.autonav import AutoNav
from terrain_nerf.airsim_utils import get_pose2D, airsim_pose_to_Rt
from terrain_nerf.feature_map import FeatureMap, CostMap
from terrain_nerf.global_planner import GlobalPlanner

## -------------------------- PARAMS ------------------------ ##
# Unreal environment
# (in unreal units, 100 unreal units = 1 meter)
UNREAL_PLAYER_START = np.array([-117252.054688, 264463.03125, 25148.908203])
UNREAL_GOAL = np.array([-83250.0, 258070.0, 24860.0])

GOAL_POS = (UNREAL_GOAL - UNREAL_PLAYER_START)[:2] / 100.0
print("GOAL_POS: ", GOAL_POS)

VISUALIZE = True
REPLAN = True
RECORD = True

## -------------------------- SETUP ------------------------ ##
global_img = cv.imread('../../data/airsim/images/test_scenario_3.png')
global_img = global_img[::2, ::2, :]
start_px = (138, 141)
goal_px = (78, 493)

costmap_data = np.load('../../data/airsim/costmap.npz')
costmap = CostMap(costmap_data['mat'], costmap_data['clusters'], costmap_data['vals'])

feat_map = FeatureMap(global_img, start_px, goal_px, UNREAL_PLAYER_START, UNREAL_GOAL)
global_planner = GlobalPlanner(costmap, feat_map, goal_px)
nav_goal = global_planner.replan(np.zeros(3))[1]
if REPLAN:
    autonav = AutoNav(nav_goal)
else:
    autonav = AutoNav(GOAL_POS)

## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    path_idx = 0
    current_pose = get_pose2D(client)

    # Visualization
    if VISUALIZE:
        f, ax = plt.subplots(2, 2)
        # Set figure size
        f.set_figwidth(16)
        f.set_figheight(9)
        ax[0,0].set_title("RGB Image")
        ax[0,1].set_title("Depth Image")
        im3 = global_planner.plot(ax[1,1])
        plt.ion()
        #plt.show()

    # Parameters
    throttle = 0.4
    N_iters = 1e5
    idx = 0

    # brake the car
    car_controls.brake = 1
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    # wait until car is stopped
    time.sleep(1)

    # release the brake
    car_controls.brake = 0

    # start recording data
    if RECORD:
        client.startRecording()

    try:
        while idx < N_iters:
            start_time = time.time()
            current_pose = get_pose2D(client)
            print("idx = ", idx)
            print(" current_pose: ", current_pose)
            print(" nav_goal: ", nav_goal)
            if np.linalg.norm(current_pose[:2] - GOAL_POS) < 10:
                print("Reached goal!")
                break

            # Get depth image
            responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
            camera_info = client.simGetCameraInfo("Depth")
            cam_pose = airsim_pose_to_Rt(camera_info.pose)
            depth_float = np.array(responses[0].image_data_float)
            depth_float = depth_float.reshape(responses[0].height, responses[0].width)
            # Get RGB image
            image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
            image = cv.imdecode(np.frombuffer(image, np.uint8), -1)

            cost_vals = autonav.update_costmap(current_pose, depth_float)
            if REPLAN:
                global_planner.update_costmap(cost_vals)
            path = global_planner.replan(current_pose)
            if len(path) > 1:
                nav_goal = path[1]
            else:
                nav_goal = GOAL_POS
            autonav.update_goal(nav_goal)
            arc, cost, w = autonav.replan(current_pose)

            car_controls.steering = w / 1.6
            car_controls.throttle = throttle
            client.setCarControls(car_controls)
            #print("steering: ", car_controls.steering, "throttle: ", car_controls.throttle)
            print(" planning time: ", time.time() - start_time)
            print("--------------------------------------------------------------------------------")

            if VISUALIZE:
                #ax1.clear(); ax2.clear(); ax3.clear()
                ax[0,0].clear(); ax[0,1].clear(); ax[1,0].clear(); ax[1,1].clear()
                ax[0,0].set_title("RGB Image")
                ax[0,1].set_title("Depth Image")
                ax[0,0].imshow(image)
                depth_image = Image.fromarray(depth_float)
                depth_image = depth_image.convert("L")
                ax[0,1].imshow(depth_image)
                im2 = autonav.plot_costmap(ax[1,0], show_arcs=True)
                ax[1,0].set_title(f"Local costmap \n Max cost = {np.max(autonav.costmap)}")
                ax[1,0].set_xlabel("y (m)")
                ax[1,0].set_ylabel("x (m)")
                im3 = global_planner.plot(ax[1,1])
                ax[1,1].set_title(f"Global costmap \n Max cost = {np.max(global_planner.costmap.mat)}")
                ax[1,1].set_xlabel("x (m)")
                ax[1,1].set_ylabel("y (m)")
                #cbar3.set_clim(vmin=0, vmax=np.max(global_planner.costmap))
                # plt.colorbar(im3, ax=ax[1], fraction=0.05, aspect=10)  # FIXME: makes a new colorbar every time
                plt.pause(autonav.arc_duration)
            else:
                time.sleep(autonav.arc_duration)
            idx += 1
    
    except KeyboardInterrupt:
        if RECORD:
            client.stopRecording()
        # Restore to original state
        client.reset()
        client.enableApiControl(False)

    if RECORD:
        client.stopRecording()
    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



