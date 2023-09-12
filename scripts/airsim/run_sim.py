"""Run full simulation

Scenarios:
    Test scenario:
        Player start = (-117252.054688, 264463.03125, 25148.908203)
        Goal = (-83250.0, 258070.0, 24860.0)
    Scenario 2:
        Player start = (-38344.066406, 21656.935547, 30672.384766)
        Goal = (75536.0, 102175.0, 25713.0)

"""

#import airsim_data_collection.common.setup_path
import airsim
import os
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 as cv
from PIL import Image
import pickle

from nerfnav.autonav import AutoNav
from nerfnav.airsim_utils import get_pose2D, airsim_pose_to_Rt
from nerfnav.feature_map import FeatureMap, CostMap
from nerfnav.global_planner import GlobalPlanner

## -------------------------- PARAMS ------------------------ ##
# Unreal environment (FIXME: y inverted)
# (in unreal units, 100 unreal units = 1 meter)
UNREAL_PLAYER_START = np.array([-117252.054688, -264463.03125, 25148.908203])
UNREAL_GOAL = np.array([-83250.0, -258070.0, 24860.0])

GOAL_POS = (UNREAL_GOAL - UNREAL_PLAYER_START)[:2] / 100.0
print("GOAL_POS: ", GOAL_POS)

AUTONAV_REPLAN = 7.5  # seconds
PLAN_TIME = 7.0  # seconds
THROTTLE = 0.35
MAX_ITERS = 1e5
GOAL_TOLERANCE = 20  # meters

VISUALIZE = True
REPLAN = True
RECORD = False
DEBUG = True

# MPL text color
COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

## -------------------------- SETUP ------------------------ ##
global_img = cv.imread('../../data/airsim/images/test_scenario.png')
global_img = global_img[::2, ::2, :]  # downscale
#global_img = global_img[::4, ::4, :]  # downscale
start_px = (138, 141)
goal_px = (78, 493)
# start_px = (70, 70)
# goal_px = (38, 248)

costmap_data = np.load('../../data/airsim/costmap.npz', allow_pickle=True)
costmap = CostMap(costmap_data['mat'], costmap_data['cluster_labels'], costmap_data['cluster_masks'])

feat_map = FeatureMap(global_img, start_px, goal_px, UNREAL_PLAYER_START, UNREAL_GOAL)
global_planner = GlobalPlanner(costmap, feat_map, goal_px, interp_method='krr', interp_features='rgb')
nav_goal = global_planner.replan(np.zeros(3))[1]
if REPLAN:
    autonav = AutoNav(nav_goal, arc_duration=AUTONAV_REPLAN)
else:
    autonav = AutoNav(GOAL_POS, arc_duration=AUTONAV_REPLAN)

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
        f.set_facecolor(0.15 * np.ones(3))
        ax[0,0].set_title("RGB Image")
        ax[0,1].set_title("Depth Image")
        im3 = global_planner.plot(ax[1,1])
        ax[1,0].set_title("Local costmap")
        ax[1,0].set_xlabel("y (m)")
        ax[1,0].set_ylabel("x (m)")
        ax[1,1].set_title("Global costmap")
        ax[1,1].set_xlabel("x (m)")
        ax[1,1].set_ylabel("y (m)")
        plt.ion()
        plt.show()

    input("Press Enter to start...")

    idx = 0

    # brake the car
    car_controls.brake = 1
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    # wait until car is stopped
    time.sleep(1)

    # release the brake
    car_controls.brake = 0
    speed = 2.5

    # start recording data
    if RECORD:
        client.startRecording()

    try:
        while idx < MAX_ITERS:
            start_time = time.time()
            current_pose = get_pose2D(client)
            print("idx = ", idx)
            print("  current_pose: ", current_pose)
            print("  nav_goal: ", nav_goal)
            if np.linalg.norm(current_pose[:2] - GOAL_POS) < GOAL_TOLERANCE:
                print("Reached goal!")
                print("TOTAL COST: ", autonav.total_cost)
                break

            # Get depth image
            responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
            camera_info = client.simGetCameraInfo("Depth")
            cam_pose = airsim_pose_to_Rt(camera_info.pose)
            depth_float = np.array(responses[0].image_data_float)
            depth_float = depth_float.reshape(responses[0].height, responses[0].width)
            depth_float[depth_float > 100] = 100.0
            # Get RGB image
            image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
            image = cv.imdecode(np.frombuffer(image, np.uint8), -1)

            img_time = time.time()
            if DEBUG:
                print("  img capture time: ", img_time - start_time)

            cost_vals = autonav.update_costmap(current_pose, depth_float)
            # arc, cost, w = autonav.replan(current_pose)
            local_update_time = time.time()
            if DEBUG:
                print("  local cost update time: ", local_update_time - img_time)
            if REPLAN:
                global_planner.update_costmap(cost_vals)
                #global_planner.naive_update_costmap(cost_vals)
                global_update_time = time.time()
                if DEBUG:
                    print("  global cost update time: ", global_update_time - local_update_time)

            current_pose = get_pose2D(client)
            path = global_planner.replan(current_pose)
            global_replan_time = time.time()
            if DEBUG and REPLAN:
                print("  global replan time: ", global_replan_time - global_update_time)
            if len(path) > 1:
                nav_goal = path[1]
            else:
                nav_goal = GOAL_POS
            autonav.update_goal(nav_goal)

            current_pose = get_pose2D(client)
            arc, cost, w = autonav.replan(current_pose)
            local_replan_time = time.time()
            if DEBUG:
                print("  local replan time: ", local_replan_time - global_replan_time)

            plan_time = time.time() - start_time
            if DEBUG:
                print("  planning time: ", plan_time)

            # Drive for 7.5 seconds
            car_controls.brake = 0
            car_controls.steering = w / 1.2
            #speed = client.getCarState().speed
            car_controls.throttle = THROTTLE + np.clip(0.2 * (2.5 - speed), 0, 0.2)
            print("speed: ", speed, "throttle: ", car_controls.throttle)
            client.setCarControls(car_controls)

            if VISUALIZE:
                title_font = 16
                ax[0,0].clear(); ax[0,1].clear(); ax[1,0].clear(); ax[1,1].clear()
                ax[0,0].set_title("RGB Image", fontsize=title_font)
                ax[0,1].set_title("Depth Image", fontsize=title_font)
                ax[0,0].imshow(image)
                ax[0,1].imshow(depth_float)
                ax[0,0].axis('off')
                ax[0,1].axis('off')
                im2 = autonav.plot_costmap(ax[1,0], show_arcs=True)
                ax[1,0].set_title("Local costmap", fontsize=title_font)
                ax[1,0].set_xlabel("y (m)")
                ax[1,0].set_ylabel("x (m)")
                im3 = global_planner.plot(ax[1,1])
                ax[1,1].set_title("Global costmap", fontsize=title_font)
                ax[1,1].set_xlabel("x (m)")
                ax[1,1].set_ylabel("y (m)")
                #cbar3.set_clim(vmin=0, vmax=np.max(global_planner.costmap))
                # plt.colorbar(im3, ax=ax[1], fraction=0.05, aspect=10)  # FIXME: makes a new colorbar every time
                #plt.pause(autonav.arc_duration - PLAN_TIME)
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15, hspace=0.05)
                # if plan_time < AUTONAV_REPLAN:
                #     plt.pause(AUTONAV_REPLAN - plan_time)
                # else:
                #     plt.pause(0.01)
                plt.pause(AUTONAV_REPLAN)
            else:
                #time.sleep(autonav.arc_duration - PLAN_TIME)
                time.sleep(AUTONAV_REPLAN)
                # if plan_time < AUTONAV_REPLAN:
                #     time.sleep(AUTONAV_REPLAN - plan_time)


            speed = client.getCarState().speed
            
            # brake the car
            car_controls.brake = 1
            car_controls.throttle = 0
            client.setCarControls(car_controls)
            
            #print("steering: ", car_controls.steering, "throttle: ", car_controls.throttle)
            
            # if plan_time < PLAN_TIME:
            #     time.sleep(PLAN_TIME - plan_time)
            print("TOTAL COST: ", autonav.total_cost)
            print("--------------------------------------------------------------------------------")

            idx += 1
    
    except KeyboardInterrupt:
        if RECORD:
            client.stopRecording()

        print("TOTAL COST: ", autonav.total_cost)

        #print("cluster costs: ", global_planner.cluster_costs)
        with open('cluster_costs.pickle', 'wb') as handle:
            pickle.dump(global_planner.cluster_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Restore to original state
        client.reset()
        client.enableApiControl(False)

    if RECORD:
        client.stopRecording()
    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



