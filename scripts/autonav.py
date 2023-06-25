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

## -------------------------- PARAMS ------------------------ ##
# Unreal environment
# (in unreal units, 100 unreal units = 1 meter)
UNREAL_PLAYER_START = np.array([-117252.054688, 264463.03125, 25148.908203])
UNREAL_GOAL = np.array([210111.421875, 111218.84375, 32213.0])

GOAL = (UNREAL_GOAL - UNREAL_PLAYER_START)[:2] / 100.0
print(GOAL)


## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    # Initialize AutoNav class
    autonav = AutoNav()

    # Parameters
    throttle = 0.5
    N_iters = 10
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
            # Get image
            png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
            print("captured image")
            decoded = cv.imdecode(np.frombuffer(png_image, np.uint8), -1)
            # fig = px.imshow(decoded)
            # fig.show()
            # # display image in pyplot window
            # # img = cv.imread(png_image)
            # plt.imshow(decoded)
            # plt.show(block=False)

            autonav.update_costmap(decoded)
            arc, cost, w = autonav.replan()

            car_controls.steering = w / 1.6
            car_controls.throttle = throttle
            client.setCarControls(car_controls)
            print("steering: ", car_controls.steering, "throttle: ", car_controls.throttle)

            time.sleep(autonav.arc_duration)
            idx += 1
    
    except KeyboardInterrupt:
        # Restore to original state
        client.reset()
        client.enableApiControl(False)


    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



