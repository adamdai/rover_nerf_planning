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


## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    # brake the car
    car_controls.brake = 1
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    # wait until car is stopped
    time.sleep(1)

    car_controls.brake = 0
    for i in range(10):

        png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
        print("captured image")
        decoded = cv.imdecode(np.frombuffer(png_image, np.uint8), -1)
        # print(type(decoded))
        fig = px.imshow(decoded)
        fig.show()
        # # display image in pyplot window
        # # img = cv.imread(png_image)
        # plt.imshow(decoded)
        # plt.show(block=False)

        # get vehicle position
        car_state = client.getCarState()
        print("car state: %s" % car_state)

        car_controls.steering = 0.0
        car_controls.throttle = 0.5
        client.setCarControls(car_controls)

        time.sleep(3)


    # try:
    #     print("driving routes")
    #     while(True):
    #         car_controls.steering = 0.5  #0.2 * np.sin(0.01 * count)
    #         car_controls.throttle = 1.0
    #         client.setCarControls(car_controls)


    # except KeyboardInterrupt:
    #     # Restore to original state
    #     client.reset()
    #     client.enableApiControl(False)


    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



