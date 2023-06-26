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

    time.sleep(0.2)

    mode = 'STEREO'  # 'FRONT', 'BIRDSEYE', 'STEREO'
    path = 'C:/Users/Adam/NAVLAB/NeRF/terrain-nerf/data/airsim/images/'
    timestamp = str(time.time())

    if mode == 'FRONT':
        png_image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
        filename = 'front_' + str(time.time()) + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'BIRDSEYE':
        png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
        filename = 'birdseye_' + str(time.time()) + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'STEREO':
        left_image = client.simGetImage("StereoCameraLeft", airsim.ImageType.Scene)
        right_image = client.simGetImage("StereoCameraRight", airsim.ImageType.Scene)
        airsim.write_file(os.path.normpath(path + 'left_' + str(time.time()) + '.png'), left_image)
        airsim.write_file(os.path.normpath(path + 'right_' + str(time.time()) + '.png'), right_image)

    print("captured image")

    



