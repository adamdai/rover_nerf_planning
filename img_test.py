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

    time.sleep(1)

    png_image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
    #png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
    print("captured image")
    # save image
    airsim.write_file(os.path.normpath('C:/Users/Adam/Documents/AirSim/image.png'), png_image)


    # # Restore to original state
    # client.reset()
    # client.enableApiControl(False)

    



