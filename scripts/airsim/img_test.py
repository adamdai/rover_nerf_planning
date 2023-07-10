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
from PIL import Image
import argparse

from terrain_nerf.airsim_utils import airsim_pose_to_Rt

arg_parser = argparse.ArgumentParser(description="Take an image in airsim")
arg_parser.add_argument("--mode", type=str, help="image mode (front, birdseye, stereo, depth)")   
args = arg_parser.parse_args()

## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":
    # TODO: add arg for mode

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()

    time.sleep(0.2)

    #mode = 'FRONT'  # 'FRONT', 'BIRDSEYE', 'STEREO', 'DEPTH'
    mode = args.mode
    path = 'C:/Users/Adam/NAVLAB/NeRF/terrain-nerf/data/airsim/images/'
    timestamp = str(time.time())

    if mode == 'front':
        png_image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
        filename = 'front_' + timestamp + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'birdseye':
        png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
        filename = 'birdseye_' + timestamp + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'stereo':
        left_image = client.simGetImage("StereoCameraLeft", airsim.ImageType.Scene)
        right_image = client.simGetImage("StereoCameraRight", airsim.ImageType.Scene)
        airsim.write_file(os.path.normpath(path + 'left_' + timestamp + '.png'), left_image)
        airsim.write_file(os.path.normpath(path + 'right_' + timestamp + '.png'), right_image)
    
    elif mode == 'depth':
        responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
                                         airsim.ImageRequest("Disparity", airsim.ImageType.DisparityNormalized, pixels_as_float=True, compress=False)])
        
        camera_info = client.simGetCameraInfo("Depth")

        depth_float = np.array(responses[0].image_data_float)
        depth_float = depth_float.reshape(responses[0].height, responses[0].width)
        np.save(os.path.normpath(path + 'depth_' + timestamp + '.npy'), depth_float)
        depth_image = Image.fromarray(depth_float)
        depth_image = depth_image.convert("L")
        depth_image.save(os.path.normpath(path + 'depth_' + timestamp + '.png'))

    print("captured image")

    



