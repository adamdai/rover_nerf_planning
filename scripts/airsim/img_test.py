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
    path = 'C:/Users/Adam/NAVLAB/Neural_Reps/terrain-nerf/data/airsim/images/rover/'
    timestamp = str(time.time())

    if mode == 'front':
        # responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
        # response = responses[0]
        # # get numpy array
        # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # print(img1d.shape)

        # # reshape array to 4 channel image array H X W X 4
        # img_rgb = img1d.reshape(response.height, response.width, 3)

        # # original image is fliped vertically
        # img_rgb = np.flipud(img_rgb)

        # # write to png 
        # airsim.write_png(os.path.normpath('test.png'), img_rgb) 

        responses = client.simGetImages([airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene, pixels_as_float=False, compress=False),
                                         airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)],
                                        vehicle_name="Rover")
        image_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        image_data = image_data.reshape(responses[0].height, responses[0].width, 3)
        airsim.write_png(os.path.normpath(path + 'front_' + timestamp + '.png'), image_data)

        depth_data = np.array(responses[1].image_data_float)
        depth_data = depth_data.reshape(responses[1].height, responses[1].width)
        np.save(os.path.normpath(path + 'depth_' + timestamp + '.npy'), depth_data)
        
        #png_image = client.simGetImage("FrontCamera", airsim.ImageType.Scene, vehicle_name="Rover")
        #Image.frombytes("RGB", (800, 600), png_image).save("test.png")


        # filename = 'front_' + timestamp + '.png'
        # airsim.write_file(os.path.normpath(path + filename), png_image)

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
        # responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
        #                                  airsim.ImageRequest("Disparity", airsim.ImageType.DisparityNormalized, pixels_as_float=True, compress=False)])
        responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
        
        camera_info = client.simGetCameraInfo("Depth")

        depth_float = np.array(responses[0].image_data_float)
        depth_float = depth_float.reshape(responses[0].height, responses[0].width)
        np.save(os.path.normpath(path + 'depth_' + timestamp + '.npy'), depth_float)
        depth_image = Image.fromarray(depth_float)
        depth_image = depth_image.convert("L")
        depth_image.save(os.path.normpath(path + 'depth_' + timestamp + '.png'))

    print("captured image")

    



