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


## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()

    time.sleep(0.2)

    mode = 'DEPTH'  # 'FRONT', 'BIRDSEYE', 'STEREO', 'DEPTH'
    path = 'C:/Users/Adam/NAVLAB/NeRF/terrain-nerf/data/airsim/images/'
    timestamp = str(time.time())

    if mode == 'FRONT':
        png_image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
        filename = 'front_' + timestamp + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'BIRDSEYE':
        png_image = client.simGetImage("BirdsEyeCamera", airsim.ImageType.Scene)
        filename = 'birdseye_' + timestamp + '.png'
        airsim.write_file(os.path.normpath(path + filename), png_image)

    elif mode == 'STEREO':
        left_image = client.simGetImage("StereoCameraLeft", airsim.ImageType.Scene)
        right_image = client.simGetImage("StereoCameraRight", airsim.ImageType.Scene)
        airsim.write_file(os.path.normpath(path + 'left_' + timestamp + '.png'), left_image)
        airsim.write_file(os.path.normpath(path + 'right_' + timestamp + '.png'), right_image)
    
    elif mode == 'DEPTH':
        # depth_image = client.simGetImage("FrontCamera", airsim.ImageType.DepthPlanar)
        # img_decoded = cv.imdecode(np.frombuffer(depth_image, np.uint8), -1)
        # print(np.unique(img_decoded))
        # depth_vis = client.simGetImage("FrontCamera", airsim.ImageType.DepthVis)
        responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
                                         airsim.ImageRequest("Disparity", airsim.ImageType.DisparityNormalized, pixels_as_float=True, compress=False)])
        camera_info = client.simGetCameraInfo("FrontCamera")
        print(camera_info.proj_mat)
        depth_float = np.array(responses[0].image_data_float)
        depth_float = depth_float.reshape(responses[0].height, responses[0].width)
        print(np.min(depth_float), np.max(depth_float))
        np.save(os.path.normpath(path + 'depth_' + timestamp + '.npy'), depth_float)
        depth_image = Image.fromarray(depth_float)
        depth_image = depth_image.convert("L")
        depth_image.save(os.path.normpath(path + 'depth_' + timestamp + '.png'))
        # filename = 'depth_' + str(time.time()) + '.png'
        # airsim.write_file(os.path.normpath(path + filename), depth_image)

    print("captured image")

    



