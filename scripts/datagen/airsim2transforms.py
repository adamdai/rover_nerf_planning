import numpy as np
import math
import os
import json
import csv
import imageio
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R


#%% Helper functions

def quat_to_R(quat):
    """Quaternion in scalar-last (x, y, z, w) format"""
    r = R.from_quat(quat)
    return r.as_matrix()


def euler_to_R(euler, seq='XYZ'):
    r = R.from_euler(seq, euler, degrees=True)
    return r.as_matrix()


def quat_to_euler(quat):
    """Quaternion in scalar-last (x, y, z, w) format"""
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=True)


def axis_angle_to_rot_mat(axis, angle):
    """Angle in radians"""
    axis = axis / np.linalg.norm(axis)
    q = np.hstack((axis * np.sin(angle/2), np.cos(angle/2)))
    return quat_to_R(q)


def get_intrinsic(imgdir):
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]

    H, W, C = imageio.imread(imgfiles[0]).shape
    vfov = 90

    focal_y = H / 2  / np.tan(np.deg2rad(vfov/2))
    focal_x = H / 2  / np.tan(np.deg2rad(vfov/2))

    return H, W, focal_x, focal_y
    

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your meta')
    parser.add_argument("--filename", type=str, default='airsim_rec.txt', help='file name')
    parser.add_argument("--imgdir", type=str, default='images', help='image directory name')
    parser.add_argument("--ds_rate", type=int, default=1, help='downsample rate')
    
    return parser
    

#%% Main

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()
 
    data = {}
    ds_rate = args.ds_rate

    with open(os.path.join(args.datadir, args.filename), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row

        data['cameraFrames'] = []

        for i, row in enumerate(reader):
            if i % ds_rate == 0:
                vehicle_name = row[0]
                timestamp = int(row[1])
                pos_x = float(row[2])
                pos_y = float(row[3])
                pos_z = float(row[4])
                q_w = float(row[5])
                q_x = float(row[6])
                q_y = float(row[7])
                q_z = float(row[8])
                image_file = row[9]

                roll, pitch, yaw = quat_to_euler([q_x, q_y, q_z, q_w])
                
                data['cameraFrames'].append({
                    'position': {
                        'x': pos_x,
                        'y': pos_y,
                        'z': pos_z
                    },
                    'rotation': {
                        'x': roll,
                        'y': pitch,
                        'z': yaw,
                        'qvec': [q_x, q_y, q_z, q_w]
                    },
                    'image': image_file
                    ,
                    'timestamp': timestamp
                })

    H, W, focal_x, focal_y = get_intrinsic(os.path.join(args.datadir, args.imgdir))

    frames = []
    image_list = np.sort(os.listdir(os.path.join(args.datadir, args.imgdir)))
    print("Found images...", len(image_list))
    print("Found entries...", len(data['cameraFrames']))

    for i in range(len(data['cameraFrames'])):
        position = data['cameraFrames'][i]['position']
        pos_x = position['x']
        pos_y = position['y']
        pos_z = position['z']
        xyz = np.array([pos_x, pos_y, -pos_z])

        roll = data['cameraFrames'][i]['rotation']['x']
        pitch = data['cameraFrames'][i]['rotation']['y']
        yaw = data['cameraFrames'][i]['rotation']['z']
        print("Roll", roll, "Pitch", pitch, "Yaw", yaw)
        
        # Swap pitch and roll
        rotation = euler_to_R([pitch, roll, yaw], seq='xyz')
        translation = xyz.reshape(3, 1)
        
        c2w = np.concatenate([rotation, translation], 1)
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], 0)

        # Align camera
        x_axis = c2w[:3, 0]
        init_R = axis_angle_to_rot_mat(x_axis, np.deg2rad(90))  # Local 90 X
        init_R = euler_to_R([-90], seq='z') @ init_R  # Global 90 Z

        c2w[:3,:3] = init_R @ c2w[:3,:3]
        
        if not os.path.exists(os.path.join(args.datadir, args.imgdir, data['cameraFrames'][i]['image'])):
            print("Image not found", os.path.join(args.imgdir, data['cameraFrames'][i]['image']))
            continue

        frame = {
            "file_path": os.path.join(args.imgdir, data['cameraFrames'][i]['image']),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": i,
        }
        
        frames.append(frame)

    print("Frames: ", len(frames))

    out = {
        "w": W,
        "h": H,
    }
    
    out["fl_x"] = focal_x
    out["fl_y"] = focal_y
    out["cx"] = W/2
    out["cy"] = H/2
    out["k1"] = 0.0
    out["k2"] = 0.0
    out["p1"] = 0.0
    out["p2"] = 0.0
        
    out["frames"] = frames

    print("Saving...", os.path.join(args.datadir, 'transforms.json'))
    with open(os.path.join(args.datadir, 'transforms.json'), 'w', encoding="utf-8") as f: 
        json.dump(out, f, indent=4)