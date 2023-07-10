import numpy as np
import math
import os
import json
import csv
import imageio
from sklearn.metrics import pairwise_distances

def get_intrinsic(imgdir):
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]

    H, W, C = imageio.imread(imgfiles[0]).shape
    vfov = 40

    focal_y = H / 2  / np.tan(np.deg2rad(vfov/2))
    focal_x = H / 2  / np.tan(np.deg2rad(vfov/2))

    return H, W, focal_x, focal_y
    
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def pad_rot(rot):
    padh = lambda x: np.hstack([x, np.zeros((x.shape[0], 1))])
    padv = lambda x: np.vstack([x, np.zeros((1, x.shape[1]))])

    rot_mat = padv(padh(rot))
    rot_mat[-1,-1] = 1
    return rot_mat


def quaternion_to_euler(q_w, q_x, q_y, q_z):
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*q_y*q_y - 2*q_z*q_z, 2*q_x*q_y - 2*q_w*q_z, 2*q_x*q_z + 2*q_w*q_y],
        [2*q_x*q_y + 2*q_w*q_z, 1 - 2*q_x*q_x - 2*q_z*q_z, 2*q_y*q_z - 2*q_w*q_x],
        [2*q_x*q_z - 2*q_w*q_y, 2*q_y*q_z + 2*q_w*q_x, 1 - 2*q_x*q_x - 2*q_y*q_y]
    ])

    # Extract Euler angles from rotation matrix
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return roll, pitch, yaw

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your meta')
    parser.add_argument("filename", type=str, help='file name')
    parser.add_argument("--imgdir", type=str, default='images', help='image directory name')
    
    return parser
    

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()
 
    data = {}

    with open(os.path.join(args.datadir, args.filename), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row

        data['cameraFrames'] = []

        for row in reader:
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

            roll, pitch, yaw = quaternion_to_euler(q_w, q_x, q_y, q_z)
            
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
                    'qvec': [q_w, q_x, q_y, q_z]
                },
                'image': image_file
                ,
                'timestamp': timestamp
            })
            



    GES_pos = np.array([[data['cameraFrames'][i]['position']['x'], 
                            data['cameraFrames'][i]['position']['y'],
                            data['cameraFrames'][i]['position']['z']] 
                        for i in range(len(data['cameraFrames']))])

    H, W, focal_x, focal_y = get_intrinsic(os.path.join(args.datadir, args.imgdir))

    nxyz = []
    frames = []
    
    image_list = np.sort(os.listdir(os.path.join(args.datadir, args.imgdir)))

    print("Found images...", len(image_list))
    
    print("Found entries...", len(data['cameraFrames']))
    for i in range(len(data['cameraFrames'])):
        position = data['cameraFrames'][i]['position']
        pos_x = position['x']
        pos_y = position['y']
        pos_z = position['z']
        xyz = np.array([pos_x, pos_y, pos_z])
        #print(xyz)
        
        rotation = qvec2rotmat(data['cameraFrames'][i]['rotation']['qvec'])
        # Rotate rotation matrix 
        airsim_transform = np.array([[1, 0, 0], 
                                     [0, 0, 1], 
                                     [0, -1, 0]])
        #rotation = airsim_transform @ rotation

        translation = xyz.reshape(3, 1)

        
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        # c2w[0:3, 1:3] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1

        # c2w[1, :], c2w[2, :] = c2w[2, :], c2w[1, :].copy()
        # c2w[0, :] = -c2w[0, :]
        # c2w[2, :] = -c2w[2, :]
        # c2w[1, :] = -c2w[1, :]

        c2w[:3,:3] = airsim_transform @ c2w[:3,:3]
        c2w[2,3] = 1.0
        print(c2w[:3, 3])

        nxyz.append(c2w[:3, :3].dot(np.array([0, 0, 1])))
        
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
    nxyz = np.array(nxyz)
    dists = np.sqrt(np.sum(nxyz**2, -1))

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
    # applied_transform = np.eye(4)[:3, :]
    # out["applied_transform"] = applied_transform.tolist()

    print("Saving...", os.path.join(args.datadir, 'transforms.json'))
    with open(os.path.join(args.datadir, 'transforms.json'), 'w', encoding="utf-8") as f: 
        json.dump(out, f, indent=4)