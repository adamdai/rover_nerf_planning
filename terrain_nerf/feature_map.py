"""Feature map class

"""

import numpy as np



def px_to_global(depth, cam_pose, cam_params, px):
    """Determine global coordinates for pixel coordinate
    
    Parameters
    ----------
    cam_pose : np.array (4, 3)
        Camera pose in global frame (R, t)
    
    """
    i, j = px
    z = depth[i,j]
    x = (j - cam_params['cx']) * z / cam_params['fx']
    y = (i - cam_params['cy']) * z / cam_params['fy']
    local_xyz = np.array([z, x, y])
    global_xyz = np.dot(cam_pose[:3,:3], local_xyz) + cam_pose[:3,3]
    return global_xyz


def depth_to_global(depth, cam_pose, cam_params, depth_thresh=50.0, patch_size=20):
    """Determine global coordinates for depth image
    
    Parameters
    ----------
    cam_pose : np.array (4, 3)
        Camera pose in global frame (R, t)
    
    """
    w, h = depth.shape

    I, J = np.mgrid[0:w:patch_size, 0:h:patch_size]

    I = I.flatten()
    J = J.flatten()
    D = depth[0:w:patch_size, 0:h:patch_size].flatten()
    I = I[D < depth_thresh]
    J = J[D < depth_thresh]
    D = D[D < depth_thresh]

    G = np.zeros((D.shape[0], 5))
    G[:,0] = D
    G[:,1] = (J - cam_params['cx']) * D / cam_params['fx']
    G[:,2] = (I - cam_params['cy']) * D / cam_params['fy']
    G[:,3] = I
    G[:,4] = J
    return G



class FeatureMap:
    """Feature map class

    Given global image, generate feature map for local planning

    """
    def __init__(self, img, start_px, goal_px, start_unreal, goal_unreal):
        """Initialize feature map from global image

        Parameters
        ----------
        img : np.array (N x M x 3)
        
        """
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.start_px = start_px
        self.goal_px = goal_px

        goal_global = (goal_unreal - start_unreal)[:2] / 100.0
        self.scale = np.abs(goal_global[0] / (start_px[1] - goal_px[1]))  # meters per pixel
        x_min, y_min = self.img_to_global(0, 0)
        x_max, y_max = self.img_to_global(self.height, self.width)
        self.bounds = [x_min, x_max, y_min, y_max] 


    def img_to_global(self, img_x, img_y):
        """Convert image coordinates to global coordinates

        """
        y = self.scale * (img_x - self.start_px[0])
        x = self.scale * (img_y - self.start_px[1])
        return x, y

    def get_img_feature(self, x, y):
        """Get image feature for global coordinates (x,y)

        """
        img_x = int(y / self.scale) + self.start_px[0]
        img_y = int(x / self.scale) + self.start_px[1]
        return self.img[img_x, img_y, :]
    

