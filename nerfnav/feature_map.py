"""Feature map class

"""

import numpy as np
import cv2 as cv

from nerfnav.utils import rgb2gray


class FeatureMap:
    """Feature map class

    Given global image, generate feature map for local planning

    """
    def __init__(self, img, start_px, goal_px, start_unreal, goal_unreal):
    #def __init__(self, img, start_px, scale):
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
        # self.scale = scale
        x_min, y_min = self.img_to_global(self.height, 0)
        x_max, y_max = self.img_to_global(0, self.width)
        self.bounds = [x_min, x_max, y_min, y_max] 


    def in_bounds(self, x, y):
        """Check if global coordinates are within bounds

        """
        return x >= self.bounds[0] and x <= self.bounds[1] and y >= self.bounds[2] and y <= self.bounds[3]


    def img_to_global(self, img_x, img_y):
        """Convert image coordinates to global coordinates

        """
        y = self.scale * (-img_x + self.start_px[0])
        x = self.scale * (img_y - self.start_px[1])
        return x, y
    

    def global_to_img(self, x, y):
        """Convert global coordinates to image coordinates

        """
        img_x = int(-y / self.scale) + self.start_px[0]
        img_y = int(x / self.scale) + self.start_px[1]
        return img_x, img_y
    

    def get_img_coords(self, x):
        """Get image coordinates for global coordinates
        
        Parameters
        ----------
        x : np.array (N, 2)
            Global coordinates
        
        Returns
        -------
        np.array (N, 2)
            Image coordinates

        """
        img_x = np.round(-x[:,1] / self.scale).astype(int) + self.start_px[0]
        img_y = np.round(x[:,0] / self.scale).astype(int) + self.start_px[1]
        return np.array([img_x, img_y]).T


    def get_features(self, x):
        """
        
        Parameters
        ----------
        x : np.array (N, 2)
            Global coordinates

        Returns
        -------
        np.array (N, 3)
            RGB features

        """
        img_x = np.round(-x[:,1] / self.scale).astype(int) + self.start_px[0]
        img_y = np.round(x[:,0] / self.scale).astype(int) + self.start_px[1]
        return self.img[img_x, img_y, :]
    

    def cluster(self, k=3):
        """Cluster based on image

        """
        img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        norm = 20.0
        X, Y = np.meshgrid(np.linspace(0, self.width/norm, self.width), np.linspace(0, self.height/norm, self.height))
        values = np.float32(np.hstack((pixel_values, X.reshape((-1,1)), Y.reshape((-1,1)))))

        # Cluster global img with k-means
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv.kmeans(values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers[:,0:3])

        self.labels = labels.reshape((img.shape[0], img.shape[1]))
        self.num_clusters = k
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        # reshape back to the original image dimension
        self.segmented_image = segmented_image.reshape(img.shape)


    def init_costmap(self):
        """Generate costmap from image

        TODO: use image gradients

        """
        img_gray = rgb2gray(self.segmented_image)
        img_gray = img_gray / np.max(img_gray)
        cost = 1.0 - img_gray
        return cost
    

    def update_costmap(self, image, depth):
        """Update costmap from rover image

        """
        pass




class ImgFeatureMap:
    """Image feature map class

    Initialize RGB feature map from image

    """
    def __init__(self, img, start_px, goal_px, start_unreal, goal_unreal):
    #def __init__(self, img, start_px, scale):
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
        # self.scale = scale
        x_min, y_min = self.img_to_global(0, 0)
        x_max, y_max = self.img_to_global(self.height, self.width)
        self.bounds = [x_min, x_max, y_min, y_max] 



class CostMap:
    """Global costmap class

    Assumes fixed clusters.

    Attributes
    ----------
    mat : np.array (N x M)
        Costmap matrix
    cluster_labels : np.array (N x M)
        Cluster labels for each pixel
    num_clusters : int
        Number of clusters
    cluster_idxs : list of np.array
        List of indexes for each cluster

    """
    def __init__(self, mat, cluster_labels, cluster_masks):
        self.cluster_labels = cluster_labels.astype(int)
        #self.num_clusters = len(cluster_masks)
        self.num_clusters = np.max(self.cluster_labels) + 1
        self.cluster_masks = cluster_masks
        self.mat = mat

        self.cluster_idxs = []
        for i in range(self.num_clusters):
            self.cluster_idxs.append(np.array(np.where(self.cluster_labels == i)).T)

    # def __call__(self, x, y):
    #     return self.vals[int(self.clusters[x,y])]