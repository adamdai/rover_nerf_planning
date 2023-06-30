"""Feature map class

"""

import numpy as np


class FeatureMap:
    """Feature map class

    Given global image, generate feature map for local planning

    """
    def __init__(self, img, start_px, scale):
        """Initialize feature map from global image
        
        """
        self.costmap = costmap
        self.width = costmap.shape[0]
        self.height = costmap.shape[1]