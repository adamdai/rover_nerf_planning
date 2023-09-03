"""Cost interpolation functions

"""

import numpy as np
from sklearn.neighbors import KernelDensity


def kde_interp_cost(sample_features, costs, kernel='linear', bandwidth=0.2, **kwargs):
    """Interpolate cost values with Kernel Density Estimation
    
    Parameters
    ----------
    sample_features : np.array (N, 5)
        sample features (x, y, r, g, b)
    costs : np.array (N,)

    Returns
    -------

    
    """
    kde = KernelDensity(kernel='linear', bandwidth=0.5).fit(sample_features, costs)