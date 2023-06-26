"""
General utilities

"""

import numpy as np


def wrap_angle(a):
    """
    Wrap angle to [-pi, pi)

    """
    return (a + np.pi) % (2 * np.pi) - np.pi