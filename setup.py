#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='rover_nerf',
    version='1.0',
    description='Neural Radiance Field (NeRF) based path planning for planetary rovers',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/rover_nerf_planning',
    packages=find_packages(),
)