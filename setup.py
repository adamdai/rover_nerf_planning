#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='terrain_nerf',
    version='1.0',
    description='Terrain NeRF for Path Planning',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/terrain-nerf',
    packages=find_packages(),
)