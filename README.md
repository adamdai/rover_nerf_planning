# NeRF-based Path Planning for Planetary Rovers

This repo contains code for Neural Radiance Field (NeRF)-based path planning for planetary rovers. 

We train a NeRF of the extraterrestrial surface based on prior imagery, and store it on the rover for onboard use as a global map.
The rover runs a local planner for collision avoidance and global planner for high-level waypoint routing, and we use the 
NeRF map to update the global costmap based on local observations.





## Setup

Clone the GitHub repository:

    git clone https://github.com/adamdai/rover_nerf_planning.git

Create and activate conda environment:

    conda create -n rover_nerf python=3.8   
    conda activate rover_nerf
    
Install dependencies:

    pip install -r requirements.txt
    pip install -e .

## NeRF training

We use [nerfstudio]((https://docs.nerf.studio/en/latest/)) for NeRF training, rendering, evaluation, and other utilities.
Download checkpoint file.

## Jupyter notebooks

The `/notebooks` folder contains python notebooks for testing different parts of the pipeline.

## AirSim Moon simulation

We use [AirSim](https://microsoft.github.io/AirSim/) and [Unreal Engine](https://www.unrealengine.com/en-US) for simulation. 

Install Unreal Engine (Windows or Linux)
Install AirSim
Setup custom Unreal environment

For AirSim, we need to create a separate conda environment, since there is some incompatibility with the jupyter and airsim package dependencies.

    conda create -n airsim_rover python=3.8
    conda activate airsim_rover
    pip install -r airsim_requirements.txt

Run script
