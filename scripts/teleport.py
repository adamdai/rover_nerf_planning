"""Script to teleport a drone to target (x,y) location in the environment

"""

import airsim
import argparse

arg_parser = argparse.ArgumentParser(description="Teleport the rover to an (x,y) location")
arg_parser.add_argument("--x", type=float, help="x coordinate")   
arg_parser.add_argument("--y", type=float, help="y coordinate")   
args = arg_parser.parse_args()

client = airsim.CarClient()
client.confirmConnection()

pose = client.simGetVehiclePose()

pose.position.x_val = args.x
pose.position.y_val = args.y
pose.position.z_val = -10

client.simSetVehiclePose(pose, True)