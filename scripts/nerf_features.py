from nerfstudio.utils.eval_utils import eval_setup
import torch
from pathlib import Path
from nerfstudio.cameras.rays import RayBundle
# from nerfstudio.field_components.field_heads import FieldHeadNames

import math

torch.manual_seed(42)

################################
# USER PARAMS
################################
num_rays = 6000
num_nerf_samples = 1024

# z_shift = -0.36
z_shift = -1.5
# z_shift = -1.7
# z_shift = -2


x_points = [0.05, 0.0, -0.05, -0.1, -0.15]
y_points = [0.0, -0.05, -0.1, -0.15, -0.2, -0.25]

# x_offset = 0
# y_offset = 0

# San Jose Low
x_offset = 0.58
y_offset = -0.03


################################
# SCRIPT FUNCTIONS
################################

# Load in the model (inside pipeline)

# GSDC NeRF in San Jose (25 m up)
# config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/GSDCSanJose/nerfacto/2023-06-22_195714/config.yml'))
# save_folder = "GSDCSanJose/torch_outputs_full_sphere/"

# Durand NeRF (not actually centered at Durand)
# config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/Durand/nerfacto/2023-06-13_000653/config.yml'))
# save_folder = "Durand/torch_outputs_full_sphere_z_shift--2/"

# San Jose AKA GSDC NeRF (100 m up)
config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/San_Jose/nerfacto/2023-06-23_184307/config.yml'))
save_folder = "San_Jose/torch_outputs_full_sphere_low_med/"

print(f"Using checkpoint_path: {checkpoint_path}")
print(f"Will save to {save_folder}")


def fibonacci_sphere(samples: int):
    """
    Create points uniformly spread out on a sphere. This is _NOT_ random. 
    """
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y_ = 1 - (i / float(samples - 1)) * 2  # y_ goes from 1 to -1
        radius = math.sqrt(1 - y_ * y_)  # radius at y_

        theta = phi * i  # golden angle increment

        x_ = math.cos(theta) * radius
        z_ = math.sin(theta) * radius

        points.append((x_, y_, z_))

    return points


def main_gen_outputs(x_: float, y_: float, z_: float = z_shift):

    # Set the origin at the NeRF origin
    origins = torch.zeros((num_rays, 3), device=pipeline.device)
    origins[:, 0] = x_
    origins[:, 1] = y_
    origins[:, 2] = z_

    # Set the directions based on the Fibonacci Sphere
    raw_directions = fibonacci_sphere(num_rays)   # Just a normal python list
    directions = torch.tensor(raw_directions, device=pipeline.device)

    # If you only want the top points
    # directions[..., 2] = torch.abs(directions[..., 2])

    # Directions should already be normalized, but just to be sure
    directions = directions / directions.norm(dim=-1, keepdim=True) 

    # Leave as default
    pixel_area = torch.ones_like(origins[..., :1])
    camera_indices = torch.zeros_like(origins[..., :1])

    # Make the RayBundle to query the NeRF
    ray_bundle = RayBundle(
                origins=origins, directions=directions, 
                pixel_area=pixel_area, camera_indices=camera_indices
            )

    # Sets the near and far properties
    ray_bundle = pipeline.model.collider(ray_bundle)

    # Update the number of samples of the last layer so that we get 
    # a finer resolution on the density profile
    if num_nerf_samples > 0:
        pipeline.model.proposal_sampler.num_nerf_samples_per_ray = 1024

    # RANDOM HERE
    # Sample the Nerfacto proposal sampler and get the samples along the rays
    # the weights, and the ray samples at each level as a list
    ray_samples, weights_list, ray_samples_list = pipeline.model.proposal_sampler(
        ray_bundle, density_fns=pipeline.model.density_fns)

    # Call the NeRF at the ray samples
    # field_outputs = pipeline.model.field.forward(ray_samples, compute_normals=pipeline.model.config.predict_normals)

    # DENSITY 1: Directly from the field forward pass
    # density_1 = field_outputs[FieldHeadNames.DENSITY]

    # THIS IS A DIFFERENT DENSITY
    # pipeline.model.density_fns[-1](ray_samples.frustums.get_positions())

    # DENSITY 2: Use the helper funtion
    # (same numerically as Density 1)
    density_2, _ = pipeline.model.field.get_density(ray_samples)

    # Run volume rendering to get the pixel values
    pixel_values_rgbd_acc = pipeline.model(ray_bundle)

    # These are things to save that are just normal tensors
    out_save_tensors = {
        "Origins": origins, 
        "Directions": directions, 
        "weight_list": weights_list, 
        "Density": density_2,
        "rgbd_pixel": pixel_values_rgbd_acc,
        "frustrum_positions": ray_samples.frustums.get_positions(),
        "ray_samples_deltas": ray_samples.deltas,
        "ray_samples_spacing_starts": ray_samples.spacing_starts,
        "ray_samples_spacing_ends": ray_samples.spacing_ends
        }

    # These are Nerfstudio objects
    # out_save_frustrum = {
    #     "ray_samples_frustrums": ray_samples.frustums
    #     }

    # Save each
    extension = ".pt"
    descriptor = f"_rays-{num_rays}_nerfsamp-{num_nerf_samples}_fib_sphere"
    loc_descriptor = f"_x-{int(x_ * 1000)}_y-{int(y_ * 1000)}_z-{int(z_ * 1000)}"
    full_descriptor = descriptor + loc_descriptor + extension

    torch.save(out_save_tensors, save_folder + "density" + full_descriptor)
    # torch.save(out_save_frustrum, save_folder + "frustrums" + full_decriptor)

    return True


################################
# SCRIPT FUNCTIONS
################################

for x in x_points:
    for y in y_points:
        main_gen_outputs(x + x_offset, y + y_offset)

print("Completed")
