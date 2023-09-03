"""
Generate NeRF features for a given scene

Run from /nerstudio_ws directory

"""

import torch
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle


torch.manual_seed(42)

#%% ------------------------------------ Parameters ------------------------------------ %%

# Moon Test Scenario
config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/MoonTestScenario/nerfacto/2023-07-21_013712/config.yml'))
save_folder = "MoonTestScenario/features/"

print(f"Using checkpoint_path: {checkpoint_path}")
print(f"Will save to {save_folder}")


#%% ------------------------------------ Functions ------------------------------------ %%

def nerf_render_rays(origins, directions):
    """Accumulated NeRF outputs for N rays

    Parameters
    ----------
    origins : np.array (N, 3)
        Ray origins
    directions : np.array (N, 3)
        Ray direction vectors

    """
    # Leave as default
    pixel_area = torch.ones_like(origins[..., :1])
    camera_indices = torch.zeros_like(origins[..., :1])

    ray_bundle = RayBundle(origins=origins, directions=directions, 
                           pixel_area=pixel_area, camera_indices=camera_indices)
    
    # Sets the near and far properties
    ray_bundle = pipeline.model.collider(ray_bundle)
    
    return pipeline.model(ray_bundle)


#%% ------------------------------------ Main ------------------------------------ %%

if __name__ == "__main__":

    # Grid of xy points in NeRF coordinates ([-1, 1] x [-1, 1])
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xyz = torch.stack([xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())], dim=-1)
    origins = torch.tensor(xyz, device=pipeline.device)

    N_rays = len(origins)

    # origins = torch.zeros((N_rays, 3), device=pipeline.device)

    # All rays pointing down
    directions = torch.tensor([[0, 0, 1]], device=pipeline.device).repeat(N_rays, 1)

    print("Generating features for NeRF coordinates")
    rgbd = nerf_render_rays(origins, directions)

    print(rgbd['rgb'])
    print(rgbd['accumulation'])
    print(rgbd['depth'])


