# import pymesh

# print("loading mesh...")
# mesh = pymesh.load_mesh("../data/3dmodels/NewMap.OBJ")
# print(mesh.num_vertices)

# import openmesh

# mesh = openmesh.read_trimesh("../data/3dmodels/NewMap.OBJ")
# print(mesh.n_vertices())

# points = mesh.points()
# vertices = mesh.vertices()

import open3d as o3d
print("loading mesh...")
o3mesh = o3d.io.read_triangle_mesh("../data/3dmodels/NewMap.OBJ")
print("read mesh")