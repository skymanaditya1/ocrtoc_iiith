
import os
import yaml
import open3d as o3d
import numpy as np
import pickle
import math
import rospkg
from geometry_msgs.msg._Quaternion import Quaternion

# Constants

# path to mesh parent
rospack = rospkg.RosPack()
MESH_PARENT_PATH = os.path.join(rospack.get_path('ocrtoc_materials'), 'models')

# Point-cloud
VOXEL_SIZE = 0.01
OCTREE_DEPTH = 6

# Get Dimensions

def read_mesh(mesh_name):
    mesh_path = os.path.join(MESH_PARENT_PATH, mesh_name, 'visual.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh

def get_obj_points(mesh_name):
    mesh_path = os.path.join(MESH_PARENT_PATH, mesh_name, 'visual.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    points = np.asarray(mesh.vertices)
    return points

def get_obj_dimensions(obj_name):
    obj_vertices = get_obj_points(obj_name)
    obj_dim = np.max(obj_vertices, axis=0) - np.min(obj_vertices, axis=0)
    obj_dim = obj_dim.tolist()
    return obj_dim

def get_dimensions(path):
    obj_name_list = list(filter(lambda x: not x.endswith('yaml'), os.listdir(path)))
    obj_dim_dict = dict()
    for obj_name in obj_name_list:
        obj_dim_dict[obj_name] = get_obj_dimensions(obj_name)
    return obj_dim_dict

# Point-cloud related

def get_octree_size(node):
    count = 0
    if not isinstance(node, o3d.geometry.OctreeLeafNode):
        for child in node.children:
            if isinstance(child, o3d.geometry.OctreeNode):
                childCount = get_octree_size(child)
                count += childCount
    else:
        count = 1
    return count

def get_pcd_vol(pcd):

    # get volumne of point cloud as estimate of voxels
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
    # voxels = voxel_grid.get_voxels()

    octree = voxel_grid.to_octree(OCTREE_DEPTH)
    octree_count = get_octree_size(octree.root_node)

    return octree_count
