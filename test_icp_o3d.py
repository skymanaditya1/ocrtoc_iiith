
import open3d as o3d
import copy
import numpy as np
import open3d_plus as o3dp
import os
from tqdm import tqdm
from itertools import product
#Fast global registration or #ICP - local
#ICP here
def viewpoint_params_to_matrix(towards, angle):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix

def pcd_rotate_translate_setup( pcd, object_pose):
        
            x = object_pose[0]
            y = object_pose[1]
            z = object_pose[2]
            translation = [x, y, z]
            # R = object_pose['pose'][:3, :3]
            # rotation = self.rotationMatrixToEulerAngles(R)
        
            roll = object_pose[3]
            pitch = object_pose[4]
            yaw = object_pose[5]
            # print(roll, pitch, yaw)
            R = pcd.get_rotation_matrix_from_xyz((roll, pitch, yaw))
            # print(R)
            pcd_r = pcd.rotate(R, center=(0, 0, 0))
            
            
            pcd_t = pcd_r.translate((translation[0], translation[1], translation[2]))
            
            
            return pcd_t



def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def generate_angles(N):
    return np.arange(N) / N * 2.0 * np.pi

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp]) #,
                                    #   zoom=0.4459,
                                    #   front=[0.9288, -0.2951, -0.2242],
                                    #   lookat=[1.6784, 2.0612, 1.4451],
                                    #   up=[-0.3402, -0.9189, -0.1996])




pcd_file = "/scratch/shankara/ocrtoc_iiith/1-1-1.pcd"


folder_path = "/scratch/shankara/ocrtoc_iiith/ocrtoc_materials/models"
folder_path = os.path.join(folder_path, 'clear_box_1')
mesh_file = os.path.join(folder_path, 'textured.obj')
mesh = o3d.io.read_triangle_mesh(mesh_file)
pcd_mesh = mesh.sample_points_uniformly(20000)

object_pose = [0.3, 0.15, 1.5, 0.0, 0.0, 1.57]

pcd_mesh = pcd_rotate_translate_setup(pcd_mesh, object_pose)
mesh_pcd = copy.deepcopy(pcd_mesh)



source = mesh_pcd
target = o3d.io.read_point_cloud(pcd_file)

threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)


print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)



# o3d.visualization.draw_geometries([source1])




#get pose





ratio = reg_p2p.fitness
if ratio > 0:
    pose = reg_p2p.transformation

source1 = source
source1.transform(pose)


# generating template rotations
num_views = 50
num_angles = 12
views = generate_views(num_views)
angles = generate_angles(num_angles)



for view, angle in tqdm(product(views, angles), 'template matrix'):
        init_rotation_matrix = viewpoint_params_to_matrix(view, angle)
        