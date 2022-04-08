
import os
# import yaml
import pickle
import rospkg
import open3d as o3d
import numpy as np

# def save_dict(yaml_dict, yaml_name):
#     yaml_path = os.path.join(os.getcwd(), yaml_name)
#     print(yaml_path)
#     with open(yaml_path, 'w') as f:
#         yaml.dump(yaml_dict, f, default_flow_style=False)
#     print(yaml_name + ' saved ...')
#     return

def save_dict_pkl(pkl_dict, pkl_name):
    pkl_path = os.path.join('/root/ocrtoc_ws', pkl_name)
    print(pkl_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_dict, f)
    print(pkl_name + ' saved ...')
    return

def get_obj_points(mesh_name):
    rospack = rospkg.RosPack()
    MESH_PARENT_PATH = os.path.join(rospack.get_path('ocrtoc_materials'), 'models')
    NUM_POINTS = 100
    mesh_path = os.path.join(MESH_PARENT_PATH, mesh_name, 'visual.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    points = np.asarray(mesh.vertices)
    return points

def get_obj_dimensions(obj_name):
    obj_vertices = get_obj_points(obj_name)
    obj_dim = np.max(obj_vertices, axis=0) - np.min(obj_vertices, axis=0)
    obj_dim = obj_dim.tolist()
    return obj_dim

def get_dimensions(goal_cartesian_pose_dic):
    obj_dim_dict = {}
    for obj_key in goal_cartesian_pose_dic.keys():
        obj_name = obj_key.split('_v')[0]
        obj_dim_dict[obj_key] = get_obj_dimensions(obj_name)
    return obj_dim_dict

def process_scene(pose_dic, goal_cartesian_pose_dic, available_grasp_pose_dic):
    print(' ################ Scene Classification happening here ################ ')

    # save_dict(pose_dic, 'pose_perception.yaml')
    save_dict_pkl(pose_dic, 'pose_perception.pkl')
    save_dict_pkl(goal_cartesian_pose_dic, 'goal_cartesian_pose_dic.pkl')
    save_dict_pkl(available_grasp_pose_dic, 'available_grasp_pose_dic.pkl')

    obj_dim_dict = get_dimensions(goal_cartesian_pose_dic)

    import pdb
    pdb.set_trace()
