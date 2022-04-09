'''Stack-relationship detector using Ray Casting/Tracing
Intuition: In order to understand which objects are placed on top of which other objects, we can 
simply cast a one ray from each point of a given object vertically downward and among the set of objects 
that are hit by these rays, find the one that is closest to the current object. 

Functionality:
Multiple objects can be under a given object, so we shall maintain a stack for each object which tells 
us which objects are to be placed before we place this object in its destination.

NOTE: RAY CASTING exists only in open3d version>=0.14.0 (Here, it is being tested with v0.15.2 of open3d)

Team Lumos
'''

from unicodedata import name
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from misc import Object
from rendering_scenes_in_o3d import get_object_list_from_yaml, save_pcd, draw_geometries
import copy
# import json
import pickle
import os
import sys
import argparse
import numpy as np

class SGNode:
    def __init__(self, object_name, mesh_id, parent_mesh_ids, parents):
        '''
        object_name: Current object name
        parents: Parents' node ids
        '''
        self.object = object_name
        self.mesh_id = mesh_id
        self.parent_mesh_ids = parent_mesh_ids
        self.parents = parents

# def get_a_graph_from_obj_dict(object_stacks):

    

def get_a_graph_from_obj_dict(object_stacks):
    '''
    Parameters:
    object_stacks: {
        'mesh_id': {
            'current_object_info': {
                'object': <object_label>,
                'mesh_id': <object_mesh_id>
            },
            'mesh_ids_of_objects_under_it': [i, j, ...],
            'objects_under_it': [obj_name1, ....]
        }
    }

    Return:
    stacks: list of stacks = [[stack_1 mesh ids (left to right ids indicate bottom to top in the stack)], 
                                [stack 2], ...etc]
    '''

    stacks = []
    # nodes = []
    node_dict = {}
    for i, key in enumerate(object_stacks.keys()):
        # Here key == mesh_id of the object
        obj_name = object_stacks[key]['current_object_info']['object']
        mesh_id = key
        parents_mesh_ids = object_stacks[key]['mesh_ids_of_objects_under_it']
        parents = object_stacks[key]['objects_under_it']
        node = SGNode(obj_name, mesh_id, parents_mesh_ids, parents)   
        node_dict[key] = node

    # Detect and break cycles in the obtained graph
    print('Node dict: {}'.format(node_dict.keys()))
    # dict_keyz = copy.deepcopy(node_dict.keys())
    for i, name in enumerate(node_dict.keys()):
        # Run bfs based cycle-detection
        
        visited = np.zeros(len(node_dict.keys())+1)
        # print(visited)
        head = node_dict[name]
        # print("Node type: {}, mesh id type: {}".format(type(head), int(head.mesh_id)))
        to_visit_list = []
        # print(i)
        while head != None:
            # print(i)
            # print(i, int(head.mesh_id))
            visited[int(head.mesh_id)] = 1
            cycle_parents = []
            for parent in head.parent_mesh_ids:
                if visited[int(parent)] == 1:
                    # Found a cycle
                    # Break the cycle by removing the parent from the parent list
                    cycle_parents.append(parent)
                    continue
                to_visit_list.append(parent)
            for parent in cycle_parents:
                head.parent_mesh_ids.remove(parent)
                print(parent, node_dict.keys())
                head.parents.remove(node_dict[str(parent)].object)
                if parent in to_visit_list:
                    to_visit_list.remove(parent)

            if len(to_visit_list)!=0:
                # print(type(to_visit_list[0]))
                # node_dict[name] = copy.deepcopy(head)
                head = node_dict[str(to_visit_list[0])]
                to_visit_list.pop(0) 
            else:
                break

    # Print the obtained tree after removing all the cycles in the given graph
    print("\nPrinting the modified trees:\n")
    for i, key in enumerate(node_dict.keys()):
        print("Object_name: {}\tMesh_id: {}\tParent_list: {}".format(node_dict[key].object, node_dict[key].mesh_id, node_dict[key].parents))      

    return node_dict  

def generate_scene_graph_from_object_dict(object_dict, mesh_dir='/root/ocrtoc_ws/src/ocrtoc_materials/models', save_path='/root/ocrtoc_ws/src/stack_detection/scene_dict.npz'):
    '''Generates stack info for each object

    Parameters:
    object_dict: {
        object_name: {
            object_label: <object_label>,
            pose: object_6D_pose ([x, y, z, roll, pitch, yaw])
        }
    }

    Return:
    None
    
    Save:
        - object_stacks: {
            'current_object_info': {
                'object': <object_label>,
                'mesh_id': <object_mesh_id>
            },
            'mesh_ids_of_objects_under_it': [geometry_ids],
            'objects_under_it': stack_obj_names
        }
        - default_path: /root/ocrtoc_ws/src/stack_detection/scene_dict.npz
    '''
    scene = o3d.t.geometry.RaycastingScene()

    object_info_dict = {}
    mesh_id_dict = {}

    meshes = []
    pcds =[]

    colors = [[0, 0, 0], [0.30, 0, 0], [0, 0.3, 0], [0, 0, 0.3], [0.3, 0.3, 0], [0.3, 0, 0.3], [0.3, 0.3, 0.3]]
    counter = -1


    print(object_dict)
    for i, key in enumerate(object_dict.keys()):
        object_label = object_dict[key]['object_label']
        object_pose = object_dict[key]['pose']
        counter += 1

        mesh = o3d.io.read_triangle_mesh('{}{}/textured.obj'.format(mesh_dir, object_label))
        mesh2 = o3d.io.read_triangle_mesh('{}{}/textured.obj'.format(mesh_dir, object_label))

        R = mesh.get_rotation_matrix_from_xyz((object_pose[3], object_pose[4], object_pose[5]))
        center = np.array(mesh.get_center())
        mesh.rotate(R, center=(center[0], center[1], center[2]))

        required_pos = np.array([object_pose[0], object_pose[1], object_pose[2]])
        dt = required_pos # required_pos - center

        mesh.translate(dt)

        pcd = mesh.sample_points_poisson_disk(number_of_points=50, init_factor=5, pcl=None)

        # Color the objects for visual debugging
        # pcd.paint_uniform_color(colors[counter])
        # mesh.paint_uniform_color(colors[counter])

        meshes.append(mesh)
        pcds.append(pcd)

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        mesh_id = scene.add_triangles(mesh)
        print("Object name: {}\tpose: {}\tMesh id: {}".format(key, object_pose, mesh_id))

        info_dict = {
            'object': key,
            'mesh_id': mesh_id,
            'pcd': np.asarray(pcd.points)
        }
        object_info_dict[str(mesh_id)] = info_dict
        mesh_id_dict[str(mesh_id)] = {
            'object': key,
            'pose': object_pose,
        }

    # o3d.visualization.draw_geometries(meshes)
    # o3d.visualization.draw_geometries(pcds)

    ray_direction = [0, 0, -1]

    # Now get the object/objects under each object
    object_stacks = {}
    ray_list = []
    for i, key in enumerate(object_info_dict.keys()):
        # if i<=2: 
        #     continue
        # print("Object: {}\tmesh id: {}".format(key, object_info_dict[key]['object']))
        pcd = object_info_dict[key]['pcd']
        ray_list = []
        n_pts = len(pcd)
        for j in range(n_pts):
            x, y, z = pcd[j, :]
            ray_tensor = [x, y, z, ray_direction[0], ray_direction[1], ray_direction[2]]
            ray_list.append(ray_tensor)
            # x, y, z = position[0], position[1], position[2]
            # print("Position: {}".format([x, y, z]))
            # break
        # break
        rays = o3d.core.Tensor(ray_list,
                       dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        # print(ans)
        geometry_ids_init = list(set(ans['geometry_ids'].numpy())) # Contain mesh ids of the objects that the rays hit
        # print(type(geometry_ids_init))
        # print(geometry_ids_init)
        geometry_ids = []
        for g_id in geometry_ids_init:
            if g_id <= len(object_info_dict) and g_id >= 0:
                if int(g_id) == int(key):
                    continue
                geometry_ids.append(g_id)
        
        # exit()
        stack_obj_names = []
        for g_id in geometry_ids:
            stack_obj_names.append(mesh_id_dict[str(g_id)]['object'])
        object_stacks[key] = {
            'current_object_info': {
                'object': object_info_dict[key]['object'],
                'mesh_id': object_info_dict[key]['mesh_id']
            },
            'mesh_ids_of_objects_under_it': geometry_ids,
            'objects_under_it': stack_obj_names
        }
    
    # Generate node dict
    node_dict = get_a_graph_from_obj_dict(object_stacks=object_stacks)

    final_object_stacks = {}
    for i, key in enumerate(object_stacks.keys()):
        final_object_stacks[object_info_dict[key]['object']] = {
            'current_object_info': object_stacks[key]['current_object_info'],
            'mesh_ids_of_objects_under_it': node_dict[key].parent_mesh_ids,
            'objects_under_it': node_dict[key].parents
        }
        
        # print("Object name: {}\tObject mesh id: {}\nObject stack list: {}\n".format(object_info_dict[key]['object'], object_info_dict[key]['mesh_id'], geometry_ids))
    
    # Save contents to the given file location
    # np.savez_compressed(save_path, data=final_object_stacks)
    with open(save_path, 'wb') as fp:
        pickle.dump(final_object_stacks, fp, protocol=2)


def generate_scene_graph_from_object_dict2(object_dict, mesh_dir='/root/ocrtoc_ws/src/ocrtoc_materials/models', save_path='/root/ocrtoc_ws/src/stack_detection/scene_dict.npz'):
    '''Generates stack info for each object

    Parameters:
    object_dict: {
        object_name: {
            object_label: <object_label>,
            pose: object_6D_pose ([x, y, z, roll, pitch, yaw])
        }
    }

    Return:
    None
    
    Save:
        - object_stacks: {
            'current_object_info': {
                'object': <object_label>,
                'mesh_id': <object_mesh_id>
            },
            'mesh_ids_of_objects_under_it': [geometry_ids],
            'objects_under_it': stack_obj_names
        }
        - default_path: /root/ocrtoc_ws/src/stack_detection/scene_dict.npz
    '''
    scene = o3d.t.geometry.RaycastingScene()

    object_info_dict = {}
    mesh_id_dict = {}

    meshes = []
    pcds =[]

    colors = [[0, 0, 0], [0.30, 0, 0], [0, 0.3, 0], [0, 0, 0.3], [0.3, 0.3, 0], [0.3, 0, 0.3], [0.3, 0.3, 0.3]]
    counter = -1

    for i, key in enumerate(object_dict.keys()):
        for object_pose in object_dict[key]:
            counter += 1
            mesh = o3d.io.read_triangle_mesh('{}{}/textured.obj'.format(mesh_dir, str(key)))
            mesh2 = o3d.io.read_triangle_mesh('{}{}/textured.obj'.format(mesh_dir, str(key)))
            # pcd = mesh.sample_points_poisson_disk(number_of_points=2000, init_factor=5, pcl=None)
            # object_pose = object_dict[key][0]
            R = mesh.get_rotation_matrix_from_xyz((object_pose[3], object_pose[4], object_pose[5]))
            # print("{} initial mesh center: {}".format(key, mesh.get_center()))
            center = np.array(mesh.get_center())
            # Rotate in place about its own center
            mesh.rotate(R, center=(center[0], center[1], center[2]))
            # o3d.visualization.draw_geometries([mesh, mesh2])
            # Translate the mesh to the desired position
            required_pos = np.array([object_pose[0], object_pose[1], object_pose[2]])
            dt = required_pos - center
            # mesh.translate((dt[0], dt[1], dt[2]))
            mesh.translate(required_pos)

            pcd = mesh.sample_points_poisson_disk(number_of_points=50, init_factor=5, pcl=None)

            pcd.paint_uniform_color(colors[counter])
            mesh.paint_uniform_color(colors[counter])

            meshes.append(mesh)
            pcds.append(pcd)

            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            mesh_id = scene.add_triangles(mesh)
            print("Object name: {}\tpose: {}\tMesh id: {}".format(key, object_pose, mesh_id))

            info_dict = {
                'object': key,
                'mesh_id': mesh_id,
                'pcd': np.asarray(pcd.points)
            }
            object_info_dict[str(mesh_id)] = info_dict
            mesh_id_dict[str(mesh_id)] = {
                'object': key,
                'pose': object_pose,
            }

    o3d.visualization.draw_geometries(meshes)
    # o3d.visualization.draw_geometries(pcds)

    ray_direction = [0, 0, -1]

    # Now get the object/objects under each object
    object_stacks = {}
    ray_list = []
    for i, key in enumerate(object_info_dict.keys()):
        # if i<=2: 
        #     continue
        # print("Object: {}\tmesh id: {}".format(key, object_info_dict[key]['object']))
        pcd = object_info_dict[key]['pcd']
        ray_list = []
        n_pts = len(pcd)
        for j in range(n_pts):
            x, y, z = pcd[j, :]
            ray_tensor = [x, y, z, ray_direction[0], ray_direction[1], ray_direction[2]]
            ray_list.append(ray_tensor)
            # x, y, z = position[0], position[1], position[2]
            # print("Position: {}".format([x, y, z]))
            # break
        # break
        rays = o3d.core.Tensor(ray_list,
                       dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        # print(ans)
        geometry_ids_init = list(set(ans['geometry_ids'].numpy())) # Contain mesh ids of the objects that the rays hit
        # print(type(geometry_ids_init))
        # print(geometry_ids_init)
        geometry_ids = []
        for g_id in geometry_ids_init:
            if g_id <= len(object_info_dict) and g_id >= 0:
                if int(g_id) == int(key):
                    continue
                geometry_ids.append(g_id)
        
        # exit()
        stack_obj_names = []
        for g_id in geometry_ids:
            stack_obj_names.append(mesh_id_dict[str(g_id)]['object'])
        object_stacks[key] = {
            'current_object_info': {
                'object': object_info_dict[key]['object'],
                'mesh_id': object_info_dict[key]['mesh_id']
            },
            'mesh_ids_of_objects_under_it': geometry_ids,
            'objects_under_it': stack_obj_names
        }
        # print("Object name: {}\tObject mesh id: {}\nObject stack list: {}\n".format(object_info_dict[key]['object'], object_info_dict[key]['mesh_id'], geometry_ids))
    
    # Save contents to the given file location
    # with open(save_path, 'wb') as fp: # https://www.geeksforgeeks.org/save-a-dictionary-to-a-file/
    #     pickle.dump(object_stacks, fp)
    np.savez_compressed(save_path, data=object_stacks)
        # json_str = json.dumps(object_stacks)
        # json.dump(json_str, fp)
        # json.dumps(object_stacks, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default='1-1-1')
    parser.add_argument('--object_dict_path', default='/root/ocrtoc_ws/src/stack_detection/object_dict.npz')
    parser.add_argument('--mesh_dir', default='/root/ocrtoc_ws/src/ocrtoc_materials/models/')
    parser.add_argument('--save_path', default='/root/ocrtoc_ws/src/stack_detection/scene_dict.npz')
    FLAGS = parser.parse_args()

    # with open(FLAGS.object_dict_path, 'rb') as f:
    #     object_dict = pickle.load(f)

    # object_dict = np.load(FLAGS.object_dict_path, allow_pickle=True)['data'].item()
    with open(FLAGS.object_dict_path, 'rb') as f:
        object_dict = pickle.load(f)
    print("Object_dict: {}\n\nType:{}".format(object_dict, type(object_dict)))

    generate_scene_graph_from_object_dict(object_dict, mesh_dir=FLAGS.mesh_dir, save_path=FLAGS.save_path)
    # import time
    # st = time.time()
    # scene_dir = '/root/ocrtoc_ws/src/ocrtoc_materials/targets' # './final_scenes'
    # mesh_dir = '/root/ocrtoc_ws/src/ocrtoc_materials/models/'
    # scenes = ['1-1-1', '2-2-2', '5-3-1', '6-1-2', '6-1-1', '4-1-2', '4-1-1', '2-1-1', '1-5-2', '1-2-2', '3-2-1', '4-2-2']
    # yaml_paths = []
    # for scene in scenes:
    #     yaml_paths.append('{}/{}.yaml'.format(scene_dir, scene))
    # current_scene_index = 5 # len(scenes) - 1

    # object_dict = get_object_list_from_yaml(yaml_paths[current_scene_index])

    # generate_scene_graph_from_object_dict2(object_dict, mesh_dir=mesh_dir)
    # end = time.time()
    # print("Time taken: {}".format(end-st))

