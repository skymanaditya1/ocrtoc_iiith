#!/usr/bin/env python3

'''ORTOOLS based Task planner
Uses OR-Tools to find a an optimal task plan 
Team RRC
Author: Vishal Reddy Mandadi
'''

from __future__ import print_function
import argparse
import copy
import math
from platform import node
from unicodedata import name
import numpy as np
import transforms3d
import os
import yaml

from std_msgs.msg import String, Bool, Empty, Int64
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from motion_planning import MotionPlanner
from ocrtoc_common.transform_interface import TransformInterface
from ocrtoc_msg.srv import PerceptionTarget, PerceptionTargetRequest
import rospkg
import rospy

from sensor_msgs.msg import JointState
import time

# from primitives import GRASP, collision_test, distance_fn, DiscreteTAMPState, get_shift_one_problem, get_shift_all_problem
# from viewer import DiscreteTAMPViewer, COLORS
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Imports for camera related stuff
from numpy.core.numeric import full
import cv2
import open3d as o3d
# import open3d_plus as o3dp
# import open3d_plus as o3dp
# from .arm_controller_for_task_planner import ArmController
from ocrtoc_common.camera_interface import CameraInterface
# from ocrtoc_common.transform_interface import TransformInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

# ORTOOLS imports
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Imports for buffer sampling
from scipy import signal
from scipy.spatial.transform import Rotation

# for grasp refinement
from PIL import Image
from time import sleep


# Grasp refinement and collision checking class
class GraspModification:
    def __init__(self, search_space_diameter = 20):
        '''Grasp modification class
        Modifies or finds a better grasp at a given location
        search_space_diameter: the max width and length of the heigtmap of the scene, to be searched in for
        '''
        self.ee_length = 14
        self.ee_width = 4
        self.finger_max_dist = 8
        self.finger_len = 1 # Length of the face of finger facing the ground
        self.finger_bre = 1 # Breadth of the face of the finger facing the ground
        self.finger_height = 8
        self.max_penetration_len = 4 # Max amount of penetration of the object between the two fingers
    def get_gripper_standard_collision_map_1cm_resolution(self):
        '''Gets the gripper collision map that can be used to evaluate collisions at different points. 
        The produced map is for the gripper in its canonical/standard pose (it's length parellel to 
        the length of the table)
        The resolution of the map -> 1 pixel occupies 1cm*1cm square box in the real world
        '''
        collision_map = np.full(shape=(self.ee_length, self.ee_length), fill_value=10) 
        for i in range(int(self.ee_length//2 - self.ee_width//2), int(self.ee_length//2 + self.ee_width//2)):
            collision_map[i:i+1] = 0
            # print("ee - {}".format(i))
            if i == (self.ee_length//2 - 1) or i == self.ee_length//2:
                collision_map[i, 2] = -1*self.finger_height # self.max_penetration_len # or use self.finger_height, if full penetration is allowed
                collision_map[i, 11] = -1*self.finger_height # self.max_penetration_len
                collision_map[i, 3:11] = -1*self.max_penetration_len # (not allowing any object to penetrate more than 4 cm towards the EE, between the fingers)
                # print("finger - {}".format(i))
            
        print("Final collision map:\n{}".format(collision_map))
        return collision_map

    def get_gripper_standard_collision_map_5mm_resolution(self):
        '''Gets the gripper collision map that can be used to evaluate collisions at different points. 
        The produced map is for the gripper in its canonical/standard pose (it's length parellel to 
        the length of the table)
        The resolution of the map -> 1 pixel occupies 0.5cm*0.5cm square box in the real world
        '''
        collision_map = np.full(shape=(self.ee_length*2, self.ee_length*2), fill_value=10) 
        for i in range(int(2*self.ee_length//2 - 2*self.ee_width//2), int(2*self.ee_length//2 + 2*self.ee_width//2)):
            collision_map[i:i+1] = 0
            print("ee - {}".format(i))
            # print(range((2*self.ee_length//2 - 2),  2*self.ee_length//2+2))
            if i in range((2*self.ee_length//2 - 2),  2*self.ee_length//2+2):
                collision_map[i, 4:6] = -1*self.finger_height# self.max_penetration_len # or use self.finger_height, if full penetration is allowed
                collision_map[i, 22:24] = -1*self.finger_height # self.max_penetration_len
                print("finger - {}".format(i))
            
        print("Final collision map:\n{}".format(collision_map))
        return collision_map

    def get_grip_validation_map_1cm_res(self):
        '''Generates grip validation map with 1cm resolution (1 pixel == 1cm*1cm area)
        The produced map is for the gripper in its canonical/standard pose (it's length parellel to 
        the length of the table)
        Used for grasp validation 
        Size = 14*14 (general)
        '''
        v_map = np.full(shape=(self.ee_length, self.ee_length), fill_value=10) 
        for i in range(int(self.ee_length//2 - self.ee_width//2), int(self.ee_length//2 + self.ee_width//2)):
            # print("ee - {}".format(i))
            if i == (self.ee_length//2 - 1) or i == self.ee_length//2:
                v_map[i:i+1, 3:11] = 0
            
        print("Final validation map:\n{}".format(v_map))
        return v_map


    def rotate_kernel_map(self, kernel_map, angle):
        '''Converts the given map into an image and rotates it by the given angle
        Parameters:
        kernel_map = np.ndarray (2D array)
        angle: in degrees
        '''
        # modified_k_map = kernel_map + 10
        # print("Modified map: {}".format(modified_k_map))
        print("Kernel map: {}".format(abs(kernel_map+self.finger_height)))
        k_img = Image.fromarray(np.uint8(kernel_map+self.finger_height)) # Converts all negative elements to +ve (will be reversed later)
        r_img = k_img.rotate(angle=angle, fillcolor=10+self.finger_height) # Angle in degrees
        rk_map = np.array(r_img, dtype=float) 
        print("Rotated kernel: {}".format(rk_map-self.finger_height))
        
        cv2.imshow("Kernel image", np.uint8((kernel_map)*15))
        cv2.waitKey(0)

        cv2.imshow("Rotated image", np.uint8((rk_map-self.finger_height)*15))
        cv2.waitKey(0)
        # k_img.show()
        # sleep(10)
        # r_img.show()
        # sleep(30)
        return rk_map-self.finger_height

    def rotate_v_map(self, kernel_map, angle):
        '''Converts the given map into an image and rotates it by the given angle
        Parameters:
        kernel_map = np.ndarray (2D array)
        angle: in degrees
        '''
        # modified_k_map = kernel_map + 10
        # print("Modified map: {}".format(modified_k_map))
        print("Valid map: {}".format(abs(kernel_map+self.finger_height)))
        k_img = Image.fromarray(np.uint8(kernel_map+self.finger_height)) # Converts all negative elements to +ve (will be reversed later)
        r_img = k_img.rotate(angle=angle, fillcolor=10+self.finger_height) # Angle in degrees
        rk_map = np.array(r_img, dtype=float) 
        print("Rotated Valid: {}".format(rk_map-self.finger_height))
        
        cv2.imshow("Valid image", np.uint8((kernel_map)*15))
        cv2.waitKey(0)

        cv2.imshow("Rotated Valid image", np.uint8((rk_map-self.finger_height)*15))
        cv2.waitKey(0)
        return rk_map-self.finger_height

    def generate_random_h_map_1cm_res(self):
        '''Generates a height map (random) of size (20*20) with resolution of 1cm
        '''
        h_map = np.zeros(shape=(20, 20), dtype=float)

        h_map[3:10] = 6

        print("Height map (random): {}".format(h_map))
        return h_map

    def return_collision_and_valid_maps(self, scene_hmap, gripper_map, v_map, grasp_height):
        '''Checks for collisions between scene and gripper and returns a map, with 1s placed at 
        collision free areas, and 0s placed in all the other places
        scene_hmap: 20*20
        gripper_map: 14*14 map
        '''
        n_r, n_c = scene_hmap.shape
        g_nr, g_nc = gripper_map.shape
        collision_map = np.zeros(shape=(n_r, n_c))
        valid_map = np.zeros(shape=(n_r, n_c))

        for i in range(0, n_r - g_nr):
            for j in range(0, n_c - g_nc):
                sub_map = scene_hmap[i:i+g_nr, j:j+g_nc]
                print("Sub map shape: {}".format(sub_map.shape))
                # Collision checking
                if np.any(gripper_map + grasp_height - sub_map<0): # Collision detected
                    collision_map[i+int(g_nr/2), j+int(g_nc/2)] = 0
                else: # No collision
                    collision_map[i+int(g_nr/2), j+int(g_nc/2)] = 1
                    if np.min(v_map+grasp_height-self.finger_height - sub_map) < -1:# Valid grasp pose
                        valid_map[i+int(g_nr/2), j+int(g_nc/2)] = np.max(-1*(v_map+grasp_height-self.finger_height - sub_map))# 1
    
        return collision_map, valid_map

    def pcd_to_height_map_1cm_res(self, pcd, target_pos, x_range=0.30, y_range=0.30):
        '''Cuts a 30cm*30cm boundary around target pos in the given pcd and converts it into a height map
        This gives an effective search space of size - 17cm*17cm (Assuming the kernel map to be of size 
        14cm * 14cm)
        Height map resolution: 1cm (1 pixel == 1cm*1cm)
        Parameters
        pcd: pcd.points (np.array) (point cloud of the scene)
        target_pos: [x, y, z]: list (target grasp pose)
        Returns:
        (height_map, (height_map_origin_coords_world_frame)) - 
        '''
        # For resolution of 1cm*1cm, multiply pcd by 100
        pcd_m = (np.round(pcd * 100))
        pcd_m[:, 2] = pcd[:, 2] # Only x and y coordinates are scaled up to represent pixels. Z values retain their meaning (value in cm)
        x_minmax = [target_pos[0] - (x_range/2), target_pos[0]+(x_range/2 - 0.01)]*100
        y_minmax = [target_pos[1] - (y_range/2), target_pos[1]+(y_range/2 - 0.01)]*100
        # x_range = 0.6
        # y_range = 1.2

        pixel_maxes = np.zeros(shape=(int(x_range*100), int(y_range*100)), dtype=float)

        for point in pcd:
            # print(point)
            if point[0] >= x_minmax[0] and point[0] <= x_minmax[1] and point[1] >= y_minmax[0] and point[1] <= y_minmax[1]:
                # Valid point
                # print("Yes")
                pixel_coord = [int(100*(point[0] - x_minmax[0])), int(100*(point[1] - y_minmax[0]))]
                print(point, pixel_coord)
                if pixel_maxes[pixel_coord[0], pixel_coord[1]] < point[2]:
                    # print("Yes")
                    pixel_maxes[pixel_coord[0], pixel_coord[1]] = point[2]
        
        cv2.imshow("Scene hmap", np.uint8(pixel_maxes*255/np.max(pixel_maxes)))
        cv2.waitKey(0)

        return pixel_maxes


# Stacking related functions
class SGNode:
    def __init__(self, object_name, mesh_id, parents):
        '''
        object_name: Current object name
        parents: Parents' node ids
        '''
        self.object = object_name
        self.mesh_id = mesh_id
        self.parents = parents

def get_a_graph_from_obj_dict(object_stacks):
    '''
    Parameters:
    object_stacks: {
        'mesh_id': {
            'current_object_info': {
                'object': <object_label>,
                'mesh_id': <object_mesh_id>
            },
            'mesh_ids_of_objects_under_it': [i, j, ...]
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
        parents = object_stacks[key]['mesh_ids_of_objects_under_it']
        node = SGNode(obj_name, mesh_id, parents)   
        node_dict[key] = node

    # Detect and break cycles in the obtained graph
    for i, key in enumerate(node_dict.keys()):
        # Run bfs based cycle-detection
        visited = np.zeros(len(node_dict.keys()))
        # print(visited)
        head = node_dict[key]
        # print("Node type: {}, mesh id type: {}".format(type(head), int(head.mesh_id)))
        to_visit_list = []
        # print(i)
        while head != None:
            # print(i)
            # print(i, int(head.mesh_id))
            visited[int(head.mesh_id)] = 1
            cycle_parents = []
            for parent in head.parents:
                if visited[int(parent)] == 1:
                    # Found a cycle
                    # Break the cycle by removing the parent from the parent list
                    cycle_parents.append(parent)
                to_visit_list.append(parent)
            for parent in cycle_parents:
                head.parents.remove(parent)
                if parent in to_visit_list:
                    to_visit_list.remove(parent)

            if len(to_visit_list)!=0:
                # print(type(to_visit_list[0]))
                head = node_dict[str(to_visit_list[0])]
                to_visit_list.pop(0) 
            else:
                break

    # Print the obtained tree after removing all the cycles in the given graph
    print("\nPrinting the modified trees:\n")
    for i, key in enumerate(node_dict.keys()):
        print("Object_name: {}\tMesh_id: {}\tParent_list: {}".format(node_dict[key].object, node_dict[key].mesh_id, node_dict[key].parents))      

    return node_dict  


# Miscellineous functions class - contains functions from https://github.com/GouMinghao/open3d_plus/blob/main/open3d_plus/geometry.py 
# As open3d_plus import failed (due to unknown reasons - TO BE FIXED)
class MiscFunctions:
    '''
    Open3d_plus library functions taken from - https://github.com/GouMinghao/open3d_plus/blob/main/open3d_plus/geometry.py
    Due to open3d_plus import failure
    '''
    def __init__(self):
        pass
    
    def array2pcd(self, points, colors):
        """
        Convert points and colors into open3d point cloud.
        Args:
            points(np.array): coordinates of the points.
            colors(np.array): RGB values of the points.
        Returns:
            open3d.geometry.PointCloud: the point cloud.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def pcd2array(self, pcd):
        """
        Convert open3d point cloud into points and colors.
        Args:
            pcd(open3d.geometry.PointCloud): the point cloud.
        Returns:
            np.array, np.array: coordinates of the points, RGB values of the points.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        return points, colors

class Object:
    def __init__(self, mesh_path):
        '''
        '''
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.copy_mesh = None
        self.kernel_map = np.array([])
    
    def render_to_pose(self, object_pose):
        '''
        Parameters:
        object_pose: [6 element list] (x, y, z, roll, pitch, yaw)
        '''
        self.copy_mesh = copy.deepcopy(self.mesh)
        R = self.copy_mesh.get_rotation_matrix_from_xyz((object_pose[3], object_pose[4], object_pose[5]))
        center = np.array(self.copy_mesh.get_center())
        self.copy_mesh.rotate(R, center=True)
        required_pos = np.array([object_pose[0], object_pose[1], object_pose[2]])
        dt = required_pos - center
        self.copy_mesh.translate((dt[0], dt[1], dt[2]))


    def get_pcd_from_copy_mesh(self, n_pts = 1000):
        '''
        Parameters:
        n_pts: Number of points to be sampled
        '''
        pcd = self.copy_mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5, pcl=None)
        return pcd

    def render_to_pose_and_get_pcd(self, object_pose, n_pts = 1000):
        '''
        Parameters:
        object_pose: [6 element list] (x, y, z, roll, pitch, yaw)
        n_pts: Number of points to be sampled
        '''
        self.render_to_pose(object_pose=object_pose)
        return self.get_pcd_from_copy_mesh(n_pts=n_pts)

    def get_occupancy_map_kernel(self, target_6D_pose, occupancy_buffer, debug=False):
        '''Create an occupancy map of the object centered at [0, 0]. Use this as a kernel for collision checking
        Size differs from object to object (it is like a tightly fitting bounding box)

        Parameters:
        target_6D_pose = list of 6 coordinates [x, y, z, roll, pitch, yaw] 

        Return:
        kernel_map = 2D numpy array (occupancy maps with 0s and 1s)
        occupancy_and_buffer: OccupancyAndBuffer class object
        '''
        # 1. Create a canonical pose with orienation and z value similar to target 6d pose
        canonical_pose = [ 0.0, 0.0, target_6D_pose[2], target_6D_pose[3], target_6D_pose[4], target_6D_pose[5]]
        entity_pcd = self.render_to_pose_and_get_pcd(object_pose=canonical_pose)
        entity_omap = occupancy_buffer.generate_2D_occupancy_map(world_dat = np.asarray(entity_pcd.points)*100, threshold=1, dir_path='./results/object_{}-{}.png'.format('2-2-2', 'green_bowl'), save=True)
        if debug==True:
            print("Shape: {}".format(entity_omap.shape))
            print(entity_omap)
            cv2.imshow("occupancy map", (entity_omap * 255).astype(np.uint8))
            cv2.waitKey(0)
        
        self.kernel_map = entity_omap

    
class OccupancyAndBuffer:
    def __init__(self):
        pass
    
    def generate_2D_occupancy_map(self, world_dat, x_min=None, y_min=None, x_range=None, y_range=None, threshold=3, dir_path='/root/occ_map.png', save=False):
        '''
        A non-traditional way to mark occupied areas on a grid. In this method, we simply look for 
        areas with z>threshold (threshold~0) and mark them as occupied. This is expected to be highly
        effective for OCRTOC
        '''
        if x_range == None:
            x_range = int(np.round(np.max(world_dat[:, 0]))-np.round(np.min(world_dat[:, 0])))
            print("x_range: {}".format(x_range))
        if y_range == None:
            y_range = int(np.round(np.max(world_dat[:, 1]))-np.round(np.min(world_dat[:, 1])))
            print("y_range: {}".format(y_range))
        # Since, pixels have integral coordinates, let us round off all the values in world_dat and remove y coordinates
        world_dat_rounded = (np.round(world_dat)).astype(int)
        # world_dat_rounded = (np.delete(world_dat_rounded, 2, 1)).astype(int) # Remove y coordinates column
        if x_min == None:
            x_min = (np.round(np.min(world_dat[:, 0]))).astype(int)
            print("x_min: {}".format(x_min))
        if y_min == None:
            y_min = (np.round(np.min(world_dat[:, 1]))).astype(int)
            print("y_min: {}".format(y_min))

        pixel_counts = np.zeros(shape=(x_range+1, y_range+1), dtype=float)
        count = 0
        for point in world_dat_rounded:
            #print(point-np.array([x_min, z_min], dtype=int))
            x_coord = point[0]-x_min
            y_coord = point[1] - y_min
            if x_coord > x_range or y_coord > y_range or x_coord < 0 or y_coord < 0:
                continue
            if point[2] > 0: 
                pixel_counts[point[0]-x_min, point[1]-y_min]+=1

        mean_points = 1# np.mean(pixel_counts)
        occ_map = np.zeros(shape=pixel_counts.shape, dtype=np.uint8)
        for i in range(x_range):
            for j in range(y_range):
                if pixel_counts[i, j] >= threshold:
                    occ_map[i, j] = 1 # Occupied

        import cv2
        cv2.imwrite(dir_path,(occ_map * 255).astype(np.uint8))
        # cv2.imshow('Occupancy map', (occ_map * 255).astype(np.uint8))
        # cv2.waitKey(0)

        # print(pixel_counts)

        return occ_map
    
    def visualize_pcd_with_global_coordinate_frame(self, pcd):
        mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-0.3, -.6, 0])
        mesh_frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[.3, -.6, 0])
        mesh_frame4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[.3, .6, 0])
        mesh_frame5 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-.3, .6, 0])
        o3d.visualization.draw_geometries([pcd, mesh_frame1, mesh_frame2, mesh_frame3, mesh_frame4, mesh_frame5])
        
    def get_closest_to(self, empty_spots, target_pos = np.array([.0, .0]), collision_diameter=0.1):
        distances = np.linalg.norm(empty_spots - target_pos, axis=1) 
        valid_empties = []
        valid_distances = []
        for i in range(len(empty_spots)):
            if distances[i] > collision_diameter:
                valid_empties.append(empty_spots[i])
                valid_distances.append(distances[i])
        print("Valid distances: {}".format(valid_distances))
        closest_ind = np.argmin(np.array(valid_distances))
        print("Closest_ind: {}".format(closest_ind))
        print("Closest vector: {}".format(valid_empties[closest_ind]))
        # print(distances)
        # print("Valid empties: {}".format(valid_empties))
        return valid_empties[closest_ind]

    def get_occupancy_percentage(self, target_pose_6D, scene_omap, entity):
        '''Gets the percentage of target space that is occupied
        '''
        entity_pcd = entity.render_to_pose_and_get_pcd(object_pose=target_pose_6D)
        entity_omap = self.generate_2D_occupancy_map(world_dat = np.asarray(entity_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120,
                                                     threshold=1, dir_path='./results/object_{}-{}.png'.format('2-2-2', 'green_bowl'), save=False) # xy_min_max=[-30, 30, -60, 60], 
        # cv2.imshow('Occupancy map', (scene_omap * 255).astype(np.uint8))
        # cv2.waitKey(0)
        if len(entity_omap) == 0:
            return 100
        occupancy = np.logical_and(scene_omap, entity_omap)
        # print("Occupancy shape: {}".format(occupancy.shape))
        total_occupied_space_of_object = np.sum(entity_omap)
        occ_percent = float(np.sum(occupancy)*100.0)/(total_occupied_space_of_object)

        return occ_percent

    def sample_8_pts_around_given_point(self, given_pt, step=0.1):
        '''Samples 9 points symmetrically in a square fashion around the given point in sample plane (z unchanged)
        xxx
        xox
        xxx
        Here, 'x' denote the sampled point, while 'o' denotes the given point
        Parameters:
        given_pt: list (1, 3): given point's 3D position
        step: size of the step
        Return:
        pts = list (9, 3) # Includes the current point
        '''
        pts = []
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                # print(given_pt)
                x = given_pt[0]+(i*step)
                y = given_pt[1]+(j*step)
                z = given_pt[2]
                # print(x, y, z)
                new_pt = [x, y, z]
                # print(new_pt)
                pts.append(new_pt)
        return pts

    def marching_grid(self, scene_pcd, object_mesh_path, target_pose_6D, OCC_THRESH=0.0, scene_name='-', object_name='-'):
        '''Marching Grid algorithm
        Returns the closest empty buffer space to the given target using marching grid algorithm
        Parameters:
        scene_pcd: Point cloud of the 3D scene
        object_mesh_path: Path to the mesh file of the object
        target_position_3D: list [x, y, z, roll, pitch, yaw] - Target 6D pose of object in world frame
        OCC_THRESH: The maximum % for which a pose is considered to be free
        Return:
        buffer_spot: np.ndarray [x, y, z, roll, pitch, yaw] in world frame (same frame as the given world_data and 
                        target_position 3D)
        '''
        scene_omap = self.generate_2D_occupancy_map(np.asarray(scene_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120,
                                                    threshold=1, dir_path='./results/{}-{}.png'.format(scene_name, object_name), save=False) # xy_min_max=[-29.97, 29.97, -59.96, 59.99]
        entity = Object(mesh_path=object_mesh_path)

        occ_percent = self.get_occupancy_percentage(target_pose_6D=target_pose_6D, scene_omap=scene_omap, entity=entity)
        print("Occupancy %: {}%".format(occ_percent))
        if occ_percent<=OCC_THRESH:
            print("Found!")
            return target_pose_6D


        done = False
        search_step = 0.1
        current_position = [target_pose_6D[0], target_pose_6D[1], target_pose_6D[2]]
        prev_min_val = 100
        n_steps = 3
        while not done and n_steps>=0:
            n_steps = n_steps - 1
            sampled_pts = self.sample_8_pts_around_given_point(given_pt=current_position, step=search_step)
            occ_percents = []
            for i, pt in enumerate(sampled_pts):
                pose_6d = [pt[0], pt[1], pt[2], target_pose_6D[3], target_pose_6D[4], target_pose_6D[5]]
                occ_percent = self.get_occupancy_percentage(target_pose_6D=pose_6d, scene_omap=scene_omap, entity=entity)
                occ_percents.append(occ_percent)
            print("Occupancy percents: {}".format(occ_percents))
            
            min_val = min(occ_percents)
            min_index = occ_percents.index(min_val)
            print("min_index: {}\tmin_val: {}".format(min_index, min_val))
            if min_val <= OCC_THRESH:
                print("Min pose found - min occupancy: {}".format(min_val))
                done = True
                current_position = sampled_pts[min_index]
            elif min_val < prev_min_val:
                current_position = sampled_pts[min_index]
            elif min_val==prev_min_val:
                current_position = sampled_pts[min_index]
                search_step = search_step/2
            elif min_val > prev_min_val:
                search_step = search_step/2

        buffer_spot = [current_position[0], current_position[1], current_position[2], 
                    target_pose_6D[3], target_pose_6D[4], target_pose_6D[5]]
        print("Buffer spot: {}".format(buffer_spot))
        print("Actual spot: {}".format(target_pose_6D))

        return buffer_spot

    def convolutional_buffer_sampler(self, scene_omap, entity_omap, target_6D_pose, object_mesh_path='', scene_name='-', object_name='-', debug=False):
        '''Convolutional Buffer Sampler
        Samples buffer spot by fast-occupancy-collision checking using the convolution operation
        Parameters:
        scene_omap: Occupancy map of the scene (usually of size 60*120)
        entity_omap: Occupancy map of the target object (usually of varying sizes (of order 10*20 or so))
        target_6D_pose: The target goal pose of the object (closest to which should be our sampled spot)
        '''
        # Convolve scene map with kernel map by adding 'same' padding (with padding value=1 (not zero, to prevent boundary placements))
        convolved_omap = signal.convolve2d(scene_omap, entity_omap, mode='same', boundary='fill', fillvalue=1)

        max_val = np.amax(convolved_omap)
        min_val = np.amin(convolved_omap)
        print("Minimum occupancy in the scene: {}".format(min_val))
        print("Indices of positions with min occupied value: {}".format(np.where(convolved_omap <= min_val)))

        min_indices = np.where(convolved_omap <= min_val)
        print('Min indices shape: {}'.format(min_indices[0].shape))

        if debug==True:
            cv2.imshow("Convolved map", (convolved_omap/(max_val)* 255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.imwrite('./results/convolved_{}-{}.png'.format('2-2-2', 'green_bowl'), (convolved_omap/(max_val)* 255).astype(np.uint8))

        # Get the closest optimal spot
        nrows, ncols = convolved_omap.shape
        target_2d = np.array([target_6D_pose[0], target_6D_pose[1]])
        min_dist = 0.6*0.6 + 1.2*1.2
        min_pos = np.array([-1, -1])
        for i in range(nrows):
            for j in range(ncols):
                if convolved_omap[i][j] == min_val:
                    current_pos = np.array([float(i)/100 - 0.3, float(j)/100 - 0.6])
                    dist_to_target = np.linalg.norm(current_pos - target_2d)
                    # print("dist to the target: {}".format(dist_to_target))
                    if dist_to_target < min_dist:
                        min_pos = current_pos
                        min_dist = dist_to_target
        
        # Check if the returned pos is valid
        print("Predicted min pos: {}".format(min_pos))
        if min_pos[0] < 0.33 and min_pos[0] > -0.33:
            if min_pos[1] < 0.63 and min_pos[1] > -0.63:
                if debug==True:
                    if object_mesh_path=='':
                        print("Invalid object mesh path in debug mode")
                        exit()
                    buffer_spot = [min_pos[0], min_pos[1], target_6D_pose[2], target_6D_pose[3], target_6D_pose[4], target_6D_pose[5]]
                    entity = Object(mesh_path=object_mesh_path)
                    entity_pcd = entity.render_to_pose_and_get_pcd(object_pose=buffer_spot)
                    entity_omap = self.generate_2D_occupancy_map(world_dat = np.asarray(entity_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120, threshold=1, dir_path='./results/object_{}-{}.png'.format('2-2-2', 'green_bowl'), save=True)
                    fused_omap = np.logical_or(entity_omap, scene_omap)

                    cv2.imshow("Buffer spotted!", (fused_omap * 255).astype(np.uint8))
                    cv2.waitKey(0)
                    cv2.imwrite('./results/buffer_{}-{}.png'.format(scene_name, 'green_bowl'),(fused_omap * 255).astype(np.uint8))
                return min_pos

        return []
    
    
    def get_empty_spot(self, pcd = [], occ_map = [], closest_target = np.array([])):
        '''
        First get occupancy grid for the given point cloud. Now, use the coordinates of unoccupied cells
        as buffers (scale them down and transform them approximately).
        '''
        # self.visualize_pcd_with_global_coordinate_frame(pcd)
        if len(occ_map) == 0:
            occ_map = self.generate_2D_occupancy_map(np.asarray(pcd.points)*100, threshold=1, dir_path='./results/occ_map=2-2-2.png')
        print("occ_map shape: {}".format(occ_map.shape))
        # convert_to_occupancy_map(np.asarray(pcd.points)*100, threshold=500, dir_path='./results/occ_map_thr=1.png')
        # x_limits = [-.3, .3]
        # y_limits = [-.6, .6]

        # Generating empty spots (random sampling)
        x_lims, y_lims = occ_map.shape
        zero_coords = np.where(occ_map == 0)
        print(zero_coords, type(zero_coords))
        empty_spots = np.zeros(shape=(len(zero_coords[0]), 2), dtype=float)
        empty_spots[:, 0] = zero_coords[0]
        empty_spots[:, 1] = zero_coords[1]
        empty_spots = (empty_spots/100) - np.array([0.3, 0.6], dtype=float)
        print(empty_spots, type(empty_spots), empty_spots.shape)

        closest_empty_spot = None
        if len(closest_target) == 0:
            closest_empty_spot = self.get_closest_to(empty_spots)
        else:
            closest_empty_spot = self.get_closest_to(empty_spots, target_pos=closest_target)
        return closest_empty_spot
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[closest_empty_spot[0], closest_empty_spot[1], 0])
        # o3d.visualization.draw_geometries([pcd, mesh_frame])

class TaskPlanner(object):
    """Plan object operation sequence for the whole task.
    Receive the names and target poses of target objects. Get the current poses of target objects by calling the
    perception node. Then plan the sequence operate the list of objects. Finally, complete each object operation
    with the class called MotionPlanner.
    """

    def __init__(self, blocks=[], goal_cartesian_poses=[]):
        """Inits TaskPlanner with object names and corresponding target poses.
        :param blocks: A list of object names
        :param goal_cartesian_poses: A list of target poses of objects
        """
        print("start construct task planner")
        # load parameters
        rospack = rospkg.RosPack()
        task_config_path = os.path.join(rospack.get_path('ocrtoc_planning'), 'config/task_planner_parameter.yaml')
        with open(task_config_path, "r") as f:
            config_parameters = yaml.load(f)
            self._max_repeat_call = config_parameters["max_repeat_call"]
            self._grasp_distance = config_parameters["grasp_distance"]
            self._min_angle = config_parameters["min_angle"] * math.pi / 180
            self._grasp_up_translation = config_parameters["grasp_up_translation"]
            self._start_grasp_index = config_parameters["start_grasp_index"]
            self._end_grasp_index = config_parameters["end_grasp_index"]
            self._pick_via_up = config_parameters["pick_via_up"]
            # if the plane distance between two objects smaller than distance threshold,
            # the two objects are considered to be stacked
            self._distance_threshold = config_parameters["distance_threshold"]
        self._task_planner_sub = rospy.Subscriber('/start_task_planner', Int64, self.task_planner_callback, queue_size=1)
        self._service_get_sim_pose = rospy.ServiceProxy('/get_model_state', GetModelState)
        self._service_get_target_pose = rospy.ServiceProxy('/perception_action_target', PerceptionTarget)
        self._service_get_fake_pose = rospy.ServiceProxy('/fake_perception_action_target', PerceptionTarget)
        self._world_axis_z = np.array([0, 0, 1])
        self._transformer = TransformInterface()
        self._motion_planner = MotionPlanner()
        
        # Setting up camera interfaces
        self.arm_topic= 'arm_controller/command'
        self.color_info_topic_name= '/realsense/color/camera_info'
        self.color_topic_name= '/realsense/color/image_raw'
        self.depth_topic_name= '/realsense/aligned_depth_to_color/image_raw'
        self.points_topic_name= '/realsense/depth/points'
        self.kinect_color_topic_name= '/kinect/color/image_rect_color'
        self.kinect_depth_topic_name= '/kinect/depth_to_color/image_raw'
        self.kinect_points_topic_name= '/kinect/depth/points'
        self.transform_from_frame= 'world'
        
        # self.arm_controller = ArmController(topic = self.config['arm_topic'])
        self.camera_interface = CameraInterface()
        self.transform_interface = TransformInterface()
        # Subscribing to realsense and setting it up (camera attached to arm's end effector link)
        self.camera_interface.subscribe_topic(self.color_info_topic_name, CameraInfo)
        self.camera_interface.subscribe_topic(self.color_topic_name, Image)
        self.camera_interface.subscribe_topic(self.points_topic_name, PointCloud2)
        time.sleep(2)
        self.color_transform_to_frame = self.get_color_image_frame_id()
        self.points_transform_to_frame = self.get_points_frame_id()
        
        # Subscribing to kinect and setting it up (external camera)
        self.camera_interface.subscribe_topic(self.kinect_color_topic_name, Image)
        self.camera_interface.subscribe_topic(self.kinect_points_topic_name, PointCloud2)
        time.sleep(2)
        self.kinect_color_transform_to_frame = self.get_kinect_color_image_frame_id()
        self.kinect_points_transform_to_frame = self.get_kinect_points_frame_id()
        
        # Reconstruction config
        self.reconstruction_config = {
            'x_min': -0.20,
            'y_min': -0.6,
            'z_min': 0.0, # z_min: -0.05
            'x_max': 0.3,
            'y_max': 0.6,
            'z_max': 0.4,
            'nb_neighbors': 50,
            'std_ratio': 2.0,
            'voxel_size': 0.0015,
            'icp_max_try': 5,
            'icp_max_iter': 2000,
            'translation_thresh': 3.95,
            'rotation_thresh': 0.02,
            'max_correspondence_distance': 0.02
        }
        
        self.clear_box_flag = False
        
        # MiscFunctions class instance named o3dp (as it is same as o3dp (since the import failed, we are directly using the source code 
        #   from the repo))
        self.o3dp = MiscFunctions()
        
        # Occupancy map and buffer spot sampler class
        self.occ_and_buffer = OccupancyAndBuffer()


        self.object_label_mesh_path_dict = {}
        self.duplicate_object_real_label_dict = {}
        
        
        self.block_labels = blocks
        self.object_goal_pose_dict = self.get_goal_pose_dict(self.block_labels, goal_cartesian_poses)
        self.block_labels_with_duplicates = self.object_goal_pose_dict.keys()
        
        self.object_init_pose_dict = {}
        self.object_pick_grasp_pose_dict = {}
        self.object_place_grasp_pose_dict = {}
        self.detected_object_label_list = []
        # self.red_nodes = []
        # self.black_nodes = []

        self.global_object_states = {} # <object_name>: {done: True/False, object_stack: list_of_other_objects, occupied:True/False, temp_done: True/False}
        self.object_stacks = {}

        self.object_entities = {}
        
        print("#"*40)
        print("Block labels: {}".format(self.block_labels))
        print("Goal pose dictionary: {}".format(self.object_goal_pose_dict))
        print("#"*40)
        print("task planner constructed")

    # def get_goal_pose_list_from_input(self, goal_cartesian_poses):
    #     '''
    #     The task information is very crude. This function helps us extract the goal poses from task information 
    #     and returns a list of goal poses
    #     '''
    #     goal_poses = []
    #     for goal in goal_cartesian_poses:
    #         pose = goal.poses[0]
    #         goal_poses.append(pose)
    #     return goal_poses
    
    def get_goal_pose_dict(self, block_labels, goal_poses):
        '''
        Given a list of goal poses and block labels, this function returns a dictionary with labels as keys
        
        Parameters:
        block_labels: List of labels (strings)
        goal_poses: List of lists of block poses 
        '''
        goal_pose_dict = {}
        # print(zip(block_labels, goal_poses))
        for label, poses in zip(block_labels, goal_poses):
            print("Poses: {}".format(poses))
            print("Poses type: {}".format(type(poses)))
            for i, pose in enumerate(poses.poses):
                goal_pose_dict["{}_v{}".format(label, i)] = pose
                self.object_label_mesh_path_dict["{}_v{}".format(label, i)] = '/root/ocrtoc_ws/src/ocrtoc_materials/models/{}/textured.obj'.format(label)
                self.duplicate_object_real_label_dict["{}_v{}".format(label, i)] = label
            # goal_pose_dict[label] = pose
        return goal_pose_dict    

    def get_pose_perception(self, target_object_list):
        """
        This function gets current and pick grasp poses from the perception node. It then updates object_init_pose_dict, 
        object_pick_grasp_pose_dict and object_place_grasp_pose_dict. Place grasp poses have same orienatation as pick 
        grasp poses. Their position will be equal to target pose of the object, with modified z (position.z += thresh), 
        as we are trying to drop the object from certain height instead of placing it precisely in its position.
        
        :param target_object_list: A list of target object names (with duplicate objects named as _v<duplicate_number>)
        
        # perception service message:
        # string[] target_object_list
        # ---
        # ocrtoc_perception/PerceptionResult[] perception_result_list
        # ocrtoc_perception/PerceptionResult.msg
        # string object_name
        # bool be_recognized
        # geometry_msgs/PoseStamped object_pose
        # bool is_graspable
        # geometry_msgs/PoseStamped grasp_pose
        """
        rospy.wait_for_service('/perception_action_target')
        request_msg = PerceptionTargetRequest()
        request_msg.target_object_list = target_object_list
        print("Target object_list: {}".format(target_object_list))
        rospy.loginfo('Start to call perception node')
        perception_result = self._service_get_target_pose(request_msg)
        rospy.loginfo('Perception finished')
        
        print("Perception result: {}".format(perception_result))
        
        rospy.loginfo(str(len(perception_result.perception_result_list)) + ' objects are graspable:')
        rospy.loginfo('Graspable objects information: ')
        
               

        for result in perception_result.perception_result_list:
            if result.be_recognized: 
                def get_rotation (quat):
                        orientation_q = quat.orientation
                        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                        return np.array([roll, pitch, yaw])
                f_pose = self.object_goal_pose_dict[result.object_name] #self._goal_cartesian_pose_dic[result.object_name]
                print("result.object_pose.pose: {}\nf_pose: {}".format(result.object_pose.pose, f_pose))
                
                # rrc: Object pose and goal pose comparison happens here (the objects with similar goal and initial poses will be deleted)
                
                init_cart_coords = np.array([result.object_pose.pose.position.x, result.object_pose.pose.position.y, result.object_pose.pose.position.z])
                final_cart_coords = np.array([f_pose.position.x, f_pose.position.y, f_pose.position.z])
                
                init_quat = get_rotation(result.object_pose.pose)
                final_quat = get_rotation(f_pose)
                
                cart_dist = np.linalg.norm(init_cart_coords - final_cart_coords)
                quat_dist = np.linalg.norm(init_quat - final_quat) # np.linalg.norm(result.object_pose.pose[4:7] - self._goal_cartesian_pose_dic[result.object_name][4:7])
                cartesian_dist_thresh = 0.05
                quaternion_dist_thresh = 0.001
                print("distance check !!!!!!",result.object_name , np.abs(init_cart_coords[1] - final_cart_coords[1]))
                if 'clear_box' not in result.object_name:
                    if (cart_dist < cartesian_dist_thresh) and (quat_dist < quaternion_dist_thresh):
                        # del self._completed_objects[result.object_name]
                        continue
                else:
                    if (np.abs(init_cart_coords[1] - final_cart_coords[1]) < 0.2):
                        # del self._completed_objects[result.object_name]
                        continue
                
                # rrc: Object pose and goal pose comparison ends here
                
                if result.is_graspable:
                    
                    self.object_init_pose_dict[result.object_name] = result.object_pose.pose
                    self.object_pick_grasp_pose_dict[result.object_name] = copy.deepcopy(result.grasp_pose.pose)
                    
                    self.object_place_grasp_pose_dict[result.object_name] = self.get_target_grasp_pose2(result.object_pose.pose, result.grasp_pose.pose, self.object_goal_pose_dict[result.object_name])
                    
                    # self.object_place_grasp_pose_dict[result.object_name] = copy.deepcopy(result.grasp_pose.pose)
                    # self.object_place_grasp_pose_dict[result.object_name].position.z += 0.2 # Dropping from certain minimal height
                    
                    self.detected_object_label_list.append(result.object_name)
            else:
                pass
            
    

    # modulate intelligence grasp pose to avoid collision
    def get_artificial_intelligence_grasp_pose(self, intelligence_grasp_pose):
        artificial_intelligence_grasp_pose = Pose()
        intelligence_pose_matrix = self._transformer.ros_pose_to_matrix4x4(intelligence_grasp_pose)
        intelligence_pose_axis_z = intelligence_pose_matrix[0:3, 2]
        print('intelligence pose z axis: {}'.format(intelligence_pose_axis_z))
        intelligence_cos_alpha = np.dot(intelligence_pose_axis_z, self._world_axis_z)
        intelligence_alpha = math.acos(intelligence_cos_alpha)
        print('intelligence pose angle (degree): {}'.format(intelligence_alpha * 180 / math.pi))

        if intelligence_alpha > self._min_angle:
            rospy.loginfo('intelligence pose in specified range, no need to adjust')
            artificial_intelligence_grasp_pose = intelligence_grasp_pose
        else:
            rospy.loginfo('intelligence pose out of specified range, may cause undesired behavior, it should be adjusted')
            delta_angle = self._min_angle - intelligence_alpha
            rotation_axis = np.cross(intelligence_pose_axis_z, -self._world_axis_z)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            print('rotation axis in world frame:\n{}'.format(rotation_axis))
            intelligence_rotation_matrix = intelligence_pose_matrix[0:3, 0:3]
            rotation_axis = np.matmul(np.linalg.inv(intelligence_rotation_matrix), rotation_axis)  # transform rotation axis from world frame to end-effector frame
            print('rotation axis in end frame: \n{}'.format(rotation_axis))
            rotation_quaternion = np.array([math.cos(delta_angle/2), rotation_axis[0]*math.sin(delta_angle/2),\
            rotation_axis[1]*math.sin(delta_angle/2), rotation_axis[2]*math.sin(delta_angle/2)])  # [w, x, y, z]
            rotation_matrix = transforms3d.quaternions.quat2mat(rotation_quaternion)
            transformation_matrix = np.eye(4, dtype=np.float)
            transformation_matrix[0:3, 0:3] = rotation_matrix
            artificial_intelligence_grasp_pose_matrix = np.matmul(intelligence_pose_matrix, transformation_matrix)  # do rotation

            linear_transformation_matrix = np.eye(4, dtype=np.float)
            linear_transformation_matrix[2, 3] = self._grasp_up_translation
            artificial_intelligence_grasp_pose_matrix = np.matmul(linear_transformation_matrix, artificial_intelligence_grasp_pose_matrix)  # do upright translation
            artificial_intelligence_grasp_pose = self._transformer.matrix4x4_to_ros_pose(artificial_intelligence_grasp_pose_matrix)

            artificial_intelligence_grasp_pose_axis_z = artificial_intelligence_grasp_pose_matrix[0:3, 2]
            artificial_intelligence_cos_alpha = np.dot(artificial_intelligence_grasp_pose_axis_z, self._world_axis_z)
            artificial_intelligence_alpha = math.acos(artificial_intelligence_cos_alpha)
            print('delta angle: {}'.format(delta_angle * 180 / math.pi))
            print('artificial intelligence pose angle (degree): {}'.format(artificial_intelligence_alpha * 180 / math.pi))
            print('artificial intelligence pose z axis: {}'.format(artificial_intelligence_grasp_pose_axis_z))

        return artificial_intelligence_grasp_pose

    # get grasp pose in target configuration
    def get_target_grasp_pose(self, pose1, grasp_pose1, pose2):
        pose1_matrix = self._transformer.ros_pose_to_matrix4x4(pose1)
        grasp_pose1_matrix = self._transformer.ros_pose_to_matrix4x4(grasp_pose1)
        transformation_matrix = np.dot(np.linalg.inv(pose1_matrix), grasp_pose1_matrix)

        pose2_matrix = self._transformer.ros_pose_to_matrix4x4(pose2)
        grasp_pose2_matrix = np.dot(pose2_matrix, transformation_matrix)
        grasp_pose2_matrix[2, 3] = grasp_pose2_matrix[2, 3] + 0.02  # to avoid collision with desk
        grasp_pose2 = self._transformer.matrix4x4_to_ros_pose(grasp_pose2_matrix)

        return grasp_pose2

    # get grasp pose in target configuration
    def get_target_grasp_pose2(self, pose1, grasp_pose1, pose2):
        pose1_matrix = self._transformer.ros_pose_to_matrix4x4(pose1)
        grasp_pose1_matrix = self._transformer.ros_pose_to_matrix4x4(grasp_pose1)
        transformation_matrix = np.dot(np.linalg.inv(pose1_matrix), grasp_pose1_matrix)

        # change target pose orientation to initial pose orientation
        pose2.orientation.x = pose1.orientation.x
        pose2.orientation.y = pose1.orientation.y
        pose2.orientation.z = pose1.orientation.z
        pose2.orientation.w = pose1.orientation.w

        pose2_matrix = self._transformer.ros_pose_to_matrix4x4(pose2)
        grasp_pose2_matrix = np.dot(pose2_matrix, transformation_matrix)
        grasp_pose2_matrix[2, 3] = grasp_pose2_matrix[2, 3] + 0.02  # to avoid collision with desk
        grasp_pose2 = self._transformer.matrix4x4_to_ros_pose(grasp_pose2_matrix)

        return grasp_pose2
    
    def search_strings2(self, object_name, searchable_list):
        return any([x in object_name for x in searchable_list])
    
    def search_strings(self, string_list, searchable):
        '''
        Searches for a substring in a given list of strings
        
        Input: 
        string_list: str (list of strings to be searched in)
        searchable: str (substring to search for)
        
        Return:
        Bool: True/False
        '''
        return any([searchable in x for x in string_list])
        
    def get_color_image(self): # newly added
        return self.camera_interface.get_numpy_image_with_encoding(self.color_topic_name)[0]
        
    def get_color_image_frame_id(self):
        '''Realsense topic'''
        return self.camera_interface.get_ros_image(self.color_topic_name).header.frame_id
    
    def get_points_frame_id(self):
        '''Realsense topic'''
        return self.camera_interface.get_ros_points(self.points_topic_name).header.frame_id
    
    def get_kinect_color_image_frame_id(self):
        '''Kinect camera'''
        return self.camera_interface.get_ros_image(self.kinect_color_topic_name).header.frame_id

    def get_kinect_depth_image_frame_id(self):
        '''Kinect camera'''
        return self.camera_interface.get_ros_image(self.kinect_depth_topic_name).header.frame_id

    def get_kinect_points_frame_id(self):
        '''Kinect camera'''
        return self.camera_interface.get_ros_points(self.kinect_points_topic_name).header.frame_id
    
    def get_kinect_points_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.kinect_points_transform_to_frame)
    
    def get_kinect_color_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.kinect_color_transform_to_frame)
    
    def kinect_get_pcd(self, use_graspnet_camera_frame = False):
        return self.camera_interface.get_o3d_pcd(self.kinect_points_topic_name)
    
    def kinect_process_pcd(self, pcd, reconstruction_config):
        points, colors = self.o3dp.pcd2array(pcd)
        mask = points[:, 2] > reconstruction_config['z_min']
        mask = mask & (points[:, 2] < reconstruction_config['z_max'])
        mask = mask & (points[:, 0] > reconstruction_config['x_min'])
        mask = mask & (points[:, 0] < reconstruction_config['x_max'])
        mask = mask & (points[:, 1] < reconstruction_config['y_max'])
        mask = mask & (points[:, 1] > reconstruction_config['y_min'])
        pcd = self.o3dp.array2pcd(points[mask], colors[mask])
        return pcd.voxel_down_sample(reconstruction_config['voxel_size'])
    
    def capture_pcd(self, use_camera='kinect'):
        t1 = time.time()
        pcds = []
        color_images = []
        camera_poses = []
        # capture images by realsense. The camera will be moved to different locations.
        if use_camera == 'kinect': # in ['kinect', 'both']:
            points_trans_matrix = self.get_kinect_points_transform_matrix()
            full_pcd_kinect = self.kinect_get_pcd(use_graspnet_camera_frame = False) # in sapien frame.
            full_pcd_kinect.transform(points_trans_matrix)
            full_pcd_kinect = self.kinect_process_pcd(full_pcd_kinect, self.reconstruction_config)
                # pcds.append(full_pcd_kinect)
                # kinect_image = self.get_kinect_image()
                # kinect_image = cv2.cvtColor(kinect_image, cv2.COLOR_RGBA2RGB)
                # kinect_image = cv2.cvtColor(kinect_image, cv2.COLOR_RGB2BGR)
                # if self.debug_kinect:
                #     cv2.imshow('color', cv2.cvtColor(kinect_image, cv2.COLOR_RGB2BGR))
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                # color_images.append(kinect_image)

                # if self.debug:
                #     print('points_trans_matrix:', points_trans_matrix)
                # camera_poses.append(self.get_kinect_color_transform_matrix())
            return full_pcd_kinect
        elif use_camera == 'realsense':
            print("Try with Kinect please, Realsense is not ready yet;( ******++++++++++++***********")
        
    def get_point_cloud_from_kinect(self):
        '''
        Capture and get point cloud from Kinect camera (Will be used to build occupancy map)
        '''
        pcd_kinect = self.capture_pcd(use_camera='kinect')
        # import open3d as o3d
        # o3d.visualization.draw_geometries([pcd_kinect])
        return pcd_kinect

    def create_global_object_dict(self):
        for i, key in enumerate(self.object_stacks.keys()):
            self.global_object_states[key] = {
                'done': False,
                'object_stack': self.object_stacks[key]['objects_under_it'],
                'occupied': False,
                'temp_done': False
            }
        print("Global object state dictionary: {}".format(self.global_object_states))

        
    def create_data_model(self, nodes):
        """Stores the data for the problem."""
        data = {}
        from scipy.spatial import distance_matrix

        #nodes = [[0, 0], [41, 44], [2, 1], [27, 5], [3, 49], [20, 4], [18, 4], [39, 44], [23, 12]] # 2431
        #nodes = [[0, 0], [23, 28], [34, 28], [42, 46], [11, 44], [19, 25], [36, 27], [0, 25], [7, 38]] # 4123
        #nodes = [[6, 6], [16, 27], [14, 2], [24, 45], [6, 14], [49, 27], [44, 0], [17, 11], [36, 17]] # 4132
        #nodes = [[0, 0], [43, 19], [18, 47], [41, 38], [50, 16], [45, 12], [32, 17], [4, 5], [43, 32]] # 2143
        #nodes = [[6, 6], [11, 47], [27, 4], [5, 48], [21, 34], [26, 19], [37, 10], [8, 37], [29, 9]] # 3142
        # nodes = [[0, 0], [31, 34], [19, 27], [9, 19], [12, 34], [43, 32], [5, 26], [17, 25], [46, 43]] # 2301 Working default
        #nodes = [[9, 1], [22, 29], [49, 35], [20, 19], [8, 13], [8, 13], [16, 18], [13, 11]] # 1432
        # Trying 3D nodes
        # nodes = [[0, 0, 0], [31, 34, 31], [19, 27, 1], [9, 19, 5], [12, 34, 10], [43, 32, 38], [5, 26, 65], [17, 25, 20], [46, 43, 90] ]

        dm = distance_matrix(nodes, nodes).tolist()

        data['distance_matrix'] = dm
        # data['pickups_deliveries'] = [
        #     [1, 5],
        #     [2, 6],
        #     [3, 7],
        #     [4, 8]
        # ]
        data['pickups_deliveries'] = []
        n_picks = (len(nodes) - 1)/2
        for i in range(n_picks):
            data['pickups_deliveries'].append([i+1, n_picks+i+1])
        
        data['num_vehicles'] = 1
        data['depot'] = 0
        
        data['demands'] = [] # [0] # [0, 1, 1, 1, 1, -1, -1, -1, -1] Had to comment for python2.7
        for i in range(len(nodes)):
            if i==0:
                data['demands'].append(0)
            elif i<= int(len(nodes)/2):
                data['demands'].append(1)
            else:
                data['demands'].append(-1)
                
        data['vehicle_capacities'] = [1] # Commented for python2.7
        return data

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        # print(f"Objective: {solution.ObjectiveValue()}")
        print("Objective: {}".format(solution.ObjectiveValue()))
        # print("")
        total_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(float(route_distance)/100)
            print(plan_output)
            total_distance += float(route_distance)/100
        print('Total Distance of all routes: {}m'.format(total_distance))
        # [END solution_printer]
        
    def get_final_sequence_from_ORsolution(self, data, manager, routing, solution):
        """Returns the final sequence of actions by decoding the solution spit out by OR TOOLS optimization
        Returns:
            final_sequence: [list of action_indices]
        """
        final_sequence = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                final_sequence.append(manager.IndexToNode(index))
                # plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                # route_distance += routing.GetArcCostForVehicle(
                    # previous_index, index, vehicle_id)
            final_sequence.append(manager.IndexToNode(index))
            # plan_output += '{}\n'.format(manager.IndexToNode(index))
            # plan_output += 'Distance of the route: {}m\n'.format(float(route_distance)/100)
            print("Final sequence: {}, type: {}".format(final_sequence, type(final_sequence)))
            # total_distance += float(route_distance)/100
        return final_sequence

    def solve_orTools(self, data):
        """Entry point of the program."""
        # Instantiate the data problem.
        # [START data]
        # data = create_data_model()
        # [END data]

        # Create the routing index manager.
        # [START index_manager]
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'], data['depot'])
        # [END index_manager]

        # Create Routing Model.
        # [START routing_model]
        routing = pywrapcp.RoutingModel(manager)

        # [END routing_model]
        
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            # print("From node: {}".format(from_node))
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            2,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Define cost of each arc.
        # [START arc_cost]
        def distance_callback(from_index, to_index):
            """Returns the manhattan distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node]*100)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # [END arc_cost]

        # Add Distance constraint.
        # [START distance_constraint]
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            300000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        # [END distance_constraint]

        # Define Transportation Requests.
        # [START pickup_delivery_constraint]
        for request in data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
        # [END pickup_delivery_constraint]

        # Setting first solution heuristic.
        # [START parameters]
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC) # .PARALLEL_CHEAPEST_INSERTION)
        # [END parameters]

        # Solve the problem.
        # [START solve]
        solution = routing.SolveWithParameters(search_parameters)
        # [END solve]

        # Print solution on console.
        # [START print_solution]
        if solution:
            self.print_solution(data, manager, routing, solution)
            
        # print(solution)
        plan = self.get_final_sequence_from_ORsolution(data, manager, routing, solution)
        return solution, plan
        # [END print_solution]

    def check_if_parents_done(self, object_key, local_object_states):
        '''If the given object is supposed to go into a stack, check if its detected parents are already done
        '''
        # print("Object_key {}".format(object_key))
        list_of_parents = self.global_object_states[object_key]['object_stack']
        if len(list_of_parents)==0:
            return True
        print(list_of_parents)
        print(type(list_of_parents))
        for parent in list_of_parents:
            if parent in self.detected_object_label_list:
                if local_object_states[parent]['done']==True or local_object_states[parent]['temp_done']==True:
                    continue
                else:
                    action = "done"
                    f = open('/root/plan_result.txt', 'a')
                    f.write("\nMy parents not completed ({} - is parent to - {})!".format(parent, object_key))
                    f.close()
                    return False
        return True

    def update_collision_statuses(self, debug=False):
        '''Gets point cloud and updates target collision statuses for all the objects
        '''
        # 1. Generate point cloud
        current_pcd = self.get_point_cloud_from_kinect()
        print("Total number of points in pcd: {}".format(len(current_pcd.points)))
        occ_map = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(current_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120)

        if debug == True:
            cv2.imshow('Occupancy map', (occ_map * 255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.imwrite('root/scene_occ_map.png',(occ_map * 255).astype(np.uint8))

        # 2. Get path to materials
        rospack = rospkg.RosPack()
        materials_path = rospack.get_path('ocrtoc_materials')
        print("Rospack path: {}*********************++++++++++++++++++++++++++++*************************+++++++++++++++++++++++++++".format(materials_path))
        # 3. Generate occupancy maps for each of the objects and compare the two maps. 
        #    If both the maps have entry '1' at any particular pixel, then the place is said to be occupied
        # FUTURE: ADD TOLERANCE (Concept: It is okay to have few pixels occupied, as long as it is not too many (set some threshold))
        pcds = []
        for object_name in self.detected_object_label_list:
            temp = object_name.split('_')
            print("Object name: {}\tObject split name: {}".format(object_name, temp))
            model_name = ''
            for i in range(len(temp)-1):
                if i==0:
                    model_name = model_name + temp[i]
                else:
                    model_name = model_name + '_' + temp[i]

            path_to_object_mesh = os.path.join(materials_path, 'models/{}/textured.obj'.format(model_name)) # visual.ply
            # Reading textured mesh and transforming it to the actual goal pose
            mesh = o3d.io.read_triangle_mesh(path_to_object_mesh)
            pose = self.object_goal_pose_dict[object_name]
            translation = np.array([pose.position.x, pose.position.y, pose.position.z]) - np.array(mesh.get_center())
            orientation = (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
            mesh.rotate(mesh.get_rotation_matrix_from_quaternion(orientation)) # center = True by default (implies rotation is applied w.r.t the center of the object)
            mesh.translate((translation[0], translation[1], translation[2])) 
            pcd =  mesh.sample_points_poisson_disk(1000, init_factor=5, pcl=None)
            occ1 = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120, dir_path='/root/occ_map_{model_name}.png')
            
            amount_required_occ1 = np.sum(occ1)
            if debug==True:
                cv2.imshow('Object Occupancy map', (occ1 * 255).astype(np.uint8))
                cv2.waitKey(0)
                cv2.imwrite('root/object_occ_map.png',(occ_map * 255).astype(np.uint8))
            # Check if occupied
            amount_occupied = np.sum(np.logical_and(occ_map, occ1))

            if debug==True:
                cv2.imshow('Resultant Occupancy map', (np.logical_and(occ_map, occ1) * 255).astype(np.uint8))
                cv2.waitKey(0)
                cv2.imwrite('root/resultant_occ_map.png',(occ_map * 255).astype(np.uint8))

            occ_perc = 100*float(amount_occupied)/float(amount_required_occ1)
            if occ_perc > 5:
                self.global_object_states[object_name]['occupied'] = True
            else:
                self.global_object_states[object_name]['occupied'] = False
            # is_occupied = np.any(np.logical_and(occ_map, occ1))
            # bnode.occupied = is_occupied
            print("{} percentage of {}'s target pose is occupied".format(occ_perc, object_name))
            print("So, occupied = {}".format(self.global_object_states[object_name]['occupied']))


    def get_next_best_action(self, action_list, action_sequence_mapping):
        '''Gets the next best action
        '''   
        self.update_collision_statuses(debug=False)

        done = True
        for action in action_list:
            name = action_sequence_mapping[str(action)]['name']
            object_name = action_sequence_mapping[str(action)]['object']
            if name=="rest_pose": # or object_name==None:
                continue
            print("Global object states: {}\nObject name: {}".format(self.global_object_states, object_name))
            if self.global_object_states[object_name]['done'] == True or self.global_object_states[object_name]['temp_done']==True:
                continue
            done = False
            if len(self.global_object_states[object_name]['object_stack']) == 0: # No parents in the stack
                if self.global_object_states[object_name]['occupied'] == True:
                    continue
                else:
                    return action
            elif self.check_if_parents_done(object_name, self.global_object_states) == True: # self.global_object_states[object_name]
                return action 
        
        # If the program reaches this point, it implies that either all the objects are done, or all the objects have partially occupied targets
        if done == True:
            action = "done"
            f = open('/root/plan_result.txt', 'a')
            f.write("\nDone!".format(name, object_name))
            f.close()

            return action
        
        # Get any undone object
        for action in action_list:
            name = action_sequence_mapping[str(action)]['name']
            object_name = action_sequence_mapping[str(action)]['object']
            if name=="rest_pose": # or object_name==None:
                continue
            if self.global_object_states[object_name]['done'] == True or self.global_object_states[object_name]['temp_done']==True:
                continue
            if self.check_if_parents_done(object_name, self.global_object_states) == True: # self.global_object_states[object_name]
                return action 

        action = "done"
        return action
    
    def execute_plan(self, action_list, action_sequence_mapping):
        '''Given the list of actions in sequence, execute them one by one and return the completed objects list

        '''
        success = False
        completed_objects = []

        done = False

        # f = open('/root/plan_result.txt', 'a')
        # f.close()

        while not done:
            action = self.get_next_best_action(action_list, action_sequence_mapping)
            if action == "done":
                done = True
                continue

            name = action_sequence_mapping[str(action)]['name']
            object_name = action_sequence_mapping[str(action)]['object']

            f = open('/root/plan_result.txt', 'a')
            f.write("\nAction_name: {}\tObject name: {}".format(name, object_name))
            f.close()

            buffer_pose = None

            # If target is partially or fully occupied, sample a buffer spot
            if self.global_object_states[object_name]['occupied'] == True and len(self.global_object_states[object_name]['object_stack'])==0:
                f = open('/root/plan_result.txt', 'a')
                f.write("\nGoing for a buffer spot due to partial occupancy")
                f.close()
                # position = self.object_goal_pose_dict[object_name].position
                # orientation = self.object_goal_pose_dict[object_name].orientation
                # target_cart_pose = np.array([position.x, position.y, position.z,
                #                              orientation.x, orientation.y, 
                #                              orientation.z, orientation.w])
                target_pose = self.object_goal_pose_dict[object_name]
                position = target_pose.position
                pos = [position.x, position.y, position.z]
                orientation = target_pose.orientation
                quat_or = [orientation.x, orientation.y, orientation.z, orientation.w]
                r = Rotation.from_quat(quat_or) # Convert quaternion to rotation matrix
                euler = r.as_euler('xyz', degrees=False)
                target_6D_pose = [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]

                print("object_mesh_path={}".format(self.object_label_mesh_path_dict[object_name]))
                current_pcd = self.get_point_cloud_from_kinect()
                scene_omap = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(current_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120,
                                                    threshold=1, dir_path='./results/{}-{}.png'.format('-', object_name), save=False)
                # buffer_spot_2d = self.occ_and_buffer.marching_grid(scene_pcd=current_pcd, object_mesh_path=self.object_label_mesh_path_dict[object_name],
                #                                                    target_pose_6D= target_6D_pose, OCC_THRESH=1.0, scene_name='-', object_name='-')

                buffer_spot_2d = self.occ_and_buffer.convolutional_buffer_sampler(scene_omap=scene_omap,
                                                                    entity_omap = self.object_entities[object_name].kernel_map, 
                                                                    target_6D_pose=target_6D_pose, 
                                                                    object_mesh_path=self.object_label_mesh_path_dict[object_name],
                                                                    debug=False)
                if len(buffer_spot_2d) == 0:
                    print("Invalid buffer spot!!!")
                    buffer_spot_2d = [target_6D_pose[0], target_6D_pose[1]]
                # buffer_spot_2d = self.occ_and_buffer.get_empty_spot(occ_map=occ_map, closest_target=target_cart_pose)
                buffer_pose = copy.deepcopy(self.object_init_pose_dict[object_name])
                buffer_pose.position.x = buffer_spot_2d[0]
                buffer_pose.position.y = buffer_spot_2d[1]

            # pick object
            print("Going to pick {}\t|\t".format(object_name))
            self._motion_planner.fake_place()
            success = self.go_pick_object(object_name=object_name)
            self.global_object_states[object_name]['temp_done'] = True
            if success == False:
                print("Pick failed")
            else:
                print("Pick success")

            # place object
            if success == True:
                print("Going to place {}\t|\t".format(object_name))
                success = self.go_place_object(object_name, final_place_pose=buffer_pose)
                if success == False:
                    print("Place failed")
                else:
                    print("Place success")
                    completed_objects.append(object_name)
                    self.global_object_states[object_name]['done'] = True

            f = open('/root/plan_result.txt', 'a')
            f.write("\nAction_success: {}\n-----------------------------------------------------------------\n".format(success))
            f.close()

        return completed_objects
            
    def set_temp_status_not_done(self):
        '''This function sets the temp_done status for each object as False
        '''
        for i, key in enumerate(self.global_object_states.keys()):
            self.global_object_states[key]['temp_done'] = False

    # get poses of part of objects each call of perception node, call perception node and plan task several times
    def cycle_plan_all(self):
        """Plan object operation sequence and execute operations recurrently.
        Process:
        1. Get target poses
        2. Get current poses from perception node for remaining objects
        3. Build a graph if it does not exist. Otherwise, update it with the new info on current poses
        4. Solve the graph - Real time task planning and execution 
        5. Go back to step 1 if all objects are not picked and placed yet
        """

        print("Cycle plan function started executing!")
        left_object_labels = copy.deepcopy(self.block_labels_with_duplicates)
        
        # Remove clear_box from the list of movable objects
        temp = []
        for object in left_object_labels:
            if self.search_strings2(object, ['clear_box']): #, 'book', 'round_plate', 'plate_holder']):
                continue
            temp.append(object)
        left_object_labels = copy.deepcopy(temp)

        import pickle
        object_dict = {}
        for i, key in enumerate(self.object_goal_pose_dict.keys()):
            position = self.object_goal_pose_dict[key].position
            p = [position.x, position.y, position.z]
            if self.search_strings2(key, ['clear_box', 'tray']):
                p[2] = p[2] - 0.2
            elif self.search_strings2(key, ['plate']):
                p[2] = p[2] - 0.1
            elif self.search_strings2(key, ['bowl']):
                p[2] = p[2] - 0.5
            orientation = self.object_goal_pose_dict[key].orientation
            o = [orientation.w, orientation.x, orientation.y, orientation.z]
            posep = []
            for pi in p:
                posep.append(pi)
            for oi in o:
                posep.append(oi)
            object_dict[key] = {
                'object_label': self.duplicate_object_real_label_dict[key],
                'pose': posep
            }
        OBJECT_DIR_PATH = '/root/ocrtoc_ws/src/stack_detection/object_dict.npz'
        MESH_DIR = '/root/ocrtoc_ws/src/ocrtoc_materials/models/'
        SAVE_PATH = '/root/ocrtoc_ws/src/stack_detection/scene_dict.npz' 
        # np.savez_compressed(OBJECT_DIR_PATH, data=object_dict)
        with open(OBJECT_DIR_PATH, 'wb') as f:
            pickle.dump(object_dict, f)
        print("Running stack detection")
        command = '/root/ocrtoc_ws/src/stack_detection/run_script.sh {} {} {}'.format(OBJECT_DIR_PATH, MESH_DIR, SAVE_PATH)
        os.system(command)
        # 1.1. Load the dictionary and convert it into object stack dictionary 
        with open(SAVE_PATH, 'rb') as f:
            self.object_stacks = pickle.load(f)     

        # 1.2 Create the global object dictionary using stack information
        self.create_global_object_dict()
             
        # 1.3 Create occupancy kernel maps for each object
        for obj_label in left_object_labels:
            target_pose = self.object_goal_pose_dict[obj_label]
            position = target_pose.position
            pos = [position.x, position.y, position.z]
            orientation = target_pose.orientation
            quat_or = [orientation.x, orientation.y, orientation.z, orientation.w]
            r = Rotation.from_quat(quat_or) # Convert quaternion to rotation matrix
            euler = r.as_euler('xyz', degrees=False)
            target_6D_pose = [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]
            self.object_entities[obj_label] = Object(self.object_label_mesh_path_dict[obj_label])
            self.object_entities[obj_label].get_occupancy_map_kernel(target_6D_pose, self.occ_and_buffer, debug=False)
                   
        # 1. Create black nodes for target poses
        # self.black_nodes = self.initialize_target_black_nodes(self.block_labels_with_duplicates)
        # 2. Create red nodes for initial poses
        # self.red_nodes = self.initialize_initial_red_nodes(self.block_labels_with_duplicates)

        f = open('/root/plan_result.txt', 'w+')
        f.close()
            
        count = 0
        label_count = 1
        while len(left_object_labels) > 0 and count <= 5:

            self.set_temp_status_not_done()

            count += 1
            # 3. Get information about left objects from perception node
            
            self.detected_object_label_list = []
            
            rospy.loginfo('Try to get information of left objects from perception node')
            self.get_pose_perception(left_object_labels)
            print("Detected object list: {}".format(self.detected_object_label_list))

            f = open('/root/plan_result.txt', 'a')
            f.write("\nRound number: {}\nDetected object list: {}".format(count, self.detected_object_label_list))
            f.close()

            # 4. Create pick and place grasp pose lists and final list of nodes
            rest_pose = Pose()
        
            rest_pose.position.x = -0.112957249941
            rest_pose.position.y = 2.9801544038e-05
            rest_pose.position.z = 0.590340135745
            rest_pose.orientation.x = -0.923949504923
            rest_pose.orientation.y = 0.382514458771
            rest_pose.orientation.z = -3.05585242637e-05
            rest_pose.orientation.w = 1.57706453844e-05

            pick_list = []
            place_list = []
            action_sequence_mapping = {}
            action_sequence_mapping['0'] = {
                'name': "rest_pose",
                'object': "none",
                'pose': rest_pose
            }
            
            
            detected_object_list = []
            for left_object in left_object_labels:
                if left_object in self.detected_object_label_list:
                    detected_object_list.append(left_object)
            
            
            self.detected_object_label_list = detected_object_list
            
            f = open('/root/plan_result.txt', 'a')
            f.write("\nObject list for the current round: {}".format(self.detected_object_label_list))
            f.close()
            
            n_detected_objects = len(self.detected_object_label_list )
            print(self.detected_object_label_list)
            # nodes = []
            
            
            
            for lab_index, label in enumerate(self.detected_object_label_list ):
                
                print("{} is in the object list*********************+++++++++++++*****************".format(label))
                # node.pickable = True
                pose_cartesian = self.object_init_pose_dict[label].position
                pick_grasp_cartesian = self.object_pick_grasp_pose_dict[label].position
                place_grasp_cartesian = self.object_place_grasp_pose_dict[label].position
                pick_list.append([pick_grasp_cartesian.x, pick_grasp_cartesian.y, pick_grasp_cartesian.z])
                place_list.append([place_grasp_cartesian.x, place_grasp_cartesian.y, place_grasp_cartesian.z])
                action_sequence_mapping[str(int(lab_index)+1)]={'name': "{}_pick".format(label),
                                                            'object': label,
                                                            'pose': self.object_pick_grasp_pose_dict[label]}
                action_sequence_mapping[str(n_detected_objects + int(lab_index) + 1)] = {'name': "{}_place".format(label),
                                                                                        'object': label,
                                                                                        'pose': self.object_place_grasp_pose_dict[label]}
            
            # 5. Create a list a nodes with pick and place elements in it (is specific order)
            # First node is the rest cartesian node (denotes the depot in CVRP optimization via OR TOOLS)
            rest_cartesian = [-0.112957249941, 2.9801544038e-05, 0.590340135745]
            nodes = []
            nodes.append(rest_cartesian)
            for element in pick_list:
                nodes.append(element)
            for element in place_list:
                nodes.append(element)

            # 6. Create the data model using nodes information
            data = self.create_data_model(nodes)
            # 7. Solve and get plan
            solution, plan = self.solve_orTools(data)
            print(action_sequence_mapping)
            print("Decoded solution")
            
            print("plan")
            print(plan)
            
            
            for action in plan:
                print(action)
                print('{}-->'.format(action_sequence_mapping[str(action)]['name']))
            print("End!")
            
            print("Executing the plan!")
            
            # 8. Execute plan
            completed_objects = self.execute_plan(plan, action_sequence_mapping)

            f = open('/root/plan_result.txt', 'a')
            f.write("\nCompleted objects: {}".format(completed_objects))
            f.close()
            
            # 9. Remove completed objects
            temp = []
            for object in left_object_labels:
                if object in completed_objects:
                    continue
                temp.append(object)
            left_object_labels = temp
            print("left objects: {}".format(left_object_labels))

            f = open('/root/plan_result.txt', 'a')
            f.write("\nLeft objects: {}\n===============================================================================\n".format(left_object_labels))
            f.close()

            print("************************Iteration {}***********************************************".format(count))
            
            self._motion_planner.to_rest_pose()

            if len(left_object_labels)==0:
                break
            
            
    def find_red_node(self, node_list):
        for node in node_list:
            if node.pose !=None and node.type == 'r' and node.target_black[0].type=='b' and node.target_black[0].occupied == False and node.pickable==True:
                if node.target_black[0].prev_node_in_stack == None:
                    # print("yes {}".format(node.name))
                    return node
                else:
                    if node.target_black[0].prev_node_in_stack.done == True:
                        return node
        return None

    def just_find_red(self, node_list):
        for node in node_list:
            if node.type == 'r' and len(node.target_black) > 0 and node.pose != None and node.pickable == True:
                return node
        return None

    # def solve(self, occ_map = []):
    #     '''Solver
    #     Somewhat DFS
    #     '''
    #     # dfs 
    #     completed_objects = []
    #     if len(occ_map) == 0:
    #         print("No occupancy map recieved, no buffer spots will be generated")
    #     sequence = []
    #     done = False
    #     nodes = self.red_nodes
    #     # head = self.find_red_node(nodes)
    #     counter = 3
    #     while (not done) and counter > 0:
    #         current_pcd = self.get_point_cloud_from_kinect()
    #         print("Total number of points in pcd: {}".format(len(current_pcd.points)))
    #         occ_map = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(current_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120)
    #         self.update_black_nodes(current_pcd, occ_map)
            
    #         head = self.find_red_node(nodes)
            
    #         if head == None:
    #             if len(occ_map) == 0:
    #                 done = True
    #                 continue
    #             head = self.just_find_red(nodes)
    #             if head == None:
    #                 done = True
    #                 continue
    #             buffer = RedBlackNode(name='{}_buffer'.format(head.name), node_type='r')
    #             buffer.occupied = copy.deepcopy(head.occupied)
    #             buffer.target_black = [head.target_black[0]]
    #             buffer.done = False
    #             # buffer.attached_red_nodes= head.attached_red_nodes
    #             head.done = True
    #             head.type = 'b'
    #             print(occ_map)
    #             target_cart_pose = np.array([head.target_black[0].pose.position.x, head.target_black[0].pose.position.y])
    #             buffer_spot_2d = self.occ_and_buffer.get_empty_spot(occ_map=occ_map, closest_target=target_cart_pose)
    #             buffer_pose = copy.deepcopy(self.object_init_pose_dict[head.object])
    #             buffer_pose.position.x = buffer_spot_2d[0]
    #             buffer_pose.position.y = buffer_spot_2d[1]
    #             buffer.pickable = False
                
    #             # Pick and place in buffer
    #             print("Generated buffer. Now, pick and place the object in buffer spot!")
    #             res = self.go_pick_object(object_name=head.object)
    #             # if res == True:
    #             self.go_place_object(object_name=head.object, final_place_pose=buffer_pose)
    #             print("Placed in buffer!")
    #             # import time
    #             # time.sleep(1)
    #             # rospy.sleep(1)
                
    #             sequence.append(head.name)
    #             sequence.append(buffer.name)
    #             buffer.prev_node_in_stack = head.prev_node_in_stack
    #             buffer.next_node_in_stack = head.next_node_in_stack
    #             if head.next_node_in_stack != None:
    #                 head.next_node_in_stack.prev_node_in_stack = buffer
    #             if head.prev_node_in_stack != None:
    #                 head.prev_node_in_stack.next_node_in_stack = buffer

    #             # head = self.find_red_node(nodes)
    #             nodes.append(buffer)
    #             # self.red_nodes.append(buffer)
    #             # nodes_labelled.append((buffer, 0))
    #             print("*************************++++++++**************************\nSequence: {}".format(sequence))
    #             # counter -= 1
    #             continue
    #         # print("Node: {}".format(head.name))
    #         # Find an undone red
    #         print("Picking and placing in target pose (since target pose is empty!)")
    #         self.go_pick_object(object_name=head.object)
    #         self.go_place_object(object_name=head.object)
    #         print("Placed in target pose!")
    #         sequence.append(head.name)
    #         sequence.append(head.target_black[0].name)
    #         head.type = 'b'
    #         head.done = True
    #         head.target_black[0].done = True
            
    #         completed_objects.append(head.object)
    #         # head = self.find_red_node(nodes)
    #         # if head == None:
    #         #     done = True
    #         print("Sequence: {}".format(sequence))
    #         # counter -= 1
            
    #     print('Sequence: {}'.format(sequence))  
    #     print("Completed objects: {}".format(completed_objects))
    #     print(" red nodes: {}".format(self.red_nodes))
    #     print("Nodes: {}".format(nodes))
    #     return completed_objects

    def go_pick_object(self, object_name):
        '''
        This function assumes that the arm is in it's rest pose. It first moves to pre-pick waypoint directly above the pick 
        pose. Then goes to pick pose and picks the object. 
        Returns:
        is_picked: boolean - If true, the object is successfully held in the arm, and we can proceed to place pose. Otherwise, go to next object.
        '''
        is_picked=False
        # 1. Go to pick pose
        pick_grasp_pose = self.object_pick_grasp_pose_dict[object_name]
        print("pick grasp pose")
        orientation_q = pick_grasp_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        print((roll, pitch, yaw))  
        if abs(yaw) > abs(np.deg2rad(90)) and abs(yaw) < abs(np.deg2rad(270)):
            yaw = yaw + np.pi
        orientation_q = quaternion_from_euler(roll, pitch, yaw)
        pick_grasp_pose.orientation = Quaternion(*orientation_q)   
        
        
        self.object_pick_grasp_pose_dict[object_name] = pick_grasp_pose 
                                                                                  
        plan_result = self._motion_planner.move_cartesian_space_upright(pick_grasp_pose, via_up=True, last_gripper_action='place')
        
        if plan_result == False:
            return is_picked
        
        # 2. Now pick the object up
        self._motion_planner.pick()
        
        return True
        
        
        
        
    def go_place_object(self, object_name, final_place_pose = None):
        '''
        This function assumes that the arm is in it's pick pose. It first moves to pre-place waypoint directly above the place 
        pose. Then drop the object 
        Returns:
        is_placed: boolean - If true, the object is successfully placed
        '''
        
        is_placed = False
        # 1. First go to the place pose
        grasp_pose = None
        if final_place_pose != None:
            grasp_pose = self.get_target_grasp_pose2(self.object_init_pose_dict[object_name], self.object_pick_grasp_pose_dict[object_name], final_place_pose)
        else:
            grasp_pose = self.object_place_grasp_pose_dict[object_name]
                # plan_result = self._motion_planner.move_cartesian_space(grasp_pose)  # move in cartesian straight path
                # plan_result = self._motion_planner.move_cartesian_space_discrete(grasp_pose)  # move in cartesian discrete path
                
        print("in place")
        print('$'*80) 
        print(" grasp pose is")
        print(grasp_pose)
                
        grasp_pose.position.z = 0.2 if self.clear_box_flag else (grasp_pose.position.z + 0.050)
        
        
        orientation_q = grasp_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        print((roll, pitch, yaw))  
        if abs(yaw) > abs(np.deg2rad(90)) and abs(yaw) < abs(np.deg2rad(270)):
            yaw = yaw + np.pi
        orientation_q = quaternion_from_euler(roll, pitch, yaw)
        grasp_pose.orientation = Quaternion(*orientation_q) 
             
        plan_result = self._motion_planner.move_cartesian_space_upright(grasp_pose, last_gripper_action='pick')
        rospy.sleep(1.0)
        
        is_placed = plan_result
        
        # 2. Place the object
        self._motion_planner.place()
        
        return is_placed

    def draw_state(self, viewer, state, colors):
        print("enter draw state function")
        viewer.clear()
        viewer.draw_environment()
        viewer.draw_robot(*state.conf[::-1])
        for block, pose in state.block_poses.items():
            r, c = pose[::-1]
            viewer.draw_block(r, c, name=block, color=colors[block])
        if state.holding is not None:
            pose = state.conf - GRASP
            r, c = pose[::-1]
            viewer.draw_block(r, c, name=state.holding, color=colors[state.holding])

    # mapping move action in 2d space to cartesian space
    def mapping_move(self, str_pose):
        cartesian_waypoints = []
        if str_pose in self._pose_mapping.keys():
            if self._last_gripper_action == 'pick':
                grasp_pose = self._pose_mapping[str_pose][self._available_grasp_pose_index[self._target_pick_object]]
                # plan_result = self._motion_planner.move_cartesian_space(grasp_pose)  # move in cartesian straight path
                # plan_result = self._motion_planner.move_cartesian_space_discrete(grasp_pose)  # move in cartesian discrete path
                
                print("in place")
                print('$'*80) 
                print(" grasp pose is")
                print(grasp_pose)
                
                grasp_pose.position.z = 0.2 if self.clear_box_flag else (grasp_pose.position.z + 0.050)
                
                plan_result = self._motion_planner.move_cartesian_space_upright(grasp_pose, last_gripper_action=self._last_gripper_action)  # move in cartesian discrete upright path
                if plan_result:
                    print('Move to the target position of object {} successfully, going to place it'.format(self._target_pick_object))
                else:
                    print('Fail to move to the target position of object: {}'.format(self._target_pick_object))
                    #user_input('To target place failed, Continue? Press Enter to continue or ctrl+c to stop')
                rospy.sleep(1.0)
            elif self._last_gripper_action == 'place':
                for index in range(self._start_grasp_index if self._start_grasp_index >= 0 else 0, self._end_grasp_index if self._end_grasp_index <= len(self._pose_mapping[str_pose]) else len(self._pose_mapping[str_pose])):
                    # plan_result = self._motion_planner.move_cartesian_space(self._pose_mapping[str_pose][index], self._pick_via_up)
                    # plan_result = self._motion_planner.move_cartesian_space_discrete(self._pose_mapping[str_pose][index], self._pick_via_up)
                    
                    pick_grasp_pose = self._pose_mapping[str_pose][index]
                    print("pick grasp pose")
                    
                    orientation_q = pick_grasp_pose.orientation
                    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                    print((roll, pitch, yaw))  
                    
                    if abs(yaw) > abs(np.deg2rad(135)) and abs(yaw) < abs(np.deg2rad(225)):
                        yaw = yaw + np.pi
                    orientation_q = quaternion_from_euler(roll, pitch, yaw)
                    
                    pick_grasp_pose.orientation = Quaternion(*orientation_q)
                    self._pose_mapping[str_pose][index] = pick_grasp_pose
                                                                                        
                    plan_result = self._motion_planner.move_cartesian_space_upright(self._pose_mapping[str_pose][index], self._pick_via_up, last_gripper_action=self._last_gripper_action)
                    if plan_result:
                        self._available_grasp_pose_index[self._target_pick_object] = index
                        rospy.loginfo('available grasp pose index: ' + str(index))
                        print('target pick object: {}'.format(self._target_pick_object))
                        if index == 0:
                            rospy.loginfo('Planning trajectory to intelligence grasp pose successfully')
                        elif index == 1:
                            rospy.loginfo('Fail to plan trajectory to intelligence grasp pose, but planning trajectory to artifical intelligence grasp pose successfully')
                        elif index == 2:
                            rospy.loginfo('Fail to plan trajectory to artificial intelligence grasp pose, but planning trajectory to artifical grasp pose successfully')
                        else:
                            pass
                        break
                    else:
                        #user_input('To pick action failed, Continue? Press Enter to continue or ctrl+c to stop')
                        continue
                else:
                    rospy.loginfo('All grasp pose tried, but fail to pick ' + str(self._target_pick_object))
        else:
            rospy.loginfo('target block pose illegal, No cartesian pose found corresponding to this block pose')
            rospy.loginfo('target block pose: ' + str_pose)

    def apply_action(self, state, action):
        print("enter apply action function")
        conf, holding, block_poses = state
        name, args = action
        if name == 'move':
            self._pick_success = True
            
            if self._last_gripper_action == 'pick':
                result = self.gripper_width_test()
                if result == True:
                    print('Gripper feedback: Gripping success')
                    self._pick_success = True
            else:
                result = True
            
            if result == True:
                rospy.loginfo('moving')
                _, conf = args
                x, y = conf
                pose = np.array([x, y])
                print('target block pose: {}'.format(str(pose)))
                print('available block pose inverse: {}'.format(self._available_block_pose_inverse))
                if str(pose) in self._available_block_pose_inverse.keys():
                    self._target_pick_object = self._available_block_pose_inverse[str(pose)]
                    rospy.loginfo('Going to pick the object called: ' + str(self._target_pick_object))
                else:
                    rospy.loginfo('Going to place the object called: ' + str(self._target_pick_object))
                self.mapping_move(str(pose))
                
            else:
                rospy.loginfo('couldnt grasp the object')  
                self._pick_success = False
                
        elif name == 'pick':
            rospy.loginfo('pick')
            holding, _, _ = args
            del block_poses[holding]
            self._motion_planner.pick()
            self._last_gripper_action = name
            print('{} is in hand now'.format(self._target_pick_object))
        elif name == 'place':
            rospy.loginfo('place')
            block, pose, _ = args
            holding = None
            block_poses[block] = pose
            print('nothing in hand')
            self._motion_planner.place()
            self._last_gripper_action = name
            
            if self._pick_success == True:
                self._completed_objects.append(block)
            # print("block", block)   
            # print("completed objects", self._completed_objects)
            # print("left over objs", )
            
        elif name == 'push':
            block, _, _, pose, conf = args
            holding = None
        else:
            raise ValueError(name)
        return DiscreteTAMPState(conf, holding, block_poses)

    
    def gripper_width_test(self):
        #check if the gripper distance is zero
        print("joint state")
        joint_state = rospy.wait_for_message("/joint_states", JointState)
        gripper_dist = [joint_state.position[0], joint_state.position[1]]
        print("gripper distance is", gripper_dist)
        # 0.002003880450502038
        # [0.0023614619858562946
        # 0.001750
        if gripper_dist[0] > 0.00100 and gripper_dist[1] > 0.00100:
            result = True #successully grabbed the object
            print(gripper_dist[0] > 0.00100, gripper_dist[1] > 0.00100 )
        else:
            result = False #failed to grab the object
        return result
    
    def apply_plan(self, tamp_problem, plan):
        print("enter apply plan function")
        state = tamp_problem.initial
        self._last_gripper_action = 'place'
        for action in plan:
            rospy.loginfo('action: ' + action.name)
            state = self.apply_action(state, action)
        self._motion_planner.to_home_pose()

    def task_planner_callback(self, data):
        print('enter listener callback function')
        self.once_plan_all()
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


if __name__ == '__main__':
    rospy.init_node('task_planner', anonymous=True)
    task_planner = TaskPlanner()
    alpha = 110 * math.pi / 180
    test_pose = Pose()
    test_pose.position.x = 0.1
    test_pose.position.y = 0.2
    test_pose.position.z = 0.3
    test_pose.orientation.x = 0
    test_pose.orientation.y = 0.25881904510252074
    test_pose.orientation.z = 0
    test_pose.orientation.w = 0.9659258262890683
    result = task_planner.get_artificial_intelligence_grasp_pose(test_pose)
    print('result: {}'.format(result))
    # while not rospy.is_shutdown():
    #     rospy.loginfo('Planner ready, waiting for trigger message!')
    #     rospy.spin()