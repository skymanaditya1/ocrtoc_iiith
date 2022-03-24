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
    
class OccupancyAndBuffer:
    def __init__(self):
        pass
    
    def generate_2D_occupancy_map(self, world_dat, x_min=None, y_min=None, x_range=None, y_range=None, threshold=3, dir_path='/root/occ_map.png'):
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

        print(pixel_counts)

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

# Red Black Graph (Data structure)
class RedBlackNode:
    def __init__(self, name, node_type):
        '''
        Parameters
        name: Name of the object that it is related to
        node_type: 'r' or 'b' (red implies initial/virtual object poses, black implies target or completed initial poses)
        
        Attributes:
        self.name: str: Unique identifier for each node (name+type)
        self.object: str: Name of the object associated with the node
        self.type: str: Type of node (red or black)
        self.target_black: list: Target black node for this node if this node is red
        self.done: boolean: If true, this node is done (task associated with it (pick-place) is done)
        self.occupied: boolean: If true, this node is partially occupied, otherwise, it is free
        self.prev_node_in_stack: Reference to the black node that is under the current pose in stack if the current node is black
        self.next_node_in_stack: Reference to the black node that is above the current pose in stack if the current node is black
        '''
        self.name = "{}_{}".format(name, node_type)
        self.object = name
        self.type = node_type
        # self.attached_red_nodes = []
        self.target_black = []
        self.done = False
        self.occupied = False
        self.prev_node_in_stack = None
        self.next_node_in_stack = None
        self.pose = None
        self.pick_grasp_pose = None
        self.place_grasp_pose = None
        
        self.pickable = False # Will be set to true if we get valid pick grasp and pose of the object (only for red node)
        
    def set_done(self):
        self.done = True
        self.type = 'b'
        


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
        
        
        self.block_labels = blocks
        self.object_goal_pose_dict = self.get_goal_pose_dict(self.block_labels, goal_cartesian_poses)
        self.block_labels_with_duplicates = self.object_goal_pose_dict.keys()
        
        self.object_init_pose_dict = {}
        self.object_pick_grasp_pose_dict = {}
        self.object_place_grasp_pose_dict = {}
        self.detected_object_label_list = []
        self.red_nodes = []
        self.black_nodes = []
        
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
    
    def find_target_black_node(self, pose):
        '''
        Given a pose, find the black node that is associated with it
        '''
        for node in self.black_nodes:
            if node.pose == pose:
                return node
    
    def initialize_target_black_nodes(self, object_name_list):
        '''
        Creates nodes for target poses. Includes the pose information in each of these black nodes
        
        Returns:
        A list of black nodes, each containing a target pose
        '''
        black_nodes = []
        for object_name in object_name_list:
            bnode = RedBlackNode(name=object_name, node_type='b')
            bnode.pose = self.object_goal_pose_dict[bnode.object]
            black_nodes.append(bnode)
        return black_nodes
    
    def initialize_initial_red_nodes(self, object_name_list):
        '''
        Initializes the red nodes but does not add any pose information (as it is not collected yet)
        
        Returns:
        A list of red nodes (without pose information)
        '''
        red_nodes = []
        for object_name in object_name_list:
            rnode = RedBlackNode(name=object_name, node_type='r')
            rnode.target_black = [self.find_target_black_node(self.object_goal_pose_dict[rnode.object])]
            red_nodes.append(rnode)
        return red_nodes


    def update_red_nodes(self, detected_object_label_list):
        '''
        Here, we assume that object initial and grasp poses are provided to us. We add this information to the nodes
        
        Returns:
        None
        '''
        for node in self.red_nodes:
            if node.object in detected_object_label_list:
                print("{} is in the object list*********************+++++++++++++*****************".format(node.object))
                if node.done == True or node.type == 'b':
                    continue
                node.pickable = True
                node.pose = self.object_init_pose_dict[node.object]
                node.pick_grasp_pose = self.object_pick_grasp_pose_dict[node.object]
                node.place_grasp_pose = self.object_place_grasp_pose_dict[node.object]

    def update_black_nodes(self, scene_point_cloud = [], occ_map = []):
        '''
        Currently, this function checks if a particular place is occupied or not and assigns occupancy statuses 
        to the black nodes (target pose nodes). 
        
        Stacking information: TO BE ADDED IN FUTURE (USING SCENE GRAPHS OR RELATED METHODS)
        
        Parameters:
        1. scene_point_cloud: PCD obtained from kinect/realsense
        
        Returns:
        None
        '''
        # 1. Get occupancy map
        if len(occ_map) == 0:
            print("Occupancy map building ...")
            occ_map = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(scene_point_cloud.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120)
            print("Occupancy map build!")
        else:
            print("Occupancy map recieved!")
        # 2. Get path to materials
        rospack = rospkg.RosPack()
        materials_path = rospack.get_path('ocrtoc_materials')
        print("Rospack path: {}*********************++++++++++++++++++++++++++++*************************+++++++++++++++++++++++++++".format(materials_path))
        # 3. Generate occupancy maps for each of the objects and compare the two maps. 
        #    If both the maps have entry '1' at any particular pixel, then the place is said to be occupied
        # FUTURE: ADD TOLERANCE (Concept: It is okay to have few pixels occupied, as long as it is not too many (set some threshold))
        pcds = []
        for bnode in self.black_nodes:
            if bnode.object in self.object_goal_pose_dict:
                temp = bnode.object.split('_')
                model_name = ''
                for i in range(len(temp)-1):
                    if i==0:
                        model_name = model_name + temp[i]
                    else:
                        model_name = model_name + '_' + temp[i]
                path_to_object_mesh = os.path.join(materials_path, 'models/{}/textured.obj'.format(model_name)) # visual.ply
                # Reading textured mesh and transforming it to the actual goal pose
                mesh = o3d.io.read_triangle_mesh(path_to_object_mesh)
                pose = self.object_goal_pose_dict[bnode.object]
                translation = np.array([pose.position.x, pose.position.y, pose.position.z]) - np.array(mesh.get_center())
                orientation = (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
                mesh.rotate(mesh.get_rotation_matrix_from_quaternion(orientation)) # center = True by default (implies rotation is applied w.r.t the center of the object)
                mesh.translate((translation[0], translation[1], translation[2])) 
                pcd =  mesh.sample_points_poisson_disk(1000, init_factor=5, pcl=None)
                occ1 = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120, dir_path='/root/occ_map_{model_name}.png')
                
                amount_required_occ1 = np.sum(occ1)
                # Check if occupied
                amount_occupied = np.sum(np.logical_and(occ_map, occ1))
                occ_perc = 100*float(amount_occupied)/float(amount_required_occ1)
                if occ_perc > 5:
                    bnode.occupied = True
                else:
                    bnode.occupied = False
                # is_occupied = np.any(np.logical_and(occ_map, occ1))
                # bnode.occupied = is_occupied
                print("{} percentage of {}'s target pose is occupied".format(occ_perc, bnode.object))
                print("So, occupied = {}".format(bnode.occupied))
                # pcds.append(pcd)
        # o3d.visualization.draw_geometries(pcds)
        # exit(1)
        
        
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
    
    def execute_plan(self, action_list, action_sequence_mapping):
        '''Given the list of actions in sequence, execute them one by one and return the completed objects list
        '''
        success = False
        completed_objects = []
        for action in action_list:
            name = action_sequence_mapping[str(action)]['name']
            object_name = action_sequence_mapping[str(action)]['object']
            if self.search_strings([name], searchable='pick'):
                print("Going to pick {}\t|\t".format(object_name))
                self._motion_planner.fake_place()
                success = self.go_pick_object(object_name=object_name)
                 # 3. Check if the object is grapsed
                # success = self.gripper_width_test()
                                     
                if success == False:
                    print("Pick failed")
                else:
                    print("Pick success")
            elif self.search_strings([name], searchable='place') and success==True:
                print("Going to place {}\t|\t".format(object_name))
                success = self.go_place_object(object_name)
                if success == False:
                    print("Place failed")
                else:
                    print("Place success")
                    completed_objects.append(object_name)
        return completed_objects
            
        
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
             
                   
        # 1. Create black nodes for target poses
        # self.black_nodes = self.initialize_target_black_nodes(self.block_labels_with_duplicates)
        # 2. Create red nodes for initial poses
        # self.red_nodes = self.initialize_initial_red_nodes(self.block_labels_with_duplicates)
            
        count = 0
        label_count = 1
        while len(left_object_labels) > 0 and count <= 5:
            count += 1
            # 3. Get information about left objects from perception node
            
            self.detected_object_label_list = []
            
            rospy.loginfo('Try to get information of left objects from perception node')
            self.get_pose_perception(left_object_labels)
            print("Detected object list: {}".format(self.detected_object_label_list))

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
            
            # 9. Remove completed objects
            temp = []
            for object in left_object_labels:
                if object in completed_objects:
                    continue
                temp.append(object)
            left_object_labels = temp
            print("left objects: {}".format(left_object_labels))

            print("************************Iteration {}***********************************************".format(count))
            
            self._motion_planner.to_rest_pose()
            
            
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

    def solve(self, occ_map = []):
        '''Solver
        Somewhat DFS
        '''
        # dfs 
        completed_objects = []
        if len(occ_map) == 0:
            print("No occupancy map recieved, no buffer spots will be generated")
        sequence = []
        done = False
        nodes = self.red_nodes
        # head = self.find_red_node(nodes)
        counter = 3
        while (not done) and counter > 0:
            current_pcd = self.get_point_cloud_from_kinect()
            print("Total number of points in pcd: {}".format(len(current_pcd.points)))
            occ_map = self.occ_and_buffer.generate_2D_occupancy_map(np.asarray(current_pcd.points)*100, x_min=-30, y_min=-60, x_range=60, y_range=120)
            self.update_black_nodes(current_pcd, occ_map)
            
            head = self.find_red_node(nodes)
            
            if head == None:
                if len(occ_map) == 0:
                    done = True
                    continue
                head = self.just_find_red(nodes)
                if head == None:
                    done = True
                    continue
                buffer = RedBlackNode(name='{}_buffer'.format(head.name), node_type='r')
                buffer.occupied = copy.deepcopy(head.occupied)
                buffer.target_black = [head.target_black[0]]
                buffer.done = False
                # buffer.attached_red_nodes= head.attached_red_nodes
                head.done = True
                head.type = 'b'
                print(occ_map)
                target_cart_pose = np.array([head.target_black[0].pose.position.x, head.target_black[0].pose.position.y])
                buffer_spot_2d = self.occ_and_buffer.get_empty_spot(occ_map=occ_map, closest_target=target_cart_pose)
                buffer_pose = copy.deepcopy(self.object_init_pose_dict[head.object])
                buffer_pose.position.x = buffer_spot_2d[0]
                buffer_pose.position.y = buffer_spot_2d[1]
                buffer.pickable = False
                
                # Pick and place in buffer
                print("Generated buffer. Now, pick and place the object in buffer spot!")
                res = self.go_pick_object(object_name=head.object)
                # if res == True:
                self.go_place_object(object_name=head.object, final_place_pose=buffer_pose)
                print("Placed in buffer!")
                # import time
                # time.sleep(1)
                # rospy.sleep(1)
                
                sequence.append(head.name)
                sequence.append(buffer.name)
                buffer.prev_node_in_stack = head.prev_node_in_stack
                buffer.next_node_in_stack = head.next_node_in_stack
                if head.next_node_in_stack != None:
                    head.next_node_in_stack.prev_node_in_stack = buffer
                if head.prev_node_in_stack != None:
                    head.prev_node_in_stack.next_node_in_stack = buffer

                # head = self.find_red_node(nodes)
                nodes.append(buffer)
                # self.red_nodes.append(buffer)
                # nodes_labelled.append((buffer, 0))
                print("*************************++++++++**************************\nSequence: {}".format(sequence))
                # counter -= 1
                continue
            # print("Node: {}".format(head.name))
            # Find an undone red
            print("Picking and placing in target pose (since target pose is empty!)")
            self.go_pick_object(object_name=head.object)
            self.go_place_object(object_name=head.object)
            print("Placed in target pose!")
            sequence.append(head.name)
            sequence.append(head.target_black[0].name)
            head.type = 'b'
            head.done = True
            head.target_black[0].done = True
            
            completed_objects.append(head.object)
            # head = self.find_red_node(nodes)
            # if head == None:
            #     done = True
            print("Sequence: {}".format(sequence))
            # counter -= 1
            
        print('Sequence: {}'.format(sequence))  
        print("Completed objects: {}".format(completed_objects))
        print(" red nodes: {}".format(self.red_nodes))
        print("Nodes: {}".format(nodes))
        return completed_objects

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
        if gripper_dist[0] > 0.005 and gripper_dist[1] > 0.005:
            result = True #successully grabbed the object
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
