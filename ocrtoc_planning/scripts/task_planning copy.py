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
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Imports for camera related stuff
from numpy.core.numeric import full
import cv2
import open3d as o3d
# import open3d_plus as o3dp

from ocrtoc_common.camera_interface import CameraInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

# ORTOOLS imports
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp




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
        
        self._temp_block_poses = []
        self._temp_cartesian_poses = []
        self._available_block_pose_dic = {}
        self._available_cartesian_pose_dic = {}
        self._available_grasp_pose_dic = {}
        self._goal_block_pose_dic = {}
        self._target_cartesian_pose_dic = {}
        self._target_block_pose_dic = {}
        self._pose_mapping = {}
        self._target_grasp_pose_dic = {}
        
        self.clear_box_flag = False
        
               
        self.block_labels = blocks
        self.object_goal_pose_dict = self.get_goal_pose_dict(self.block_labels, goal_cartesian_poses)
        self.block_labels_with_duplicates = self.object_goal_pose_dict.keys()
        self._goal_cartesian_pose_dic = self.object_goal_pose_dict
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
        
        self._available_cartesian_pose_dic.clear()
        self._available_grasp_pose_dic.clear()
        # self._available_grasp_pose_index.clear()
        
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
                    
                    self._available_cartesian_pose_dic[result.object_name] = result.object_pose.pose
                    self._available_grasp_pose_dic[result.object_name] = []  # pick pose list

                    # intelligence grasp strategy
                    self._available_grasp_pose_dic[result.object_name].append(result.grasp_pose.pose)  # intelligence pick pose

                    # artificial intelligence grasp strategy
                    artificial_intelligence_grasp_pose = self.get_artificial_intelligence_grasp_pose(result.grasp_pose.pose)
                    self._available_grasp_pose_dic[result.object_name].append(artificial_intelligence_grasp_pose)  # artificial intelligence pick pose

                    # artificial grasp strategy
                    artificial_grasp_pose = Pose()
                    artificial_grasp_pose.position.x = result.object_pose.pose.position.x
                    artificial_grasp_pose.position.y = result.object_pose.pose.position.y
                    artificial_grasp_pose.position.z = result.object_pose.pose.position.z + self._grasp_distance
                    artificial_grasp_pose.orientation.x = 0
                    artificial_grasp_pose.orientation.y = 1
                    artificial_grasp_pose.orientation.z = 0
                    artificial_grasp_pose.orientation.w = 0
                    self._available_grasp_pose_dic[result.object_name].append(artificial_grasp_pose)  # artificial pick pose

                    self._target_grasp_pose_dic[result.object_name] = []  # place pose list

                    # self.get_target_grasp_pose(result.object_pose.pose, result.grasp_pose.pose, self._goal_cartesian_pose_dic[result.object_name])
                    target_grasp_intelligence = \
                        self.get_target_grasp_pose2(result.object_pose.pose, result.grasp_pose.pose, self._goal_cartesian_pose_dic[result.object_name])
                    self._target_grasp_pose_dic[result.object_name].append(target_grasp_intelligence)  # intelligence place pose

                    # self.get_target_grasp_pose(result.object_pose.pose, artificial_intelligence_grasp_pose, self._goal_cartesian_pose_dic[result.object_name])
                    target_grasp_pose_artificial_intelligence = \
                        self.get_target_grasp_pose2(result.object_pose.pose, artificial_intelligence_grasp_pose, self._goal_cartesian_pose_dic[result.object_name])
                    self._target_grasp_pose_dic[result.object_name].append(target_grasp_pose_artificial_intelligence)  # artificial intelligence place pose

                    # self.get_target_grasp_pose(result.object_pose.pose, artificial_grasp_pose, self._goal_cartesian_pose_dic[result.object_name])
                    target_grasp_artificial = \
                        self.get_target_grasp_pose2(result.object_pose.pose, artificial_grasp_pose, self._goal_cartesian_pose_dic[result.object_name])
                    self._target_grasp_pose_dic[result.object_name].append(target_grasp_artificial)  # artificial place pose

                    # self._available_grasp_pose_index[result.object_name] = self._start_grasp_index if self._start_grasp_index >= 0 else 0
                    print('object name: {0}, frame id: {1}, cartesian pose: {2}'.format(result.object_name, result.object_pose.header.frame_id, result.object_pose.pose))
                    print('object name: {0}, frame id: {1}, grasp pose: {2}'.format(result.object_name, result.grasp_pose.header.frame_id, self._available_grasp_pose_dic[result.object_name]))
                                
                                      
                    
                    
                    self.object_init_pose_dict[result.object_name] = copy.deepcopy(self._available_cartesian_pose_dic[result.object_name])
                    
                    print("!!!!!!1")
                    print("self._available_grasp_pose_dic[result.object_name]" ,self._available_grasp_pose_dic[result.object_name])
                    print("result.grasp_pose.pose", result.grasp_pose.pose)
                    
                    self.object_pick_grasp_pose_dict[result.object_name] = copy.deepcopy(result.grasp_pose.pose)
                    # self.object_pick_grasp_pose_dict[result.object_name] = copy.deepcopy(self._available_grasp_pose_dic[result.object_name])
                    self.object_place_grasp_pose_dict[result.object_name] = self.get_target_grasp_pose2(result.object_pose.pose, result.grasp_pose.pose, self.object_goal_pose_dict[result.object_name])
                    
                    # self.object_place_grasp_pose_dict[result.object_name] = copy.deepcopy(self._target_grasp_pose_dic[result.object_name])
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

        logical process:
        1 get all objects goal poses
        2 get current objects poses
        # 3 compare current objects poses and goal poses
        4 plan task sequence of current relative objects
        5 task execution
        6 delete already planned objects from goal objects list and check goal objects list is empty
        7 if goal objects list is empty, stop planning, otherwise, go to step 2
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

       

    def task_planner_callback(self, data):
        print('enter listener callback function')
        self.once_plan_all()
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


if __name__ == '__main__':
    rospy.init_node('task_planner', anonymous=True)
    task_planner = TaskPlanner()
    alpha = 110 * math.pi / 180
    