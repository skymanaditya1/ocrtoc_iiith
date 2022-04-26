# license
import copy
from math import pi
from tokenize import group
import numpy as np
import os
import sys
import yaml
import tf

import actionlib
import control_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Transform
import moveit_commander
import moveit_msgs.msg
import rospkg
import rospy
import time
from sensor_msgs.msg import JointState

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from ocrtoc_common.transform_interface import TransformInterface
from ocrtoc_common.camera_interface import CameraInterface

import numpy as np
import transforms3d
# for grasp refinement
import PIL.Image as PIL_image
from time import sleep
import cv2
import open3d as o3d

# import open3d_plus as o3dp


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
        self.max_penetration_len = 8 #4 # Max amount of penetration of the object between the two fingers
    def get_gripper_standard_collision_map_1cm_resolution(self, inter_finger_dist = 8):
        '''Gets the gripper collision map that can be used to evaluate collisions at different points. 
        The produced map is for the gripper in its canonical/standard pose (it's length parellel to 
        the length of the table)
        The resolution of the map -> 1 pixel occupies 1cm*1cm square box in the real world

        Parameters:
        inter_finger_dist: distance between the two fingers in cm (default - max finger distance - 8 cm)
        '''
        collision_map = np.full(shape=(self.ee_length, self.ee_length), fill_value=10) 
        for i in range(int(self.ee_length//2 - self.ee_width//2), int(self.ee_length//2 + self.ee_width//2)):
            collision_map[i:i+1] = 0
            # print("ee - {}".format(i))
            if i == (self.ee_length//2 - 1) or i == self.ee_length//2:
                f_start = int((self.ee_length-1)//2 - (inter_finger_dist/2)) # 2
                f_end = int((self.ee_length)//2 + (inter_finger_dist/2)) # 11
                collision_map[i, f_start] = -1*self.finger_height # 2 # self.max_penetration_len # or use self.finger_height, if full penetration is allowed
                collision_map[i, f_end] = -1*self.finger_height # 14 # self.max_penetration_len
                collision_map[i, (f_start+1):f_end] = -1*self.max_penetration_len # (not allowing any object to penetrate more than 4 cm towards the EE, between the fingers)
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

    def get_grip_validation_map_1cm_res(self, inter_finger_dist=8):
        '''Generates grip validation map with 1cm resolution (1 pixel == 1cm*1cm area)
        The produced map is for the gripper in its canonical/standard pose (it's length parellel to 
        the length of the table)
        Used for grasp validation 
        Size = 14*14 (general)

        Parameters:
        inter_finger_dist = distance between the fingers in cm (default: max dist = 8cm)
        '''
        v_map = np.full(shape=(self.ee_length, self.ee_length), fill_value=10) 
        for i in range(int(self.ee_length//2 - self.ee_width//2), int(self.ee_length//2 + self.ee_width//2)):
            # print("ee - {}".format(i))
            if i == (self.ee_length//2 - 1) or i == self.ee_length//2:
                f_start = int((self.ee_length-1)//2 - (inter_finger_dist/2)) # 2
                f_end = int((self.ee_length)//2 + (inter_finger_dist/2)) # 11
                print("fstart: {}|2     f_end: {}|11".format(f_start, f_end))
                v_map[i:i+1, (f_start+1):f_end] = 0
            
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
        k_img = PIL_image.fromarray(np.uint8(kernel_map+self.finger_height)) # Converts all negative elements to +ve (will be reversed later)
        r_img = k_img.rotate(angle=angle, fillcolor=10+self.finger_height) # Angle in degrees
        rk_map = np.array(r_img, dtype=float) 
        print("Rotated kernel: {}".format(rk_map-self.finger_height))
        
        # cv2.imshow("Kernel image", np.uint8((kernel_map)*15))
        # cv2.waitKey(0)

        # cv2.imshow("Rotated image", np.uint8((rk_map-self.finger_height)*15))
        # cv2.waitKey(0)
        cv2.imwrite("/root/Rotated gripper.png", np.uint8((rk_map-self.finger_height)*15))
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
        k_img = PIL_image.fromarray(np.uint8(kernel_map+self.finger_height)) # Converts all negative elements to +ve (will be reversed later)
        r_img = k_img.rotate(angle=angle, fillcolor=10+self.finger_height) # Angle in degrees
        rk_map = np.array(r_img, dtype=float) 
        print("Rotated Valid: {}".format(rk_map-self.finger_height))
        
        # cv2.imshow("Valid image", np.uint8((kernel_map)*15))
        # cv2.waitKey(0)

        # cv2.imshow("Rotated Valid image", np.uint8((rk_map-self.finger_height)*15))
        # cv2.waitKey(0)
        cv2.imwrite("/root/Rotated valid map.png", np.uint8((rk_map-self.finger_height)*15))
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

        valid_pts = []

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
                        valid_pts.append([i+g_nr/2, j+g_nc/2])
    
        return collision_map, valid_map, valid_pts

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
        
        # cv2.imshow("Scene hmap", np.uint8(pixel_maxes*255/np.max(pixel_maxes)))
        # cv2.waitKey(0)
        cv2.imwrite("/root/scene hmap.png", np.uint8(pixel_maxes*255/np.max(pixel_maxes)))

        return pixel_maxes

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

class MotionPlanner(object):
    def __init__(self):
        # load parameters
        rospack = rospkg.RosPack()
        motion_config_path = os.path.join(rospack.get_path('ocrtoc_planning'), 'config/motion_planner_parameter.yaml')
        with open(motion_config_path, "r") as f:
            config_parameters = yaml.load(f)
            self._max_try = config_parameters["max_try"]
            self._up_distance = config_parameters["up_distance"]
            self._exit_distance = config_parameters["exit_distance"]
            self._entrance_distance = config_parameters["entrance_distance"]
            self._up1 = config_parameters["up1"]
            self._up2 = config_parameters["up2"]
            self._max_attempts = config_parameters["max_attempts"]
            self._plan_step_length = config_parameters["plan_step_length"]

            self._via_pose = Pose()
            self._via_pose.position.x = config_parameters["via_pose1_px"]
            self._via_pose.position.y = config_parameters["via_pose1_py"]
            self._via_pose.position.z = config_parameters["via_pose1_pz"]
            self._via_pose.orientation.x = config_parameters["via_pose_ox"]
            self._via_pose.orientation.y = config_parameters["via_pose_oy"]
            self._via_pose.orientation.z = config_parameters["via_pose_oz"]
            self._via_pose.orientation.w = config_parameters["via_pose_ow"]

            self._via_pose2 = Pose()
            self._via_pose2.position.x = config_parameters["via_pose2_px"]
            self._via_pose2.position.y = config_parameters["via_pose2_py"]
            self._via_pose2.position.z = config_parameters["via_pose2_pz"]
            self._via_pose2.orientation.x = config_parameters["via_pose_ox"]
            self._via_pose2.orientation.y = config_parameters["via_pose_oy"]
            self._via_pose2.orientation.z = config_parameters["via_pose_oz"]
            self._via_pose2.orientation.w = config_parameters["via_pose_ow"]

            self._via_pose3 = Pose()
            self._via_pose3.position.x = config_parameters["via_pose3_px"]
            self._via_pose3.position.y = config_parameters["via_pose3_pz"]
            self._via_pose3.position.z = config_parameters["via_pose3_pz"]
            self._via_pose3.orientation.x = config_parameters["via_pose_ox"]
            self._via_pose3.orientation.y = config_parameters["via_pose_oy"]
            self._via_pose3.orientation.z = config_parameters["via_pose_oz"]
            self._via_pose3.orientation.w = config_parameters["via_pose_ow"]

            self._home_joints = [config_parameters["home_joint1"] * pi / 180,
                                 config_parameters["home_joint2"] * pi / 180,
                                 config_parameters["home_joint3"] * pi / 180,
                                 config_parameters["home_joint4"] * pi / 180,
                                 config_parameters["home_joint5"] * pi / 180,
                                 config_parameters["home_joint6"] * pi / 180,
                                 config_parameters["home_joint7"] * pi / 180]

            self._group_name = config_parameters["group_name"]

        moveit_commander.roscpp_initialize(sys.argv)
        self._move_group = moveit_commander.MoveGroupCommander(self._group_name)
        self._move_group.allow_replanning(True)
        self._move_group.set_start_state_to_current_state()
        self._end_effector = self._move_group.get_end_effector_link()
        self._at_home_pose = False
        self._gripper_client = actionlib.SimpleActionClient('/franka_gripper/gripper_action', control_msgs.msg.GripperCommandAction)
        self._transformer = TransformInterface()
        entrance_transformation = Transform()
        entrance_transformation.translation.x = 0
        entrance_transformation.translation.y = 0
        entrance_transformation.translation.z = self._entrance_distance
        entrance_transformation.rotation.x = 0
        entrance_transformation.rotation.y = 0
        entrance_transformation.rotation.z = 0
        entrance_transformation.rotation.w = 1
        self._entrance_transformation_matrix = self._transformer.ros_transform_to_matrix4x4(entrance_transformation)

        ee_transform = self._transformer.lookup_ros_transform("panda_ee_link", "panda_link8")
        self._ee_transform_matrix = self._transformer.ros_transform_to_matrix4x4(ee_transform.transform)

        # For grasp refinement
        self.grasp_refine_class = GraspModification()

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

        self.o3dp = MiscFunctions()
        
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

        self.to_home_pose()
        # self.test()
        self.place()
        rospy.sleep(1.0)

    def test(self):
        
        print("test 1, joint space to a goal")
        pose_goal = Pose()
        # pose_goal.orientation.w = 1.0
        # pose_goal.position.x = 0.1
        # pose_goal.position.y = 0.3
        # pose_goal.position.z = 0.2
        # self.move_joint_space(pose_goal)
        
        
        print("test 2 joint space to rest")
        
        self.to_rest_pose()
        
        print("test 3 cartesian to goal")
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.1
        pose_goal.position.y = -0.3
        pose_goal.position.z = 0.2
        # self.move_joint_space(pose_goal)
        self.move_cartesian_space_upright(pose_goal)
        
        print("test 4 joint to goal")
        
        self.to_rest_pose()
        
        print("test 5 cartesian to goal")
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.1
        pose_goal.position.y = 0.3
        pose_goal.position.z = 0.2
        self.move_cartesian_space_upright(pose_goal)
        
        print("test 6 joint to goal")
        
        self.to_rest_pose()
        
    # move to specified home pose
    def to_home_pose(self):
        self._move_group.set_joint_value_target(self._home_joints)
        to_home_result = self._move_group.go()
        rospy.sleep(1.0)
        print('to home pose result:{}'.format(to_home_result))
        return to_home_result

    # generate path in joint space
    def move_joint_space(self, pose_goal):
        self._move_group.set_pose_target(pose_goal)
        plan = self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()
        rospy.sleep(1.0)

    # move to specified home pose
    def to_rest_pose(self):

        rest_pose = Pose()
        rest_pose.position.x = -0.112957249941
        rest_pose.position.y = 2.9801544038e-05
        rest_pose.position.z = 0.590340135745
        rest_pose.orientation.x = -0.923949504923
        rest_pose.orientation.y = 0.382514458771
        rest_pose.orientation.z = -3.05585242637e-05
        rest_pose.orientation.w = 1.57706453844e-05
        
        fraction = 0
        attempts = 0
        waypoints = []
        group_goal = self.ee_goal_to_link8_goal(rest_pose)
        print("group goal after tf", group_goal)
        group_goal = rest_pose
        
        waypoints.append(copy.deepcopy(group_goal))
        while fraction < 1.0 and attempts < self._max_attempts:
            (plan, fraction) = self._move_group.compute_cartesian_path(
                waypoints,                # way points
                self._plan_step_length,          # step length
                0.0,                             # disable jump
                True                             # enable avoid_collision
                )
            attempts += 1

        # if fraction == 1.0:
        rospy.loginfo('Path computed successfully, moving robot')
        self._move_group.execute(plan)
        self._move_group.stop()
        self._move_group.clear_pose_targets()
        rospy.loginfo('Path execution completed')
        move_result = True
        
        self._rest_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        # current_pose = self._move_group.get_current_pose(self._end_effector).pose
        # print("rest pose x,y,z: ", current_pose)
        self._move_group.set_joint_value_target(self._rest_joints)
        to_rest_result = self._move_group.go()
        # rospy.sleep(1.0)
        print('to rest pose result:{}'.format(to_rest_result))
        
        # time.sleep(10)
        current_pose = self._move_group.get_current_pose(self._end_effector).pose
        print("rest pose x,y,z: ", current_pose)
        return to_rest_result
    
    # move robot to home pose, then move from home pose to target pose
    def move_from_home_pose(self, pose_goal):
        from_home_result = True
        to_home_result = self.to_home_pose()

        if to_home_result:
            points_to_target = self.get_points_to_target(pose_goal)
            fraction = 0
            attempts = 0
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    points_to_target,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                self._move_group.execute(plan)
                from_home_result = True
            else:
                from_home_result = False
        else:
            from_home_result = False

        rospy.loginfo('move from home pose result' + str(from_home_result))
        return from_home_result

    # move robot to home pose, then move from home pose to target pose
    def move_from_home_pose(self, pose_goal):
        from_home_result = True
        to_home_result = self.to_home_pose()

        if to_home_result:
            points_to_target = self.get_points_to_target(pose_goal)
            fraction = 0
            attempts = 0
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    points_to_target,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                self._move_group.execute(plan)
                from_home_result = True
            else:
                from_home_result = False
        else:
            from_home_result = False

        rospy.loginfo('move from home pose result' + str(from_home_result))
        return from_home_result

    def move_home_to_target_via_entry(self, pose_goal, via_up=True, last_gripper_action='pick'):
        '''This functions moves the arm from home pose to pick/place pose via an entry waypoint
        - Usually executed prior to pick/place operation
        '''
        # Pose goal correction
        if pose_goal.position.z < 0.005:
            pose_goal.position.z = 0.01
        quaternion = [pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        print("Roll: {}, Pitch: {}, Yaw: {}".format(roll, pitch, yaw))
        quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, yaw)
        pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w = quaternion
        group_goal = self.ee_goal_to_link8_goal(pose_goal)
        target_pose = copy.deepcopy(group_goal)

        # Create an entry waypoint
        points_to_target = []
        enter_pose = copy.deepcopy(target_pose)
        enter_pose.position.z += self._up2

        points_to_target.append(copy.deepcopy(enter_pose))
        points_to_target.append(copy.deepcopy(target_pose))

        # Move
        # Move the arm to exit and then to rest pose
        for i, point in enumerate(points_to_target):
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('(upright path) Action finished, action result' + str(move_result))
        return move_result

    def move_current_to_home_via_exit(self, via_up=True, last_gripper_action='pick'):
        '''This functions moves the arm from its current (pick or place) position to home pose 
        via an exit waypoint directly above the pick pose 
        - Usually executed after pick operation or after a failed pick operation
        '''
        # create exit waypoint
        points_to_target = []
        current_pose = self._move_group.get_current_pose(self._end_effector).pose
        exit_pose = copy.deepcopy(current_pose)
        exit_pose.position.z += self._up1
        points_to_target.append(copy.deepcopy(exit_pose))

        # Move the arm to exit and then to rest pose
        for i, point in enumerate(points_to_target):
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break

            # Finally move to rest pose from exit point
            self.to_rest_pose()
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('(upright path) Action finished, action result' + str(move_result))
        return move_result
    
    
    def gripper_width_test(self):
        #check if the gripper distance is zero
        print("joint state")
        joint_state = rospy.wait_for_message("/joint_states", JointState)
        gripper_dist = [joint_state.position[0], joint_state.position[1]]
        print("gripper distance is", gripper_dist)
        # 0.0038156178900761884, 0.0036195879904434205]
        if gripper_dist[0] > 0.00100 and gripper_dist[1] > 0.00100:
            result = True #successully grabbed the object
        else:
            result = False #failed to grab the object
        return result
    
    
    # generate cartesian straight path
    def move_cartesian_space_upright(self, pose_goal, via_up = False, last_gripper_action='pick'):
        # get a list of way points to target pose, including entrance pose, transformation needed
        # transform panda_ee_link goal to panda_link8 goal.
        
        if pose_goal.position.z < 0.005:
            pose_goal.position.z = 0.01
        
        quaternion = [pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w]
        
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        
        print("Roll: {}, Pitch: {}, Yaw: {}".format(roll, pitch, yaw))
        
        quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, yaw)
        
        pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w = quaternion
        
        group_goal = self.ee_goal_to_link8_goal(pose_goal)
        # group_goal = pose_goal

        points_to_target = self.get_points_to_target_upright(group_goal)
        for i, point in enumerate(points_to_target):
        
            if i==1 and last_gripper_action=='place':
                self.to_rest_pose()
            
            if i==1 and last_gripper_action=='pick':
                self.to_rest_pose()
                success = self.gripper_width_test()
                if success == False:
                    return False

            if i==2 and last_gripper_action=='place': # At pre-pick pose, with the next pose being the pick pose
                pcd = self.get_point_cloud_from_realsense(debug=True)
                scene_hmap = self.grasp_refine_class.pcd_to_height_map_1cm_res(np.asarray(pcd.points), target_pos=[point.position.x, point.position.y, point.position.z], x_range=0.16, y_range=0.16)
                pose_found = False
                for i in range(2, 9, 2):
                    # Check if the current yaw is valid or not
                    gripper_map = self.grasp_refine_class.get_gripper_standard_collision_map_1cm_resolution(inter_finger_dist=i)
                    r_gmap = self.grasp_refine_class.rotate_kernel_map(gripper_map, np.rad2deg(yaw))
                    v_map = self.grasp_refine_class.get_grip_validation_map_1cm_res()
                    r_vmap = self.grasp_refine_class.rotate_v_map(v_map, np.rad2deg(yaw))
                    print("Grasp height: {}".format(point.position.z*100))
                    collision_map, valid_map, valid_pts = self.grasp_refine_class.return_collision_and_valid_maps(scene_hmap=scene_hmap, gripper_map=r_gmap, v_map=r_vmap, grasp_height=point.position.z*100) # grasp height in cm

                    # cv2.imshow("Collision_map", np.uint8(collision_map*255))
                    # cv2.waitKey(0)
                    cv2.imwrite("/root/collisionmap.png", np.uint8(collision_map*255))
                    print("Final collision map: {}", format(collision_map))
                    print("Max valid: {}".format(np.max(valid_map)))
                    print("Validation map final : {}".format(valid_map))
                    # cv2.imshow("Valid_map", np.uint8((valid_map/max(0.001, np.max(valid_map)))*255))
                    # cv2.waitKey(0)
                    cv2.imwrite("/root/validmap.png", np.uint8((valid_map/max(0.001, np.max(valid_map)))*255))

                    if len(valid_pts)>0:
                        print("Valid grasp pose found: {}".format(valid_pts))
                        pose_found = True
                    else:
                        print("No valid grasp pose")
                
                    if pose_found==True:
                        break
                print("POse found: {}".format(pose_found))
                
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('(upright path) Action finished, action result' + str(move_result))
        return move_result
    

    # generate cartesian straight path
    def move_cartesian_space(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        fraction = 0
        attempts = 0
        while fraction < 1.0 and attempts < self._max_attempts:
            (plan, fraction) = self._move_group.compute_cartesian_path(
                points_to_target,                # way points
                self._plan_step_length,          # step length
                0.0,                             # disable jump
                True                             # enable avoid_collision
                )
            attempts += 1

        if fraction == 1.0:
            rospy.loginfo('Path computed successfully, moving robot')
            self._move_group.execute(plan)
            self._move_group.stop()
            self._move_group.clear_pose_targets()
            rospy.loginfo('Path execution completed')
            move_result = True
        else:
            rospy.loginfo('Fisrt path planning failed, try to planing 2nd path')
            move_result = self.move_cartesian_space2(pose_goal, via_up)
            if move_result:
                pass
            else:
                move_result = self.move_cartesian_space3(pose_goal, via_up)

        rospy.loginfo('move finished, move result' + str(move_result))
        return move_result

    # generate cartesian straight path2
    def move_cartesian_space2(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target2(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        fraction = 0
        attempts = 0
        while fraction < 1.0 and attempts < self._max_attempts:
            (plan, fraction) = self._move_group.compute_cartesian_path(
                points_to_target,                # way points
                self._plan_step_length,          # step length
                0.0,                             # disable jump
                True                             # enable avoid_collision
                )
            attempts += 1

        if fraction == 1.0:
            rospy.loginfo('Path2 computed successfully, moving robot')
            self._move_group.execute(plan)
            self._move_group.stop()
            self._move_group.clear_pose_targets()
            rospy.loginfo('Path2 execution completed')
            move_result = True
        else:
            rospy.loginfo('2nd path planning failed, try to planing the 3rd path')
            move_result = False
        return move_result

    # generate cartesian straight path3
    def move_cartesian_space3(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target3(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        fraction = 0
        attempts = 0
        while fraction < 1.0 and attempts < self._max_attempts:
            (plan, fraction) = self._move_group.compute_cartesian_path(
                points_to_target,                # way points
                self._plan_step_length,          # step length
                0.0,                             # disable jump
                True                             # enable avoid_collision
                )
            attempts += 1

        if fraction == 1.0:
            rospy.loginfo('Path3 computed successfully, moving robot')
            self._move_group.execute(plan)
            self._move_group.stop()
            self._move_group.clear_pose_targets()
            rospy.loginfo('Path3 execution completed')
            move_result = True
        else:
            rospy.loginfo('All path tried, but fail to find an available path!')
            rospy.loginfo('All path tried, but fail to find an available path!')
            rospy.loginfo('All path tried, but fail to find an available path!')
            move_result = False
        return move_result

    # generate cartesian straight path
    def move_cartesian_space_discrete(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        for point in points_to_target:
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('(discrete path) Action finished, action result' + str(move_result))
        return move_result

    # generate cartesian straight path2
    def move_cartesian_space2_discrete(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target2(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        for point in points_to_target:
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('Action finished, action result' + str(move_result))
        return move_result

    # generate cartesian straight path3
    def move_cartesian_space3_discrete(self, pose_goal, via_up=False):
        points_to_target = self.get_points_to_target3(pose_goal, via_up)  # get a list of way points to target pose, including entrance pose, transformation needed
        for point in points_to_target:
            fraction = 0
            attempts = 0
            waypoints = []
            waypoints.append(copy.deepcopy(point))
            while fraction < 1.0 and attempts < self._max_attempts:
                (plan, fraction) = self._move_group.compute_cartesian_path(
                    waypoints,                # way points
                    self._plan_step_length,          # step length
                    0.0,                             # disable jump
                    True                             # enable avoid_collision
                    )
                attempts += 1

            if fraction == 1.0:
                rospy.loginfo('Path computed successfully, moving robot')
                self._move_group.execute(plan)
                self._move_group.stop()
                self._move_group.clear_pose_targets()
                rospy.loginfo('Path execution completed')
                move_result = True
            else:
                rospy.loginfo('Action failed')
                move_result = False
                break
        else:
            move_result = True
            rospy.loginfo('Action success')

        rospy.loginfo('Action finished, action result' + str(move_result))
        return move_result

    # pick action
    def pick(self):
        self._gripper_client.wait_for_server()
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = -0.1
        goal.command.max_effort = 30

        self._gripper_client.send_goal(goal)
        rospy.sleep(2.0)

    # place action
    def place(self):
        self._gripper_client.wait_for_server()
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = 0.045 #0.039
        goal.command.max_effort = 30

        self._gripper_client.send_goal(goal)
        rospy.sleep(2.0)
    
    
    #A dummy for a bug. Check it later
    def fake_place(self):
        self._gripper_client.wait_for_server()
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = 0.045 #0.039
        goal.command.max_effort = 30

        self._gripper_client.send_goal(goal)
        rospy.sleep(0.5)

    # get a list of via points from current position to target position (add exit and entrance point to pick and place position)
    def get_points_to_target_upright(self, target_pose):
        points_to_target = []
        current_pose = self._move_group.get_current_pose(self._end_effector).pose
        exit_pose = copy.deepcopy(current_pose)
        exit_pose.position.z += self._up1
        points_to_target.append(copy.deepcopy(exit_pose))

        enter_pose = copy.deepcopy(target_pose)
        enter_pose.position.z += self._up2

        points_to_target.append(copy.deepcopy(enter_pose))
        points_to_target.append(copy.deepcopy(target_pose))
        return points_to_target

    # get a list of via points from current position to target position (add exit and entrance point to pick and place position)
    def get_points_to_target(self, target_pose, via_up = False):
        points_to_target = []
        current_pose = self._move_group.get_current_pose(self._end_effector).pose
        current_pose_matrix = self._transformer.ros_pose_to_matrix4x4(current_pose)
        exit_pose_matrix = current_pose_matrix.dot(self._entrance_transformation_matrix)
        exit_pose = self._transformer.matrix4x4_to_ros_pose(exit_pose_matrix)
        points_to_target.append(copy.deepcopy(exit_pose))

        points_to_target.append(copy.deepcopy(self._via_pose))

        target_pose_matrix = self._transformer.ros_pose_to_matrix4x4(target_pose)
        enter_pose_matrix = target_pose_matrix.dot(self._entrance_transformation_matrix)
        enter_pose = self._transformer.matrix4x4_to_ros_pose(enter_pose_matrix)

        if via_up:
            up_pose = Pose()
            up_pose.position.x = enter_pose.position.x
            up_pose.position.y = enter_pose.position.y
            up_pose.position.z = enter_pose.position.z + self._up_distance
            up_pose.orientation.x = enter_pose.orientation.x
            up_pose.orientation.y = enter_pose.orientation.y
            up_pose.orientation.z = enter_pose.orientation.z
            up_pose.orientation.w = enter_pose.orientation.w
            points_to_target.append(copy.deepcopy(up_pose))
        else:
            pass

        points_to_target.append(copy.deepcopy(enter_pose))
        points_to_target.append(copy.deepcopy(target_pose))
        return points_to_target

    # get a list of via points from current position to target position (via_pose2)
    def get_points_to_target2(self, target_pose, via_up=False):
        points_to_target = []
        exit_pose = self._move_group.get_current_pose(self._end_effector).pose
        exit_pose.position.z = exit_pose.position.z + self._exit_distance
        points_to_target.append(copy.deepcopy(exit_pose))

        points_to_target.append(copy.deepcopy(self._via_pose2))

        target_pose_matrix = self._transformer.ros_pose_to_matrix4x4(target_pose)
        enter_pose_matrix = target_pose_matrix.dot(self._entrance_transformation_matrix)
        enter_pose = self._transformer.matrix4x4_to_ros_pose(enter_pose_matrix)

        if via_up:
            up_pose = Pose()
            up_pose.position.x = enter_pose.position.x
            up_pose.position.y = enter_pose.position.y
            up_pose.position.z = enter_pose.position.z + self._up_distance
            up_pose.orientation.x = enter_pose.orientation.x
            up_pose.orientation.y = enter_pose.orientation.y
            up_pose.orientation.z = enter_pose.orientation.z
            up_pose.orientation.w = enter_pose.orientation.w
            points_to_target.append(copy.deepcopy(up_pose))
        else:
            pass

        points_to_target.append(copy.deepcopy(enter_pose))
        points_to_target.append(copy.deepcopy(target_pose))
        return points_to_target

    # get a list of via points from current position to target position (via_pose3)
    def get_points_to_target3(self, target_pose, via_up=False):
        points_to_target = []
        exit_pose = self._move_group.get_current_pose(self._end_effector).pose
        exit_pose.position.z = exit_pose.position.z + self._exit_distance
        points_to_target.append(copy.deepcopy(exit_pose))

        points_to_target.append(copy.deepcopy(self._via_pose3))

        target_pose_matrix = self._transformer.ros_pose_to_matrix4x4(target_pose)
        enter_pose_matrix = target_pose_matrix.dot(self._entrance_transformation_matrix)
        enter_pose = self._transformer.matrix4x4_to_ros_pose(enter_pose_matrix)

        if via_up:
            up_pose = Pose()
            up_pose.position.x = enter_pose.position.x
            up_pose.position.y = enter_pose.position.y
            up_pose.position.z = enter_pose.position.z + self._up_distance
            up_pose.orientation.x = enter_pose.orientation.x
            up_pose.orientation.y = enter_pose.orientation.y
            up_pose.orientation.z = enter_pose.orientation.z
            up_pose.orientation.w = enter_pose.orientation.w
            points_to_target.append(copy.deepcopy(up_pose))
        else:
            pass

        points_to_target.append(copy.deepcopy(enter_pose))
        points_to_target.append(copy.deepcopy(target_pose))
        return points_to_target

    def ee_goal_to_link8_goal(self, ee_goal):
        ee_goal_matrix = self._transformer.ros_pose_to_matrix4x4(ee_goal)
        link8_goal_matrix = ee_goal_matrix.dot(self._ee_transform_matrix)
        link8_goal = self._transformer.matrix4x4_to_ros_pose(link8_goal_matrix)
        return link8_goal

    def get_color_image(self): # newly added
        return self.camera_interface.get_numpy_image_with_encoding(self.color_topic_name)[0]

    def get_color_camK(self): # newly added
        d = self.camera_interface.get_dict_camera_info(self.color_info_topic_name)
        return (np.array(d['K']).reshape((3,3)))

    def get_depth_image(self): # newly added
        return self.camera_interface.get_numpy_image_with_encoding(self.depth_topic_name)[0]

    def get_pcd(self, use_graspnet_camera_frame = False): # newly added
        pcd = self.camera_interface.get_o3d_pcd(self.points_topic_name)
        return pcd

    def get_color_transform_matrix(self): # newly added 
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.color_transform_to_frame)

    def get_points_transform_matrix(self): # newly added
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.points_transform_to_frame)
        
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
    
    def capture_pcd(self, use_camera='kinect', debug=False):
        t1 = time.time()
        pcd = None
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
            # return full_pcd_kinect
            pcd = full_pcd_kinect
        elif use_camera == 'realsense':
            print("Using real sense to get the image;( ******++++++++++++***********")
            color_image = self.get_color_image()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
            if debug:
                cv2.imshow('color', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            color_images.append(color_image)

            points_trans_matrix = self.get_points_transform_matrix()
            if debug:
                print('points_trans_matrix:', points_trans_matrix)
            camera_poses.append(self.get_color_transform_matrix())
            pcd = self.get_pcd(use_graspnet_camera_frame = False)
            pcd.transform(points_trans_matrix)
            pcd = self.kinect_process_pcd(pcd, self.reconstruction_config)
            if debug:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([pcd, frame])

            pcd = pcd        
            # return pcd
            # pcds.append(pcd)
        pcd.estimate_normals()
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors = self.reconstruction_config['nb_neighbors'], # self.nb_neighbors_config=10 # ,
            std_ratio = self.reconstruction_config['std_ratio'] # self.std_ratio_config = 0.5 # reconstruction_config['std_ratio']
        )

        return pcd
        
    def get_point_cloud_from_kinect(self, debug=False):
        '''
        Capture and get point cloud from Kinect camera (Will be used to build occupancy map)
        '''
        pcd_kinect = self.capture_pcd(use_camera='kinect')
        if debug==True:
            # import open3d as o3d
            o3d.visualization.draw_geometries([pcd_kinect])
        return pcd_kinect

    def get_point_cloud_from_realsense(self, debug=False):
        '''
        Capture and get point cloud from RealSense camera (will be used to build height maps for grasp checking and refinement)
        '''
        pcd_realsense = self.capture_pcd(use_camera='realsense')
        if debug==True:
            o3d.visualization.draw_geometries([pcd_realsense])
        return pcd_realsense

# motion planning test function
def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('sim_moveit_execution', anonymous=True)

    cartesian = rospy.get_param('~cartesian', True)
    print('move strait in cartesian space: ', cartesian)

    robot = moveit_commander.RobotCommander()

    group_names = robot.get_group_names()
    print('Available planning groups: ', group_names)  # 'manipulator' / 'endeffector'

    group_name = 'manipulator'
    move_group = moveit_commander.MoveGroupCommander(group_name)
    end_effector_link = move_group.get_end_effector_link()
    print('end-effector link: ', end_effector_link)

    move_group.allow_replanning(True)

    move_group.set_goal_position_tolerance(0.001)  # unit: meter
    move_group.set_goal_orientation_tolerance(0.001)  # unit: rad

    move_group.set_max_acceleration_scaling_factor(0.5)
    move_group.set_max_velocity_scaling_factor(0.5)

    # move_group.set_start_state_to_current_state()

    waypoints = []
    current_pose = move_group.get_current_pose(end_effector_link).pose
    print('current pose: ', current_pose)


    wpose = copy.deepcopy(current_pose)
    wpose.position.z -= 0.15
    if cartesian:
        waypoints.append(copy.deepcopy(wpose))
    else:
        move_group.set_pose_target(wpose)
        move_group.go()
        rospy.sleep(1.0)

    wpose.position.y += 0.15
    if cartesian:
        waypoints.append(copy.deepcopy(wpose))
    else:
        move_group.set_pose_target(wpose)
        move_group.go()
        rospy.sleep(1.0)

    wpose.position.x += 0.1
    if cartesian:
        waypoints.append(copy.deepcopy(wpose))
    else:
        move_group.set_pose_target(wpose)
        move_group.go()
        rospy.sleep(1.0)

    # planning
    if cartesian:
        fraction = 0.0  # path planning cover rate
        max_attemps = 100   # maximum try times
        attempts = 0     # already try times


    # plan a cartesian path that pass all waypoints
    while attempts < max_attemps and fraction < 1:
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints,  # waypoints list
            0.01,       # end-effector step, computed inverse kinematics every 0.01m
            0.0,        # jump_threshold, 0 means jump not allowed
            True        # avoid_collision
            )
        attempts += 1
        # print progress
        if attempts % 20 == 0:
            rospy.loginfo("Still trying after" + str(attempts) + "attempts......")

    # if planning succeed, fraction == 1, move robot
    if fraction == 1.0:
        rospy.loginfo("Path computed successfully, moving arm")
        move_group.execute(plan)
        rospy.loginfo("Path execution complete")
    else:
        rospy.loginfo("Path planning failed with only" + str(fraction) + "success after" + str(attempts) + "attempts")

    rospy.sleep(1.0)

    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
    


if __name__ == '__main__':
    rospy.init_node('test')
    # first test
    # main()

    # second test
    planner = MotionPlanner()
    end_effector_link = planner._move_group.get_end_effector_link()
    print('manipulator end-effector link name: {}'.format(end_effector_link))
    # pose_goal = []
    # pose_target = Pose()
    # pose_target.position.x = 0.1
    # pose_target.position.y = 0.5
    # pose_target.position.z = 0.3
    # pose_target.orientation.x = 1
    # pose_target.orientation.y = 0
    # pose_target.orientation.z = 0
    # pose_target.orientation.w = 0
    # pose_goal.append(copy.deepcopy(pose_target))

    # # pose_target.position.y += 0.15
    # # pose_goal.append(copy.deepcopy(pose_target))

    # # pose_target.position.x += 0.15
    # # pose_goal.append(copy.deepcopy(pose_target))

    # planner.move_cartesian_space(pose_goal)
