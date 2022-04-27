# Author: Minghao Gou
#! /usr/bin/env python
import numpy as np
np.set_printoptions(suppress=True)
import cv2
from numpy.core.numeric import full
import rospy
import rospkg
import random

# import tf.transformations

import math
import open3d as o3d
import open3d_plus as o3dp
from transforms3d.quaternions import mat2quat
import time
import copy
import os
from copy import deepcopy

from graspnetAPI import GraspGroup

from .arm_controller import ArmController
from .graspnet import GraspNetBaseLine
from .pose.pose_6d import get_6d_pose_by_geometry, load_model_pcd
from .pose.pose_correspondence import get_pose_superglue

from ocrtoc_common.camera_interface import CameraInterface
from ocrtoc_common.transform_interface import TransformInterface
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

def crop_pcd(pcds, i, reconstruction_config):
    points, colors = o3dp.pcd2array(pcds[i])
    mask = points[:, 2] > reconstruction_config['z_min']
    mask = mask & (points[:, 2] < reconstruction_config['z_max'])
    mask = mask & (points[:, 0] > reconstruction_config['x_min'])
    mask = mask & (points[:, 0] < reconstruction_config['x_max'])
    mask = mask & (points[:, 1] < reconstruction_config['y_max'])
    mask = mask & (points[:, 1] > reconstruction_config['y_min'])
    pcd = o3dp.array2pcd(points[mask], colors[mask])
    return pcd

def kinect_process_pcd(pcd, reconstruction_config):
    points, colors = o3dp.pcd2array(pcd)
    mask = points[:, 2] > reconstruction_config['z_min']
    mask = mask & (points[:, 2] < reconstruction_config['z_max'])
    mask = mask & (points[:, 0] > reconstruction_config['x_min'])
    mask = mask & (points[:, 0] < reconstruction_config['x_max'])
    mask = mask & (points[:, 1] < reconstruction_config['y_max'])
    mask = mask & (points[:, 1] > reconstruction_config['y_min'])
    pcd = o3dp.array2pcd(points[mask], colors[mask])
    return pcd.voxel_down_sample(reconstruction_config['voxel_size'])

def process_pcds(pcds, use_camera, reconstruction_config):
    trans = dict()
    start_id = reconstruction_config['{}_camera_order'.format(use_camera)][0]
    pcd = copy.deepcopy(crop_pcd(pcds, start_id, reconstruction_config))
    print("np.asarray(pcd.points).shape[0]", np.asarray(pcd.points).shape)
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    idx_list = reconstruction_config['{}_camera_order'.format(use_camera)]
    for i in idx_list[1:]: # the order are decided by the camera pose
        voxel_size = reconstruction_config['voxel_size']
        income_pcd = copy.deepcopy(crop_pcd(pcds, i, reconstruction_config))
        
        
        print("np.asarray(income_pcd.points).shape[0]", np.asarray(income_pcd.points).shape)
        
        if np.asarray(income_pcd.points).shape[0] == 0:
            continue
        
        income_pcd, _ = income_pcd.remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd = o3dp.merge_pcds([pcd, income_pcd])
        cd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd

class Perceptor():
    def __init__(
        self,
        config
    ):
        self.config = config
        self.debug = self.config['debug']
        self.debug_pointcloud = self.config['debug_pointcloud']
        self.debug_grasp = self.config['debug_assigned_grasp']
        self.graspnet_baseline = GraspNetBaseLine(
            checkpoint_path = os.path.join(
                rospkg.RosPack().get_path('ocrtoc_perception'),
                self.config['graspnet_checkpoint_path']
            )
        )
        self.color_info_topic_name = self.config['color_info_topic_name']
        self.color_topic_name = self.config['color_topic_name']
        self.depth_topic_name = self.config['depth_topic_name']
        self.points_topic_name = self.config['points_topic_name']
        self.kinect_color_topic_name = self.config['kinect_color_topic_name']
        self.kinect_depth_topic_name = self.config['kinect_depth_topic_name']
        self.kinect_points_topic_name = self.config['kinect_points_topic_name']
        self.use_camera = self.config['use_camera']
        self.arm_controller = ArmController(topic = self.config['arm_topic'])
        self.camera_interface = CameraInterface()
        rospy.sleep(2)
        self.transform_interface = TransformInterface()
        self.transform_from_frame = self.config['transform_from_frame']
        if self.use_camera in ['realsense', 'both']:
            self.camera_interface.subscribe_topic(self.color_info_topic_name, CameraInfo)
            self.camera_interface.subscribe_topic(self.color_topic_name, Image)
            self.camera_interface.subscribe_topic(self.points_topic_name, PointCloud2)
            time.sleep(2)
            self.color_transform_to_frame = self.get_color_image_frame_id()
            self.points_transform_to_frame = self.get_points_frame_id()
        if self.use_camera in ['kinect', 'both']:
            self.camera_interface.subscribe_topic(self.kinect_color_topic_name, Image)
            self.camera_interface.subscribe_topic(self.kinect_points_topic_name, PointCloud2)
            time.sleep(2)
            self.kinect_color_transform_to_frame = self.get_kinect_color_image_frame_id()
            self.kinect_points_transform_to_frame = self.get_kinect_points_frame_id()
        self.fixed_arm_poses = np.loadtxt(
            os.path.join(
                rospkg.RosPack().get_path('ocrtoc_perception'),
                self.config['realsense_camera_fix_pose_file'],
            ),
            delimiter = ','
        )
        self.fixed_arm_poses_both = np.loadtxt(
            os.path.join(
                rospkg.RosPack().get_path('ocrtoc_perception'),
                self.config['both_camera_fix_pose_file'],
            ),
            delimiter = ','
        )


        self.mesh_folder = "/root/ocrtoc_ws/src/ocrtoc_materials/models"

    def get_color_image(self):
        return self.camera_interface.get_numpy_image_with_encoding(self.color_topic_name)[0]
    
    def get_kinect_image(self):
        return self.camera_interface.get_numpy_image_with_encoding(self.kinect_color_topic_name)[0]

    def get_color_camK(self):
        d = self.camera_interface.get_dict_camera_info(self.color_info_topic_name)
        return (np.array(d['K']).reshape((3,3)))

    def get_depth_image(self):
        return self.camera_interface.get_numpy_image_with_encoding(self.depth_topic_name)[0]

    def get_color_image_frame_id(self):
        return self.camera_interface.get_ros_image(self.color_topic_name).header.frame_id

    def get_depth_image_frame_id(self):
        return self.camera_interface.get_ros_image(self.depth_topic_name).header.frame_id

    def get_points_frame_id(self):
        return self.camera_interface.get_ros_points(self.points_topic_name).header.frame_id

    def get_kinect_color_image_frame_id(self):
        return self.camera_interface.get_ros_image(self.kinect_color_topic_name).header.frame_id

    def get_kinect_depth_image_frame_id(self):
        return self.camera_interface.get_ros_image(self.kinect_depth_topic_name).header.frame_id

    def get_kinect_points_frame_id(self):
        return self.camera_interface.get_ros_points(self.kinect_points_topic_name).header.frame_id

    def get_pcd(self, use_graspnet_camera_frame = False):
        pcd = self.camera_interface.get_o3d_pcd(self.points_topic_name)
        return pcd

    def kinect_get_pcd(self, use_graspnet_camera_frame = False):
        return self.camera_interface.get_o3d_pcd(self.kinect_points_topic_name)

    def get_color_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.color_transform_to_frame)

    def get_points_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.points_transform_to_frame)

    def get_kinect_color_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.kinect_color_transform_to_frame)

    def get_kinect_points_transform_matrix(self):
        return self.transform_interface.lookup_numpy_transform(self.transform_from_frame, self.kinect_points_transform_to_frame)

    def capture_data(self):
        t1 = time.time()
        pcds = []
        color_images = []
        camera_poses = []
        # capture images by realsense. The camera will be moved to different locations.
        if self.use_camera in ['kinect', 'both']:
            points_trans_matrix = self.get_kinect_points_transform_matrix()
            full_pcd_kinect = self.kinect_get_pcd(use_graspnet_camera_frame = False) # in sapien frame.
            full_pcd_kinect.transform(points_trans_matrix)
            full_pcd_kinect = kinect_process_pcd(full_pcd_kinect, self.config['reconstruction'])
            if self.use_camera == 'both':
                
                # print("np.asarray(full_pcd_kinect.points).shape[0]", np.asarray(full_pcd_kinect.points).shape[0])
                                    
                pcds.append(full_pcd_kinect)
                kinect_image = self.get_kinect_image()
                kinect_image = cv2.cvtColor(kinect_image, cv2.COLOR_RGBA2RGB)
                # kinect_image = cv2.cvtColor(kinect_image, cv2.COLOR_RGB2BGR)
                # if self.debug_kinect:
                #     cv2.imshow('color', cv2.cvtColor(kinect_image, cv2.COLOR_RGB2BGR))
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                color_images.append(kinect_image)

                if self.debug:
                    print('points_trans_matrix:', points_trans_matrix)
                camera_poses.append(self.get_kinect_color_transform_matrix())
           
        if self.use_camera in ['realsense', 'both']:
            if self.use_camera == 'realsense':
                arm_poses = self.fixed_arm_poses
            else:
                arm_poses = np.array(self.fixed_arm_poses_both).tolist()
            for j, arm_pose in enumerate(arm_poses):
                self.arm_controller.exec_joint_goal(arm_pose)
                rospy.sleep(2.0)
                time.sleep(1.0)
                # if j not in self.config['reconstruction']['both_camera_order']:
                #     continue
                color_image = self.get_color_image()
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
                if self.debug:
                    cv2.imshow('color', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                color_images.append(color_image)

                points_trans_matrix = self.get_points_transform_matrix()
                if self.debug:
                    print('points_trans_matrix:', points_trans_matrix)
                camera_poses.append(self.get_color_transform_matrix())
                pcd = self.get_pcd(use_graspnet_camera_frame = False)
                pcd.transform(points_trans_matrix)
                pcd = kinect_process_pcd(pcd, self.config['reconstruction'])
                if self.debug:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    o3d.visualization.draw_geometries([pcd, frame])
                      
                pcds.append(pcd)

        # capture image by kinect
        
        # if more than one images are used, the scene will be reconstructed by regitration.
        if self.use_camera in ['realsense', 'both']:
            trans, full_pcd_realsense = process_pcds(pcds, use_camera = self.use_camera, reconstruction_config = self.config['reconstruction'])

        if self.use_camera == 'realsense':
            full_pcd = full_pcd_realsense
        elif self.use_camera == 'kinect':
            full_pcd = full_pcd_kinect
        elif self.use_camera == 'both':
            full_pcd = full_pcd_realsense
        else:
            raise ValueError('"use_camrera" should be "kinect", "realsense" or "both"')
        if self.debug:
            t2 = time.time()
            rospy.loginfo('Capturing data time:{}'.format(t2 - t1))
        

        return full_pcd, color_images, camera_poses

    def compute_6d_pose(self, full_pcd, color_images, camera_poses, pose_method, object_list):
        camK = self.get_color_camK()

        print('Camera Matrix:\n{}'.format(camK))
        if pose_method == 'icp':
            if self.debug:
                print('Using ICP to obtain 6d poses')
            object_poses = get_6d_pose_by_geometry(
                pcd = full_pcd,
                model_names = object_list,
                geometry_6d_config = self.config['6d_pose_by_geometry'],
                debug = self.debug
            )
        elif pose_method == 'superglue':
            if self.debug:
                rospy.logdebug('Using SuperGlue and PnP to obatin 6d poses')
            object_poses = get_pose_superglue(
                obj_list = object_list,
                images = color_images,
                camera_poses = camera_poses,
                camera_matrix = camK,
                superglue_config = self.config['superglue']
            )
        else:
            rospy.roserr('Unknown pose method:{}'.format(pose_method))
            raise ValueError('Unknown pose method:{}'.format(pose_method))
        if self.debug and pose_method == 'icp':
            # Visualize 6dpose estimation result.
            geolist = []
            geolist.append(full_pcd)
            for object_name in object_list:
                if object_name in object_poses.keys():
                    this_model_pcd = load_model_pcd(object_name, os.path.join(
                            rospkg.RosPack().get_path('ocrtoc_materials'),
                            'models'
                        )
                    )
                    geolist.append(this_model_pcd.transform(object_poses[object_name]['pose']))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            total_pcd = o3dp.merge_pcds([*geolist])
            o3d.visualization.draw_geometries([*geolist, frame])
            rospy.loginfo(object_poses)
        return object_poses

    def compute_grasp_pose(self, full_pcd):
        
        points, _ = o3dp.pcd2array(full_pcd)
        grasp_pcd = copy.deepcopy(full_pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-points)

        # generating grasp poses.
        gg = self.graspnet_baseline.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.config['graspnet']['refine_approach_dist']
        gg = self.graspnet_baseline.collision_detection(gg, points)
        
        print("Here are the grasp poses from the baseline {}".format(gg))

        # all the returned result in 'world' frame. 'gg' using 'graspnet' gripper frame.
        return gg
    
    def compute_grasp_poses3(self, full_pcd):
        '''
        Step 1: Call the contact graspnet API to generate the grasp poses
        '''
        # 1. Initialize all the paths and the data.
        random_value = random.randint(0, 10000)
        
        SAVE_PATH_NPZ = '/root/ocrtoc_ws/src/contact_graspnet/{}_{}.npy'.format('temp', random_value)
        SAVE_PATH_COLORS = '/root/ocrtoc_ws/src/contact_graspnet/{}_{}.npy'.format('temp_colors', random_value)
        GRASP_SAVE_PATH = '/root/ocrtoc_ws/src/contact_graspnet/{}_{}.npy'.format('grasps_temp', random_value)
        
        points, _ = o3dp.pcd2array(full_pcd)
        full_pcd_as_np = -(points.copy())
        full_pcd_colors_as_np = np.asarray(full_pcd.colors)*255 
        
        np.save(SAVE_PATH_NPZ, full_pcd_as_np)
        np.save(SAVE_PATH_COLORS, full_pcd_colors_as_np)
        
        # 3. Run contact graspnet through the bash command
        command = '/root/ocrtoc_ws/src/ocrtoc_perception/run_contact_graspnet.sh {} {} {}' \
            .format(SAVE_PATH_NPZ, SAVE_PATH_COLORS, GRASP_SAVE_PATH)
        
        print('Running Contact Graspnet')
        os.system(command)
        
        # 4. Load grasp poses generated by contact graspnet.
        grasps = np.load(GRASP_SAVE_PATH, allow_pickle=True).item()
        
        pred_grasps_cam = grasps['pred_grasps_cam'][-1]
        grasp_scores = grasps['grasp_scores'][-1]
        
        print(pred_grasps_cam.shape, grasp_scores.shape)
        
        '''
        Step 2: Formatting the grasp poses into the baseline grasp pose patterns
        '''
        grasps = []
        
        g_pose = None
        
        # follwing width, height, and depth values are taken from baseline graspnet
        width, height, depth = 0.07500000298023224, 0.019999999552965164, 0.019999999552965164
        
        for grasp_pose, grasp_score in zip(pred_grasps_cam, grasp_scores):
            grasp = np.array([grasp_score, width, height, depth, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1], dtype = np.float64)
            
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            frame.transform(grasp_pose)
            
            Rg = frame.get_rotation_matrix_from_zyx((0, -np.pi/2, np.pi/2))
            
            T = np.eye(4)
            T[:3, :3] = Rg
            
            grasp_pose = grasp_pose @ T
           
            grasp[4:13] = grasp_pose[:3, :3].reshape(-1)
            grasp[13:16] = grasp_pose[:3, -1].reshape(-1)
            
            grasps.append(grasp)
            g_pose = grasp_pose
            
        grasps = np.vstack(grasps)
        
        print('formatted shape: {}'.format(grasps.shape))
        
        gg = GraspGroup(grasps)
        
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.config['graspnet']['refine_approach_dist']
        gg = self.graspnet_baseline.collision_detection(gg, points)
        
        return gg, g_pose
    
    def compute_grasp_poses2(self, full_pcd):
        '''
        Step 1: Call the contact graspnet API to generate the grasp poses
        '''
        # 1. Initialize all the paths and the data.
        SAVE_PATH_NPZ = '/root/ocrtoc_ws/src/contact_graspnet/{}'.format('temp.npy')
        SAVE_PATH_COLORS = '/root/ocrtoc_ws/src/contact_graspnet/{}'.format('temp_colors.npy')
        GRASP_SAVE_PATH = '/root/ocrtoc_ws/src/contact_graspnet/{}'.format('grasps_temp.npy')
        
        full_pcd = copy.deepcopy(full_pcd)
        
        # 2. Transform the full pcd into the contact graspnet frame, otherwise, it will appear inverted.
        R = full_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        full_pcd.rotate(R)
        
        full_pcd_as_np = np.asarray(full_pcd.points)
        full_pcd_colors_as_np = np.asarray(full_pcd.colors)*255 

        np.save(SAVE_PATH_NPZ, full_pcd_as_np)
        np.save(SAVE_PATH_COLORS, full_pcd_colors_as_np)
        
        # 3. Run contact graspnet through the bash command
        command = '/root/ocrtoc_ws/src/ocrtoc_perception/run_contact_graspnet.sh {} {} {}' \
            .format(SAVE_PATH_NPZ, SAVE_PATH_COLORS, GRASP_SAVE_PATH)
        
        print('Running Contact Graspnet')
        os.system(command)
        
        # 4. Load grasp poses generated by contact graspnet.
        grasps = np.load(GRASP_SAVE_PATH, allow_pickle=True).item()
        
        pred_grasps_cam = grasps['pred_grasps_cam'][-1]
        grasp_scores = grasps['grasp_scores'][-1]
        
        print(pred_grasps_cam.shape, grasp_scores.shape)
        
        '''
        Step 2: Formatting the grasp poses into the baseline grasp pose patterns
        '''
        grasps = []
        
        g_pose = None
        
        # follwing width, height, and depth values are taken from baseline graspnet
        width, height, depth = 0.07500000298023224, 0.019999999552965164, 0.019999999552965164
        
        for grasp_pose, grasp_score in zip(pred_grasps_cam, grasp_scores):
            # 1. global transform
            # here we first reverse the transform that we did initially for generating the grasp poses.
            T = np.eye(4)
            T[:3, :3] = R
            
            grasp_pose = np.linalg.inv(T) @ grasp_pose
            
            # 2. local transform to align the axes properly
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            frame.transform(grasp_pose)
            
            Rg = frame.get_rotation_matrix_from_zyx((-np.pi/2, 0, 0))
            
            T = np.eye(4)
            T[:3, :3] = Rg
            T[:3, -1] = np.array([0, -0.0, 0.087])
            
            grasp_pose = grasp_pose @ T
            tx, ty, tz = grasp_pose[:3, -1]
            
            # 2. global transform part 2, here we shift the y axis a little to adjust with the baseline frame of reference.
            grasp_pose[:3, -1] = np.array([tx, ty-0.04, tz])
            
            # 3. assignment of the grasp array to make it consistent with the baseline class (GraspGroup)
            # grasp_score (1), width (1), height (1), depth (1), rotation (9), translation (3), object_id (1)
            grasp = np.array([grasp_score, width, height, depth, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1], dtype = np.float64)
           
            grasp[4:13] = grasp_pose[:3, :3].reshape(-1)
            grasp[13:16] = grasp_pose[:3, -1].reshape(-1)
            
            grasps.append(grasp)
            g_pose = grasp_pose
            
        grasps = np.vstack(grasps)
        
        print('formatted shape: {}'.format(grasps.shape))
        
        gg = GraspGroup(grasps)
        
        # 4. another transformation for axes change
        ts = gg.translations
        rs = gg.rotation_matrices
        
        # ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)
        
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]
        
        gg.translations = ts
        gg.rotation_matrices = eelink_rs
        
        return gg, g_pose 
    
    

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return x, y, z
    
    def get_bounding_box(self, object_name, object_pose):
        
        
        mesh_name = object_name.rsplit('_', 1)[0]
        folder_path = self.mesh_folder
        folder_path = os.path.join(folder_path, mesh_name)
        mesh_file = os.path.join(folder_path, 'textured.obj')
        object_pose = object_pose
        
        pcd = o3d.io.read_triangle_mesh(mesh_file)
        def pcd_rotate_setup(pcd, rotation):
        
        
            roll = rotation[0] / 360. * (2*np.pi)
            pitch = rotation[1] / 360. * (2*np.pi)
            yaw = (rotation[2] / 360.) * (2*np.pi)
            # print(roll, pitch, yaw)
            R = pcd.get_rotation_matrix_from_xyz((roll, pitch, yaw))
            # print(R)
            pcd_r = pcd.rotate(R, center=(0, 0, 0))
            return pcd 

        
        x = object_pose['pose'][0, 3]
        y = object_pose['pose'][1, 3]
        z = object_pose['pose'][2, 3]
        translation = [x, y, z]
        R = object_pose['pose'][:3, :3]
        rotation = self.rotationMatrixToEulerAngles(R)
        
        pcd_r = pcd_rotate_setup(pcd, rotation)
        pcd_t = pcd_r.translate((translation[0], translation[1], translation[2]))

        bb_box = pcd_t.get_axis_aligned_bounding_box()
        points = bb_box.get_box_points()
        points = np.asarray(points)
        
        
        return points
    
    def assign_grasp_pose3(self, gg, object_poses):
        grasp_poses = dict()
        
        dist_thresh = self.config['response']['dist_thresh']
        object_names = []
        
        ts = gg.translations
        scores = gg.scores
        # print("here are the scores for all the grasps, now decide for yourself.")
        # print(scores)

        depths = gg.depths
        rs = gg.rotation_matrices
        
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)
        
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]
        
        min_object_ids = dict()
        
        for i, object_name in enumerate(object_poses.keys()):
            object_names.append(object_name)
            object_pose = object_poses[object_name]
            
            dists = np.linalg.norm(ts - object_pose['pose'][:3, 3], axis = 1)

            # print("ts shape", ts.shape)
            # print("object_pose['pose'][:3, 3]", object_pose['pose'][:3, 3].shape)
            # print(ts)

           
            object_mask = np.logical_and(dists < dist_thresh, ts[:, 2] < 0.2)  #why 0.2 on z?
            
            
                        
            min_object_ids[i] = object_mask
            
        remain_gg = [] 
        
        for i, object_name in enumerate(object_poses.keys()):
            # initially, we only pick the grasp poses that is assigned 
            # to the current object id (i)
            
            object_pose = object_poses[object_name]
            
            object_mask = min_object_ids[i]
            
            if np.sum(object_mask) == 0:
                grasp_poses[object_name] = None
                continue
            
            i_gg = gg[object_mask]
            i_scores = scores[object_mask]
            i_ts = ts[object_mask]
            i_eelink_rs = eelink_rs[object_mask]
            
            # next, we only pick the grasp poses sorted according to 
            # the confidence threshold. We only pick the top n poses.
            
            n = 10000
            
            top_indices = (-i_scores).argsort()[:n]
            top_i_gg = i_gg[top_indices]
            top_i_eelink_rs = i_eelink_rs[top_indices]
            top_i_ts = i_ts[top_indices]
            
            
            points_bb = self.get_bounding_box(object_name, object_pose)
            min_x  = np.min(points_bb[:,0])
            min_y  = np.min(points_bb[:,1])
            
            max_x  = np.max(points_bb[:,0])
            max_y  = np.max(points_bb[:,1])
            print(object_name)
            print(points_bb, " Points bb")
            
            position = object_pose['pose'][:3, 3]
            
            print("Object position", position)
            print("Object min-max", min_x, min_y, max_x, max_y)
            
            # print('top_i_ts[1,:]', top_i_ts[1,:])
            object_mask = (top_i_ts[:, 0] < max_x) 
            object_mask = object_mask & (top_i_ts[:, 0] > min_x)
            object_mask = object_mask & (top_i_ts[:, 1] < max_y)
            object_mask = object_mask & (top_i_ts[:, 1] > min_y)
            
           
            # top_indices = top_indices[object_mask]
            print("top_i_gg", top_i_gg)
            # print("top_i_gg shape", top_i_gg.shape)
            top_i_gg = top_i_gg[object_mask]
            print("After mask")
            print("top_i_gg", top_i_gg)
            
            top_i_eelink_rs = top_i_eelink_rs[object_mask]
            top_i_ts = top_i_ts[object_mask]
                       
            print("top_i_ts", top_i_ts.shape)
            
            if top_i_ts.shape[1] == 0 or top_i_ts.shape[0] == 0 :
                print("No poses in the bbox")
                grasp_poses[object_name] = None
                continue
            
            print("test2")
            top_i_euler = np.array([self.rotationMatrixToEulerAngles(r) for r in top_i_eelink_rs])
            print('Top Eulers shape', top_i_euler.shape)
            
            
            dists = np.linalg.norm(top_i_ts - object_pose['pose'][:3, 3], axis = 1)
            
            print('Here is the dists:', dists)
            
            smallest_index = np.argmin(dists)
            
            best_gg = top_i_gg[int(smallest_index)]
            
            print("best_gg", best_gg)
            print("top_i_ts[smallest_index]", top_i_ts[smallest_index])
            print("smallest_index", smallest_index)
            
            
            remain_gg.append(best_gg.to_open3d_geometry())
            
            grasp_rotation_matrix = top_i_eelink_rs[smallest_index]
            gqw, gqx, gqy, gqz = mat2quat(grasp_rotation_matrix)
            
            # gqx, gqy, gqz, gqw = [ 0.0005629, -0.706825, 0.707388, 0.0005633 ]
                        
            grasp_poses[object_name] = {
                'x': top_i_ts[smallest_index][0],
                'y': top_i_ts[smallest_index][1],
                'z': top_i_ts[smallest_index][2],
                'qw': gqw,
                'qx': gqx,
                'qy': gqy,
                'qz': gqz
            }
            
            
            print(object_name)
            print(grasp_poses[object_name])
            points_bb = self.get_bounding_box(object_name, object_pose)
            
            min_x  = np.min(points_bb[:,0])
            min_y  = np.min(points_bb[:,1])
            
            max_x  = np.max(points_bb[:,0])
            max_y  = np.max(points_bb[:,1])
            
            print("Object min-max", min_x, max_x, min_y, max_y)
            print(object_name)
            print(grasp_poses[object_name])
    
        return grasp_poses, remain_gg
    
    
    
    def assign_grasp_pose2(self, gg, object_poses):
        grasp_poses = dict()
        
        dist_thresh = self.config['response']['dist_thresh']
        object_names = []
        
        ts = gg.translations
        scores = gg.scores
        print("here are the scores for all the grasps, now decide for yourself.")
        print(scores)
        # assert False

        depths = gg.depths
        rs = gg.rotation_matrices
        
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)
        
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]
        
        min_object_ids = dict()
        
        for i, object_name in enumerate(object_poses.keys()):
            object_names.append(object_name)
            object_pose = object_poses[object_name]
            
            dists = np.linalg.norm(ts - object_pose['pose'][:3, 3], axis = 1)

            print("ts shape", ts.shape)
            print("object_pose['pose'][:3, 3]", object_pose['pose'][:3, 3].shape)
            print(ts)

            # assert False
            
            # object_mask = ts[:, 2] < 0.2
            
            object_mask = np.logical_and(dists < dist_thresh, ts[:, 2] < 0.2)
            
            print('here is the gg[object_mask]: {}'.format(gg[object_mask]))
            
            min_object_ids[i] = object_mask
            
        remain_gg = [] 
        
        for i, object_name in enumerate(object_poses.keys()):
            # initially, we only pick the grasp poses that is assigned 
            # to the current object id (i)
            
            object_pose = object_poses[object_name]
            
            object_mask = min_object_ids[i]
            
            if np.sum(object_mask) == 0:
                grasp_poses[object_name] = None
                continue
            
            i_gg = gg[object_mask]
            i_scores = scores[object_mask]
            i_ts = ts[object_mask]
            i_eelink_rs = eelink_rs[object_mask]
            
            # next, we only pick the grasp poses sorted according to 
            # the confidence threshold. We only pick the top n poses.
            
            n = 10000
            
            # correct_indices = i_scores > 0.25
            
            # if correct_indices.sum() == 0:
            #     grasp_poses[object_name] = None
            #     continue
            
            # i_scores = i_scores[correct_indices]
            # i_gg = i_gg[correct_indices]
            # i_ts = i_ts[correct_indices]
            # i_eelink_rs = i_eelink_rs[correct_indices]
            
            top_indices = (-i_scores).argsort()[:n]
            top_i_gg = i_gg[top_indices]
            top_i_eelink_rs = i_eelink_rs[top_indices]
            top_i_ts = i_ts[top_indices]
            
            top_i_euler = np.array([self.rotationMatrixToEulerAngles(r) for r in top_i_eelink_rs])
            print('Top Eulers shape', top_i_euler.shape)
            
            print("i scores")
            print(i_scores)
            print("======")

            # next, we want the poses with the lowest gravitional angle
            # we convert to euler, ideal is np.pi, 0. We sort according
            # to the norm and then take the minimum of all the angls.
            # not the minimum, but we are putting a threshold. 
            
            # ideal_angle = np.array([np.pi, 0])
            # angles_scores = np.linalg.norm(ideal_angle - top_i_euler[:, :2], axis = 1)
            # print(" angle scores: {}".format(angles_scores))
            # print(" top i euler {}: ".format(top_i_euler[:, :2]))
            
            # smallest_index = np.argmin(angles_scores)
            
            dists = np.linalg.norm(top_i_ts - object_pose['pose'][:3, 3], axis = 1)
            
            print('Here is the dists:', dists)
            
            smallest_index = np.argmin(dists)
            print("smallest index {}".format(smallest_index))
            print("selected distance: ", dists[smallest_index])
            print("object_name: ", object_name)
            print("selected grasp ts ", top_i_ts[smallest_index])
            print("object pose: ", object_pose)
            print('Best pose: {}'.format(top_i_euler[smallest_index]))
            
            best_gg = top_i_gg[int(smallest_index)]
            
            remain_gg.append(best_gg.to_open3d_geometry())
            
            grasp_rotation_matrix = top_i_eelink_rs[smallest_index]
            gqw, gqx, gqy, gqz = mat2quat(grasp_rotation_matrix)
            
            # gqx, gqy, gqz, gqw = [ 0.0005629, -0.706825, 0.707388, 0.0005633 ]
                        
            grasp_poses[object_name] = {
                'x': top_i_ts[smallest_index][0],
                'y': top_i_ts[smallest_index][1],
                'z': top_i_ts[smallest_index][2],
                'qw': gqw,
                'qx': gqx,
                'qy': gqy,
                'qz': gqz
            }
    
        return grasp_poses, remain_gg

    def assign_grasp_pose(self, gg, object_poses):
        save_things = {
            'gg': gg,
            'object_poses': object_poses
        }
        
        np.save('/root/ocrtoc_ws/src/gg.npy', save_things)
        
        grasp_poses = dict()
        dist_thresh = self.config['response']['dist_thresh']
        # - dist_thresh: float of the minimum distance from the grasp pose center to the object center. The unit is millimeter.
        angle_thresh = self.config['response']['angle_thresh']
        # - angle_thresh:
        #             /|
        #            / |
        #           /--|
        #          /   |
        #         /    |
        # Angle should be smaller than this angle

        object_names = []
        # gg: GraspGroup in 'world' frame of 'graspnet' gripper frame.
        # x is the approaching direction.
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        # move the center to the eelink frame
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)

        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        # min_dist: np.array of the minimum distance to any object(must > dist_thresh)
        min_dists = np.inf * np.ones((len(rs)))

        # min_object_ids: np.array of the id of the nearest object.
        min_object_ids = -1 * np.ones(shape = (len(rs)), dtype = np.int32)

        # first round to find the object that each grasp belongs to.

        # Pay attention that even the grasp pose may be accurate,
        # poor 6dpose estimation result may lead to bad grasping result
        # as this step depends on the 6d pose estimation result.
        angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))
        for i, object_name in enumerate(object_poses.keys()):
            object_names.append(object_name)
            object_pose = object_poses[object_name]

            dists = np.linalg.norm(ts - object_pose['pose'][:3,3], axis=1)
            print('distances {}'.format(dists))
            object_mask = np.logical_and(dists < min_dists, dists < dist_thresh)
            print('object mask {}'.format(object_mask))

            min_object_ids[object_mask] = i
            min_dists[object_mask] = dists[object_mask]
        remain_gg = []
        # second round to calculate the parameters
        for i, object_name in enumerate(object_poses.keys()):
            object_pose = object_poses[object_name]

            obj_id_mask = (min_object_ids == i)
            add_angle_mask = (obj_id_mask & angle_mask)
            # For safety and planning difficulty reason, grasp pose with small angle with gravity direction will be accept.
            # if no grasp pose is available within the safe cone. grasp pose with the smallest angle will be used without
            # considering the angle.
            if np.sum(add_angle_mask) < self.config['response']['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                mask = obj_id_mask
                sorting_method = 'angle'
            else:
                mask = add_angle_mask
                sorting_method = 'score'
            if self.debug:
                print(f'{object_name} using sorting method{sorting_method}, mask num:{np.sum(mask)}')
            i_scores = scores[mask]
            i_ts = ts[mask]
            i_eelink_rs = eelink_rs[mask]
            i_rs = rs[mask]
            i_gg = gg[mask]
            if np.sum(mask) < self.config['response']['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                # ungraspable
                grasp_poses[object_name] = None
            else:
                if sorting_method == 'score':
                    best_grasp_id = np.argmax(i_scores)
                elif sorting_method == 'angle':
                    best_grasp_id = np.argmin(i_rs[:, 2, 0])
                else:
                    raise ValueError('Unknown sorting method')
                best_g = i_gg[int(best_grasp_id)]
                remain_gg.append(best_g.to_open3d_geometry())
                grasp_rotation_matrix = i_eelink_rs[best_grasp_id]
                if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                    if self.debug:
                        print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                    grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                else:
                    if self.debug:
                        print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                gqw, gqx, gqy, gqz = mat2quat(grasp_rotation_matrix)
                grasp_poses[object_name] = {
                    'x': i_ts[best_grasp_id][0],
                    'y': i_ts[best_grasp_id][1],
                    'z': i_ts[best_grasp_id][2],
                    'qw': gqw,
                    'qx': gqx,
                    'qy': gqy,
                    'qz': gqz
                }
        return grasp_poses, remain_gg


    def pcd_rotate_translate_setup(self, pcd, object_pose):
        
            x = object_pose['pose'][0, 3]
            y = object_pose['pose'][1, 3]
            z = object_pose['pose'][2, 3]
            translation = [x, y, z]
            R = object_pose['pose'][:3, :3]
            rotation = self.rotationMatrixToEulerAngles(R)
        
            roll = rotation[0] / 360. * (2*np.pi)
            pitch = rotation[1] / 360. * (2*np.pi)
            yaw = (rotation[2] / 360.) * (2*np.pi)
            # print(roll, pitch, yaw)
            R = pcd.get_rotation_matrix_from_xyz((roll, pitch, yaw))
            # print(R)
            pcd_r = pcd.rotate(R, center=(0, 0, 0))
            
            
            pcd_t = pcd_r.translate((translation[0], translation[1], translation[2]))
            
            
            return pcd_t

    
    
    def icp_fullpcd_mesh(self, full_pcd, mesh_pcd):
        
        reconstruction_config = self.config['reconstruction']
        pcd = copy.deepcopy(full_pcd)
        mesh_pcd = copy.deepcopy(mesh_pcd)
        print("np.asarray(pcd.points).shape[0]", np.asarray(pcd.points).shape)
        pcd.estimate_normals()
        
        voxel_size = reconstruction_config['voxel_size']
        
        income_pcd  =  copy.deepcopy(mesh_pcd)      
            
        # print("np.asarray(income_pcd.points).shape[0]", np.asarray(income_pcd.points).shape)
            
        if np.asarray(income_pcd.points).shape[0] == 0:
            print("Mesh not readable")
            return full_pcd
            
        income_pcd, _ = income_pcd.remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    income_pcd,
                    pcd,
                    reconstruction_config['max_correspondence_distance'],
                    np.eye(4, dtype = np.float),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
                )
                if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                    and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                    # trace for transformation matrix should be larger than 3.5
                    # translation should less than 0.05
                    transok_flag = True
                    break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        
        pcd = o3dp.merge_pcds([pcd, income_pcd])
        cd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
        return pcd
            
        
    
    
    def icp_finer(self, pcd, object_poses):
        
        pcd_full = pcd
        
        
        
        
        for i, object_name in enumerate(object_poses.keys()):
            
            mesh_name = object_name.rsplit('_', 1)[0]
            object_pose = object_poses[object_name]
               
            folder_path = self.mesh_folder
            folder_path = os.path.join(folder_path, mesh_name)
            mesh_file = os.path.join(folder_path, 'visual.ply')
            object_pose = object_pose
        
       
            pcd_mesh = o3d.io.read_point_cloud(mesh_file)
        
            pcd_mesh = self.pcd_rotate_translate_setup(pcd_mesh, object_pose)
            
            
            pcd_full = self.icp_fullpcd_mesh(pcd_full, pcd_mesh)

        
        
        
        
        
        return pcd_full

    def percept(
            self,
            object_list,
            pose_method,
        ):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry infomation is used in this implementation.

        There are mainly three steps.
        - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
        - Generating objects 6d poses by mainly icp matching.
        - Generating grasping poses by graspnet-baseline.

        Args:
            object_list(list): strings of object names.
            pose_method: string of the 6d pose estimation method, "icp" or "superglue".
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''
        # Capture Data
        full_pcd, color_images, camera_poses = self.capture_data()
        

             
        

        # Computer Object 6d Poses
        print("Object list in perceptor: {}".format(object_list))
        object_poses = self.compute_6d_pose(
            full_pcd = full_pcd,
            color_images = color_images,
            camera_poses = camera_poses,
            pose_method = pose_method,
            object_list = object_list
        )


        # self.pcd = full_pcd
        # full_pcd = self.icp_finer(full_pcd, object_poses )
        
        # Compute Grasping Poses (Many Poses in a Scene)
        o3d.io.write_point_cloud("/root/ocrtoc_ws/src/test.pcd", full_pcd)
        # pcd_vis=  copy.deepcopy(full_pcd)
        # o3d.visualization.draw_geometries([pcd_vis])
        gg, t = self.compute_grasp_poses3(full_pcd)
        if self.debug_pointcloud:
            # print('g pose from the return function {}'.format(t))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            frame_grasp_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # frame_grasp_pose.transform(t)
            o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list(), frame_grasp_pose])


        # Assign the Best Grasp Pose on Each Object
        grasp_poses, remain_gg = self.assign_grasp_pose3(gg, object_poses)
        
        print("grasps ready")
        print(grasp_poses)
        print(remain_gg)
        if self.debug_grasp:
            
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            frame_grasp_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
           
            o3d.visualization.draw_geometries([frame, full_pcd, *remain_gg.to_open3d_geometry_list(), frame_grasp_pose])

            
        return object_poses, grasp_poses

    def get_response(self, object_list):
        '''
        Generating the defined ros perception message given the targe object list.

        Args:
            object_list(list): strings of object names.

        Returns:
            dict: both object and grasp poses which is close to the ros msg format.
        '''
        object_poses, grasp_poses = self.percept(
            object_list = object_list,
            pose_method = self.config['pose_method']
        )

        #####################################################
        # format of response_poses:
        # -------graspable
        #     |
        #     ---object_pose
        #     |  |
        #     |  |--x
        #     |  |
        #     |  |--y
        #     |  |
        #     |  ...
        #     |  |
        #     |  ---qw
        #     |
        #     ---grasp_pose (exists if graspable == True)
        #        |
        #        |--x
        #        |
        #        |--y
        #        |
        #        ...
        #        |
        #        ---qw
        #####################################################
        ### the keys of response_poses are object names.
        response_poses = dict()
        for object_name in object_poses.keys():
            response_poses[object_name] = dict()
            qw, qx, qy, qz = mat2quat(object_poses[object_name]['pose'][:3,:3])
            response_poses[object_name]['object_pose'] = {
                'x': object_poses[object_name]['pose'][0, 3],
                'y': object_poses[object_name]['pose'][1, 3],
                'z': object_poses[object_name]['pose'][2, 3],
                'qw': qw,
                'qx': qx,
                'qy': qy,
                'qz': qz
            }
            if grasp_poses[object_name] is None:
                response_poses[object_name]['graspable'] = False
            else:
                response_poses[object_name]['graspable'] = True
                response_poses[object_name]['grasp_pose'] = grasp_poses[object_name]
        return response_poses
