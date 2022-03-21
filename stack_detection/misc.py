'''Miscellaneous helpers
This file contains various miscellaneous helpers for buffer detection
'''
import open3d as o3d
import copy
import numpy as np

class Object:
    def __init__(self, mesh_path):
        '''
        '''
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.copy_mesh = None
    
    def render_to_pose(self, object_pose):
        '''
        Parameters:
        object_pose: [6 element list] (x, y, z, roll, pitch, yaw)
        '''
        self.copy_mesh = copy.deepcopy(self.mesh)
        R = self.copy_mesh.get_rotation_matrix_from_xyz((object_pose[3], object_pose[4], object_pose[5]))
        center = np.array(self.copy_mesh.get_center())
        self.copy_mesh.rotate(R, center=(center[0], center[1], center[2]))
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