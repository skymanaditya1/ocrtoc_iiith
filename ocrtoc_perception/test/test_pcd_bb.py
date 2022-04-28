import open3d as o3d
import os
import numpy as np



# pcd_file = "/scratch/shankara/ocrtoc_iiith/1-1-1.pcd"

# pcd = o3d.io.read_point_cloud(pcd_file)

# o3d.visualization.draw_geometries([pcd])




folder_path = "/scratch/shankara/ocrtoc_iiith/ocrtoc_materials/models"
mesh_file = os.path.join(folder_path, 'cleanser/textured.obj')
print(mesh_file)

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


object_pose = [1,2,1]
translation = [5, 2, 2]
rotation = [1,2,3,4]
pcd_t = pcd.translate((translation[0], translation[1], translation[2]))

c = pcd_t.get_axis_aligned_bounding_box()
d = c.get_box_points()
e = np.asarray(d)
print(c)
print(e)

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(pcd)
# vis.add_geometry(c)
o3d.visualization.draw_geometries([pcd_t, c])
# o3d.draw_geometries([pcd, c])