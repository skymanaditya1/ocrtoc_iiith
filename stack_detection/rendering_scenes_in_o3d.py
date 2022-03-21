import open3d as o3d
import numpy as np
import yaml
import time

def get_object_list_from_yaml(yaml_path: str):
    object_list = None
    with open(yaml_path, "r") as f:
        object_list = yaml.safe_load(f)
    # print('Object Dictionary: \n{}'.format(object_list))
    return object_list

def save_pcd(pcd, save_path):
    o3d.io.write_point_cloud(save_path, pcd)

def draw_geometries(pcds):
    # viz = o3d.visualization
    # viz.rendering.Scene.GroundPlane()
    vis = o3d.visualization.Visualizer()
    # vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcds:
        vis.add_geometry(pcd)
    # vis.show_ground_plane()
    vis.run()
    vis.destroy_window()

if __name__=='__main__':

    debug = False # If set to true, you will be able to visualize each step 

    print("Started\n")
    runtime_begin = time.time()
    scene_dir = '/home/vishal/Volume_E/Active/Undergrad_research/Ocrtoc/OCRTOC_software_package/ocrtoc_materials/targets' # './final_scenes'
    scenes = ['1-1-1', '2-2-2', '5-3-1', '6-1-2']
    yaml_paths = []
    for scene in scenes:
        yaml_paths.append('{}/{}.yaml'.format(scene_dir, scene))
    current_scene_index = 1

    object_dict = get_object_list_from_yaml(yaml_paths[current_scene_index])

    meshes = []
    pcds = []
    full_pcd = o3d.geometry.PointCloud()
    for i, key in enumerate(object_dict):
        for object_pose in object_dict[key]:
            mesh = o3d.io.read_triangle_mesh('/home/vishal/Volume_E/Active/Undergrad_research/Ocrtoc/OCRTOC_software_package/ocrtoc_materials/models/{}/textured.obj'.format(str(key)))
            mesh2 = o3d.io.read_triangle_mesh('/home/vishal/Volume_E/Active/Undergrad_research/Ocrtoc/OCRTOC_software_package/ocrtoc_materials/models/{}/textured.obj'.format(str(key)))
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
            mesh.translate((dt[0], dt[1], dt[2]))

            if debug==True:
                o3d.visualization.draw_geometries([mesh, mesh2])
            # meshes.append(mesh)
            # pcd_direct = o3d.geometry.PointCloud()
            # pcd_direct.points = mesh.vertices
            # pcd_direct.colors = mesh.vertex_colors
            # print("mesh colors: {}".format(np.asarray(mesh.vertex_colors)))
            # print("pcd colors: {}".format(np.asarray(pcd_direct.colors)))

            # o3d.visualization.draw_geometries([pcd_direct])
            # exit()

            # For colored point cloud generation, refer this - https://github.com/Xelawk/mesh_utils 

            pcd = mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5, pcl=None)
            # print("Type of pcd: {}".format(type(pcd)))
            pcds.append(pcd)
            full_pcd.points = o3d.utility.Vector3dVector(np.vstack((o3d.utility.Vector3dVector(full_pcd.points), pcd.points)))

    # Creating ground plane
    points = np.hstack((np.random.uniform(low=-0.3, high=0.3, size=5000).reshape((5000, 1)),
                np.random.uniform(low=-0.6, high=0.6, size=5000).reshape((5000, 1)),
                np.zeros(shape=(5000, 1))))
    # print("shape: {}".format(points.shape))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.points = points
    pcds.append(pcd)
    full_pcd.points = o3d.utility.Vector3dVector(np.vstack((o3d.utility.Vector3dVector(full_pcd.points), pcd.points)))

    runtime_without_save = time.time()
    print("Runtime: ".format(runtime_without_save - runtime_begin))
    # full_pcd.extend(pcd)
    # exit()
    if debug==True:
        o3d.visualization.draw_geometries([full_pcd])
        o3d.visualization.draw_geometries(pcds)
        draw_geometries(pcds)
    save_path='/home/vishal/Volume_E/Active/Undergrad_research/Ocrtoc/Vishal_planner_stuff/Heuristic-graph-based-task-planner/helpers/trash/{}_final.pcd'.format(scenes[current_scene_index])
    # save_path='./test.pcd'
    # print(save_path)
    save_pcd(full_pcd, save_path=save_path)
    runtime_with_save = time.time()
    print("Runtime including PCD save operation: {}".format(runtime_with_save - runtime_begin))
    # viz = o3d.visualization.visualizer()

# mesh = o3d.io.read_triangle_mesh('./book_1/textured.obj')
# # pcd = o3d.io.read_point_cloud('./book_1/visual.ply')
# pcd = mesh.sample_points_poisson_disk(number_of_points=2000, init_factor=5, pcl=None)
# pcd2 = mesh.sample_points_poisson_disk(number_of_points=2000, init_factor=5, pcl=None)
# pcd.paint_uniform_color([1, 0.706, 0])
# R = pcd.get_rotation_matrix_from_quaternion((0, 0, 0, 1))
# pcd.rotate(R, center=(0, 0, 0))
# # pcd = mesh.sample_points_uniformly(number_of_points=20000, use_triangle_normal=False)
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = mesh.vertices
# # pcd.colors = mesh.vertex_colors
# # pcd.normals = mesh.vertex_normals
# o3d.visualization.draw_geometries([pcd, pcd2])