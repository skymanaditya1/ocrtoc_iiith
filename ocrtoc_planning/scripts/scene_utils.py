
import os
import yaml
import open3d as o3d
import numpy as np
import pickle
import math
import rospkg
from geometry_msgs.msg._Quaternion import Quaternion

# Constants

# path to mesh parent
rospack = rospkg.RosPack()
MESH_PARENT_PATH = os.path.join(rospack.get_path('ocrtoc_materials'), 'models')

# Point-cloud
PCD_OCTREE_VOL = 2000
PCD_SCALE_FACTOR = 0.001

# Get Dimensions

def read_mesh(mesh_name):
    mesh_path = os.path.join(MESH_PARENT_PATH, mesh_name, 'visual.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh

def get_obj_points(mesh_name):
    mesh_path = os.path.join(MESH_PARENT_PATH, mesh_name, 'visual.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    points = np.asarray(mesh.vertices)
    return points

def get_obj_dimensions(obj_name):
    obj_vertices = get_obj_points(obj_name)
    obj_dim = np.max(obj_vertices, axis=0) - np.min(obj_vertices, axis=0)
    obj_dim = obj_dim.tolist()
    return obj_dim

def get_dimensions(path):
    obj_name_list = list(filter(lambda x: not x.endswith('yaml'), os.listdir(path)))
    obj_dim_dict = dict()
    for obj_name in obj_name_list:
        obj_dim_dict[obj_name] = get_obj_dimensions(obj_name)
    return obj_dim_dict

# Quaternions

def quat_inverse(quat): # Quaternion
    conj = Quaternion()
    conj.w = quat.w
    conj.x = -1.0 * quat.x
    conj.y = -1.0 * quat.y
    conj.z = -1.0 * quat.z
    return conj

def quat_mult(q, p): # Quaternion
    prod = Quaternion()
    prod.w = q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z
    prod.x = q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y
    prod.y = q.w * p.y + q.y * p.w - q.x * p.z + q.z * p.x
    prod.z = q.w * p.z + q.z * p.w + q.x * p.y - q.y * p.x
    return prod

def quat_diff(q, p): # Quaternion
    qDiff = quat_mult(quat_inverse(q), p)
    return qDiff

def quat_diff_ang(q, p): # Quaternion

    # angular diff. (from stackoverflow.com/a/23263233)
    diff = quat_diff(q, p)
    diff_ang = 2 * math.atan2(math.sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z), diff.w)

    return diff_ang

def quat_diff_ang_abs(q, p): # Quaternion

    diff_ang = quat_diff_ang(q, p)

    diff_ang_abs = abs(diff_ang) # [-pi,pi] -> [0,pi]
    diff_ang_abs = min(diff_ang, math.pi - diff_ang_abs) # consider 0, pi as parallel

    return math.degrees(diff_ang_abs)

# Point-cloud related

def get_octree_size(node):
    count = 0
    if not isinstance(node, o3d.geometry.OctreeLeafNode):
        for child in node.children:
            if isinstance(child, o3d.geometry.OctreeNode):
                childCount = get_octree_size(child)
                count += childCount
    else:
        count = 1
    return count

def get_pcd_vol(pcd):
    ## get volumne of point cloud as estimate of voxels
    VOXEL_SIZE = 0.01
    OCTREE_DEPTH = 6

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
    # voxels = voxel_grid.get_voxels()

    octree = voxel_grid.to_octree(OCTREE_DEPTH)
    octree_count = get_octree_size(octree.root_node)

    return octree_count

# Other utils

# get all instances with 
def getSimilarInstances(pose_dic, similarObjDim):
    DIM_ERR = 0.01 # error margin for dimensions
    obj_dimensions = get_dimensions(MESH_PARENT_PATH)
    similarInstances = []
    for key in pose_dic.keys():
        objName = key.split('_v')[0] # remove _vi suffix
        objDim = obj_dimensions[objName]
        if all(map(lambda i: abs(objDim[i] - similarObjDim[i]) < DIM_ERR, [0, 1, 2])):
            similarInstances.append(key) # object like similar Obj found
    return similarInstances

def getLargerInstances(pose_dic, minSizeObjDim):
    DIM_ERR = 0.01 # error margin for dimensions
    obj_dimensions = get_dimensions(MESH_PARENT_PATH)
    largerInstances = []
    for key in pose_dic.keys():
        objName = key.split('_v')[0] # remove _vi suffix
        objDim = obj_dimensions[objName]
        if all(map(lambda i: objDim[i] > minSizeObjDim[i] - DIM_ERR, [0, 1, 2])):
            largerInstances.append(key) # object larger than Obj found
    return largerInstances

# rearrange obj. names in cyclic order
def getInCyclicOrder(objNameList, pose_dic):

    SLOPE_INF = 99999.0

    # Slope
    def getSlope(objName1, objName2):
        objPos1 = pose_dic[objName1].position
        objPos2 = pose_dic[objName2].position
        slope = (objPos2.y - objPos1.y)/abs(objPos2.y - objPos1.y) * SLOPE_INF if objPos1.y != objPos2.y else 0
        if objPos1.x != objPos2.x:
            slope = (objPos2.y - objPos1.y)/(objPos2.x - objPos1.x)
        return slope
    
    objNameBySlope = []
    if len(objNameList):
        originObjName = objNameList[0]

        # get slopes from first
        objSlopeDic = {}
        for objName in objNameList[1:]:
            objSlopeDic[objName] = getSlope(originObjName, objName)
        
        # sort by slope
        objNameBySlope = sorted(objSlopeDic, key=objSlopeDic.get)
        objNameBySlope = [originObjName] + objNameBySlope
    
    return objNameBySlope

# check how close to Rect
def getRectDist(objNameList, pose_dic):

    # get in cyclic order
    objNameListCyclic = getInCyclicOrder(objNameList, pose_dic)

    # get distances
    getPos = lambda x: pose_dic[objNameListCyclic[x]].position
    getXYDist = lambda i,j: math.sqrt((getPos(i).x - getPos(j).x)**2 + (getPos(i).y - getPos(j).y)**2)
    side1, side2, side3, side4 = getXYDist(0, 1), getXYDist(1, 2), getXYDist(2, 3), getXYDist(3, 1)
    diag1, diag2 = getXYDist(0, 2)/math.sqrt(2), getXYDist(1, 3)/math.sqrt(2)

    diff1 = abs(side1 - side3)
    diff2 = abs(side2 - side4)
    diff3 = abs(diag1 - diag2)
    rectDist = diff1 + diff2 + diff3

    return rectDist
