#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from pcl_helper import *
import pcl
from std_srvs.srv import Empty


class Pointcloud_remove():
    def __init__(self):
       

        # init functions
        self.ros_init()     #node init
        self.publishers()   #publishers init
        self.subscribers()  #subscribers init
        
        #start sending data
        self.setup()

    def ros_init(self):
        rospy.init_node('pointcloud_filter_collision')
        print('[INFO] Intialized Node')
        self.clear_octomap = 0
        self.grasp_data = False
        self.grasp_pose = None
    
    def publishers(self):
        self.pointcloud_pub = rospy.Publisher('/pointcloud_rm/output', PointCloud2, queue_size=1)
        self.test_cloud_pub = rospy.Publisher('/pointcloud/cropbox', PointCloud2, queue_size=2)
        
    def subscribers(self):

        rospy.Subscriber('/voxel_grid2/output', PointCloud2, self.pointcloud_callback, queue_size=5)
        rospy.Subscriber("/grasp_pose_coll/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/grasp_bool", Bool, self.grasp_callback)
        
        #subscribe to grasp pose too
    
    def collision_rm_grasp(self, grasp_pose):
        self.grasp_pose = grasp_pose
        
    
    def preprocess(self, pcl_msg):
        pcl_data = ros_to_pcl(pcl_msg, False)
        passthrough_x = pcl_data.make_passthrough_filter()
        # passthrough_x.setNegative(True)
        filter_axis = 'x'
        passthrough_x.set_filter_field_name(filter_axis)
        x_axis_min = 0.0 # to filter out the closest part of the cloud
        x_axis_max = 1.0 # to filter out the farest part of the cloud
        passthrough_x.set_filter_limits(x_axis_min, x_axis_max)
        # Finally use the filter function to obtain the resultant point cloud. 
        cloud_passthrough_2 = passthrough_x.filter()
    
        ros_cloud = pcl_to_ros(cloud_passthrough_2)
        print("preprocess")
        print(ros_cloud)
        return ros_cloud

    def crop(self, cloud_ros):
        
        cloud = ros_to_pcl(cloud_ros, color=False)
        clipper = cloud.make_cropbox()
        outcloud = pcl.PointCloud()
        print(self.grasp_pose.pose.position.y)
        tx = 0
        ty = self.grasp_pose.pose.position.y
        tz = self.grasp_pose.pose.position.z
        clipper.set_Translation(tx, ty, tz)
        print(tx,ty,tz)
        rx = 0
        ry = 0
        rz = 0
        clipper.set_Rotation(rx, ry, rz)
        
        minx = -0.1
        miny = -0.05
        minz = -0.15
        mins = 0
        maxx = 0.5
        maxy = 0.05
        maxz = 0.15
        maxs = 0
        clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
    
        clipper.set_Negative(True)
        outcloud = clipper.filter()
        ros_cloud = pcl_to_ros(outcloud, color = False)
        print("cropping")
        
        # if self.clear_octomap == 1:
        #     # for i in range(0,5):
        #     #     rospy.wait_for_service('/clear_octomap')    
        #     self.clear_octomap = 0
        x = rospy.ServiceProxy('clear_octomap', Empty) 
        return ros_cloud
    
    
    def grasp_callback(self, data):
        if data.data == True:
            self.grasp_data = True
            print("True")
    
    def pose_callback(self, data):
        
        self.clear_octomap = 1
        self.grasp_data = True
        self.grasp_pose = data
        print("Received Grasp pose", self.grasp_pose)
        
        
    def pointcloud_callback(self, data):
        pointcloud = data
        # print(pointcloud.header)
        # self.test_cloud_pub.publish(self.crop(pointcloud, []))
        if self.grasp_data == False:
            self.test_cloud_pub.publish(pointcloud)
        else:
            # pointcloud = self.preprocess(pointcloud)
            # print(pointcloud.header)
            
            self.test_cloud_pub.publish(self.crop(pointcloud))
        
   

    def setup(self):
        print("publishing pointcloud data ... ")
        

if __name__ == '__main__':
    pointcloud_init = Pointcloud_remove()
    rospy.spin()