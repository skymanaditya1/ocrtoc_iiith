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

from ocrtoc_common.transform_interface import TransformInterface

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

        self.to_home_pose()
        # self.test()
        self.place()
        rospy.sleep(1.0)
        
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

    
    def test(self):
        
        
        pose_goal = Pose()
              
        
        print("test 1, joint space to a REST POSE")
        
        self.to_rest_pose()
        rospy.sleep(2)
        time.sleep(10)
        
        
        print("test 2 REST POSE to GOAL 1")
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.1
        pose_goal.position.y = -0.3
        pose_goal.position.z = 0.2
        # self.move_joint_space(pose_goal)
        self.move_cartesian_space_upright(pose_goal)
        
        rospy.sleep(2)
        time.sleep(10)
        
        
        print("test 3 GOAL to REST POSE manual ")
        pose_goal.position.x = -0.112957249941
        pose_goal.position.y = 2.9801544038e-05
        pose_goal.position.z = 0.590340135745
        pose_goal.orientation.x = -0.923949504923
        pose_goal.orientation.y = 0.382514458771
        pose_goal.orientation.z = -3.05585242637e-05
        pose_goal.orientation.w = 1.57706453844e-05
        
        rospy.sleep(4)
        time.sleep(5)
             
        print("tes 4 REST pose TO REST pose")
        
        self.to_rest_pose()
        rospy.sleep(4)
        time.sleep(5)
        print("test 5 cartesian to goal")
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.1
        pose_goal.position.y = 0.3
        pose_goal.position.z = 0.2
        self.move_cartesian_space_upright(pose_goal)
        
        print("test 6 joint to goal")
        
        self.to_rest_pose()
    
    
    
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
        # group_goal = self.ee_goal_to_link8_goal(rest_pose)
        # print("group goal after tf", group_goal)
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
        if gripper_dist[0] > 0.0005 and gripper_dist[1] > 0.0005:
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
                # self.to_rest_pose()
                print("At rest pose")
                success = self.gripper_width_test()
                if success == False:
                    return False
                
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
        
        self.to_rest_pose()
        
        
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
