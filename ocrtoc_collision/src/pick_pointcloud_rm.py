#!/usr/bin/env python
import rospy


class heartbeat():
    def __init__(self):
       

        # init functions
        self.ros_init()     #node init
        self.publishers()   #publishers init
        self.subscribers()  #subscribers init
        
        #start sending data
        self.setup()

    def ros_init(self):
        rospy.init_node('heartbeat')
        print('[INFO] Intialized Node')
    
    def publishers(self):
        self.pub4000 = rospy.Publisher('arduino_in', String, queue_size=1)
        
    def subscribers(self):

        #Arduino
        rospy.Subscriber('/eye_ind', Float32, self.eye_callback, queue_size=1)
        


    #########  Velocity  ############
    def velocity_callback(self,data):
        self.velocity = round(data.data,2)
    
   

    def setup(self):
        print("collecting data ... ")
        while not rospy.is_shutdown():
            self.publish_4000()
            print("waiting for 5 seconds")
            rospy.Rate(0.2).sleep()
            self.data_collect()

if __name__ == '__main__':
    heartbeat_data = heartbeat()
    rospy.spin()