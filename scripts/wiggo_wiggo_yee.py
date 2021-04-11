#!/usr/bin/env python 

import roslib
import rospy
import numpy as np
import scipy as sp
import quaternion as quat
from geometry_msgs.msg import PoseStamped, Pose
from math import pi

# Dynamic reconfigure
from dynamic_reconfigure.server import Server
from ur5_control.cfg import ControlTesterConfig


### PoseStamped description:
# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# geometry_msgs/Pose pose
#   geometry_msgs/Point position
#     float64 x
#     float64 y
#     float64 z
#   geometry_msgs/Quaternion orientation
#     float64 x
#     float64 y
#     float64 z
#     float64 w

def generate_random_PoseStamped(x_lim, y_lim, z_lim, r_min, r_max, frame_id):
    # Generate random position and orientation
    x_range = x_lim[1]-x_lim[0]
    y_range = y_lim[1]-y_lim[0]
    z_range = z_lim[1]-z_lim[0]
    p = quat.from_euler_angles(2*pi*np.random.rand(3))
    x = x_lim[1]-x_range*np.random.rand()
    y = y_lim[1]-y_range*np.random.rand()
    z = z_lim[1]-z_range*np.random.rand()
    position = np.array((x,y,z))
    r = np.linalg.norm(position)
    # Regenerate position if it's outside the boundaries given by r_min and r_max
    while (r<r_min) or (r>r_max):
        x = x_lim[1]-x_range*np.random.rand()
        y = y_lim[1]-y_range*np.random.rand()
        z = z_lim[1]-z_range*np.random.rand()
        position = np.array((x,y,z))
        r = np.linalg.norm(position)
    # Fill message and publish
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = x_lim[1]-x_range*np.random.rand()
    pose_msg.pose.position.y = y_lim[1]-y_range*np.random.rand()
    pose_msg.pose.position.z = z_lim[1]-z_range*np.random.rand()
    pose_msg.pose.orientation.w = p.w
    pose_msg.pose.orientation.x = p.x
    pose_msg.pose.orientation.y = p.y
    pose_msg.pose.orientation.z = p.z
    return pose_msg


class Control_Tester:
    def __init__(self, frame_id):
        # Parameters
        self.xlim = [0.0,0.0]
        self.ylim = [0.0,0.0]
        self.zlim = [0.0,0.0]
        self.f = np.array((0.0,0.0,0.0))
        self.frame_id = frame_id
        # Publisher
        self.target_pose_pub = rospy.Publisher('/my_cartesian_motion_controller/target_frame', PoseStamped, queue_size=1)
        # DYnamic reconfigure server
        self.dyn_srv = Server(ControlTesterConfig, self.dyn_reconfigure_callback)
    
    def set_lim_cartesian(self, xlim, ylim, zlim):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
    
    def set_frame_id(self, frame_id):
        self.frame_id = frame_id
    
    def publish_PoseStamped(self, pose_msg):
        # Check if position is inside limits
        if (pose_msg.pose.position.x < self.xlim[0]) or (pose_msg.pose.position.x > self.xlim[1]):
            print('ERROR: desired position has x outside range')
        elif (pose_msg.pose.position.y < self.ylim[0]) or (pose_msg.pose.position.y > self.ylim[1]):
            print('ERROR: desired position has y outside range')
        elif (pose_msg.pose.position.z < self.zlim[0]) or (pose_msg.pose.position.z > self.zlim[1]):
            print('ERROR: desired position has x outside range')
        else:
            self.target_pose_pub.publish(pose_msg)

    def dyn_reconfigure_callback(self, config, level):
        self.f[0] = config['fx']
        self.f[1] = config['fy']
        self.f[2] = config['fz']
        return config
    
    def generate_sinusoidal_pos_cmd(self, t, A, offset, pose_msg):
        # Fill position field of a copy of pose_msg
        # Sinusoidal trajectory for the end-effector position
        xdes = offset + A*np.sin(self.f*t*2*pi)
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = xdes[0]
        pose_msg.pose.position.y = xdes[1]
        pose_msg.pose.position.z = xdes[2]
        return pose_msg


    


if __name__ == '__main__':
    try:
        NODE_NAME='wiggo_wiggo_yee'
        ### Start node and publishers
        rospy.init_node(NODE_NAME)
        cmd_pub = rospy.Publisher(
            name='/my_cartesian_motion_controller/target_frame',
            data_class=PoseStamped,
            queue_size=1
        )
        ### Set rate
        HZ = rospy.get_param('/wiggo_wiggo_yee/hz', default=100.0)
        rate = rospy.Rate(HZ)
        ### Set control tester
        x_lim = [-0.7,0.7]
        y_lim = [-0.7,0.7]
        z_lim = [0.0,0.7]
        frame_id = 'base'
        A = 0.25
        offset = np.array((-0.4,0.4, 0.45))
        ct = Control_Tester(frame_id)
        ct.set_lim_cartesian(x_lim, y_lim, z_lim)
        ###
        pose_msg_1 = PoseStamped()
        pose_msg_1.header.frame_id = 'base'
        pose_msg_1.pose.position.x = -0.4
        pose_msg_1.pose.position.y = 0.4
        pose_msg_1.pose.position.z = 0.2
        pose_msg_1.pose.orientation.w = 1.0
        pose_msg_1.pose.orientation.x = 0.0
        pose_msg_1.pose.orientation.y = 0.0
        pose_msg_1.pose.orientation.z = 0.0
        ### Loop node
        while not rospy.is_shutdown():
            t = rospy.get_time()
            pose_msg_1 = ct.generate_sinusoidal_pos_cmd(t,A,offset,pose_msg_1)
            ct.publish_PoseStamped(pose_msg_1)
            #pose_msg = generate_random_PoseStamped(x_lim, y_lim, z_lim, 0.5, 0.8, frame_id)
            # Sleep
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()