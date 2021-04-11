#!/usr/bin/env python 
import os
import numpy as np
import scipy as sp
import quaternion as quat
import joblib as jbl
from math                   import pi, sin, cos
#
import roslib
import rospy
import tf2_ros
from std_msgs.msg           import Float64
from geometry_msgs.msg      import TransformStamped
#

#
from DMP_R2H import DMP_R2H
from OrientationDynamics import OrientationDynamics

# Dynamic reconfigure
from dynamic_reconfigure.server import Server


class Discrete_Low_Pass:
    """Create a first order discrete low-pass filter
    x(k+1) = (1-dt*fc)*x(k) + K*dt*fc*u(k)
    """
    def __init__(self, dim, dt, fc, K=1):
        """initialize the filter

        Parameters
        ----------
        dim : [float]
            dimension of the input signal (1D-array,
            each component filtered as a separate signal)\n
        dt : [float]
            sampling time\n
        fc : [float]
            cut-off frequency\n
        K : int, optional
            filter's gain, by default 1\n

        Warnings
        -------
        Each filter keeps a internal state, so a different filter object should 
            be initialized for each 1-dimensional signal.\n
        Different signals shouldn't be passed to the same filter.
        """
        self.dim = dim
        self.x = np.array([0]*self.dim)
        self.dt = dt
        self.fc = fc
        self.K = K
    def reset(self):
        """Reset filter's state to an array of 0s
        """
        self.x = np.array([0]*self.dim)
    def filter(self, signal):
        """Give input and update the filter's state(=output) accordingly

        Parameters
        ----------
        signal : [np.array(self.dim)]
            input signal

        Returns
        -------
        [np.array(self.dim)]
            filter state, equal to the output of the filter
        """
        # input signal should be a NUMPY ARRAY
        self.x = (1-self.dt*self.fc)*self.x + self.K*self.dt*self.fc * signal
        return self.x
class Human_Watcher:
    def __init__(self, human_goal):
        ### Internal variables initilization
        self.human_goal = human_goal
        self.d = 1.0
        ### Subscribers
        self.human_pose_sub = rospy.Subscriber(name='vicon/Human/hand', data_class=TransformStamped, callback=self.update_human_hand_pose, queue_size=1)
        ### Publishers
        self.human_dist_pub = rospy.Publisher('/human_data/goal_distance', Float64, queue_size=1)
        self.human_dist_msg = Float64()
    
    def set_human_goal(self, human_goal):
        self.human_goal = human_goal
    
    def update_human_hand_pose(self,tf_stamped):
        position = np.array((
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z
        ))
        self.d = sp.linalg.norm((position-self.human_goal))
    
    def publish_distance(self):
        self.human_dist_msg.data = self.d
        self.human_dist_pub.publish(self.human_dist_msg)


if __name__ == '__main__':
    try:
        NODE_NAME='Human_watcher'
        ### Start node and publishers
        rospy.init_node(NODE_NAME)
        ### Set rate
        HZ = rospy.get_param('/Human_watcher/hz', default=200.0)
        rate = rospy.Rate(HZ)
        human_goal = np.array((-0.1, -0.14, 0.98))
        hw = Human_Watcher(human_goal)
        while not rospy.is_shutdown():
            hw.publish_distance()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()




# /iiwa/wiggo_wiggo_yee/parameter_descriptions
# /iiwa/wiggo_wiggo_yee/parameter_updates
# /joint_group_vel_controller/command
# /joint_states
# /my_cartesian_motion_controller/target_frame
# /my_cartesian_vforce_controller/ft_sensor_wrench
# /my_cartesian_vforce_controller/joint_des_cmd
# /my_cartesian_vforce_controller/pd_gains/rot_x/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/rot_x/parameter_updates
# /my_cartesian_vforce_controller/pd_gains/rot_y/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/rot_y/parameter_updates
# /my_cartesian_vforce_controller/pd_gains/rot_z/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/rot_z/parameter_updates
# /my_cartesian_vforce_controller/pd_gains/trans_x/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/trans_x/parameter_updates
# /my_cartesian_vforce_controller/pd_gains/trans_y/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/trans_y/parameter_updates
# /my_cartesian_vforce_controller/pd_gains/trans_z/parameter_descriptions
# /my_cartesian_vforce_controller/pd_gains/trans_z/parameter_updates
# /my_cartesian_vforce_controller/solver/parameter_descriptions
# /my_cartesian_vforce_controller/solver/parameter_updates
# /my_cartesian_vforce_controller/target_wrench
# /pos_joint_traj_controller/command
# /pos_joint_traj_controller/follow_joint_trajectory/cancel
# /pos_joint_traj_controller/follow_joint_trajectory/feedback
# /pos_joint_traj_controller/follow_joint_trajectory/goal
# /pos_joint_traj_controller/follow_joint_trajectory/result
# /pos_joint_traj_controller/follow_joint_trajectory/status
# /pos_joint_traj_controller/state
# /rosout
# /rosout_agg
# /speed_scaling_factor
# /tf
# /tf_static
# /ur_hardware_interface/io_states
# /ur_hardware_interface/robot_mode
# /ur_hardware_interface/robot_program_running
# /ur_hardware_interface/safety_mode
# /ur_hardware_interface/script_command
# /ur_hardware_interface/set_mode/cancel
# /ur_hardware_interface/set_mode/feedback
# /ur_hardware_interface/set_mode/goal
# /ur_hardware_interface/set_mode/result
# /ur_hardware_interface/set_mode/status
# /ur_hardware_interface/tool_data
# /wrench
