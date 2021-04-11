#!/usr/bin/env python 
import os
import numpy as np
import scipy as sp
import quaternion as quat

import roslib
import rospy

from ih2_azzurra.msg    import AutoGrasp
from ur5_control.msg    import ExternalWrench
from std_msgs.msg       import Bool, String
from geometry_msgs.msg  import Pose, Point

# Dynamic reconfigure
from dynamic_reconfigure.client import Client as DynRClient
from dynamic_reconfigure.server import Server
from ur5_control.cfg import ReleaseControllerConfig
#from ur5_control.cfg import DMPParamsConfig, DMPExternalEffectsConfig



class Discrete_Low_Pass_VariableStep:
    """Create a first order discrete low-pass filter
    x(k+1) = (1-dt*fc)*x(k) + K*dt*fc*u(k)
    """
    def __init__(self, dim, fc, K=1):
        """initialize the filter

        Parameters
        ----------
        dim : [float]
            dimension of the input signal (1D-array,
            each component is filtered as a separate signal)\n
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
        self.fc = fc
        self.K = K
    def reset(self):
        """Reset filter's state to an array of 0s
        """
        self.x = np.array([0]*self.dim)
    def filter(self, dt, signal, verbose=False):
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
        if dt > 1e-10:
            if (self.fc > 1.0/(2.0*dt)):
                #pass
                rospy.logwarn('Discrete_Low_Pass_VariableStep : dt too big for the selected cut-off frequency, aliasing may occur')
        # input signal should be a NUMPY ARRAY
        self.xd = (signal-self.x)/float(dt)
        self.x = (1-dt*self.fc)*self.x + self.K*dt*self.fc * signal
        if verbose:
            print('-->',signal)
            print(self.x)
            print(self.xd)
        return self.x, self.xd

class ReleaseController():
    def __init__(self, grasp_params_dict, w_ext_params_dict):
        self.grasp_type = grasp_params_dict['grasp_type']
        self.release_is_active = False
        self.w_ext = { 
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.w_ext_d = { 
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.w_ext_rel = { 
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.w_ext_rel_d = { 
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.lp_filters = {
            'w_ext': {
                'lin': Discrete_Low_Pass_VariableStep(dim = 3, fc=w_ext_params_dict['fc']),
                'ang': Discrete_Low_Pass_VariableStep(dim = 3, fc=w_ext_params_dict['fc'])
            },
            'w_ext_rel': {
                'lin': Discrete_Low_Pass_VariableStep(dim = 3, fc=w_ext_params_dict['fc']),
                'ang': Discrete_Low_Pass_VariableStep(dim = 3, fc=w_ext_params_dict['fc'])
            }
        }
        self.allowed_action_state_list = (
            'GRASP_ACTIVE',
            'GRASP_NOT_ACTIVE'
        )
        self.action_state = 'GRASP_NOT_ACTIVE'
        self.allowed_system_state_list = (
            'WAIT',
            'READY'
        )
        self.system_state = 'WAIT'
        self.allowed_grasp_state = {
            'CLOSED',
            'OPEN'
        }
        self.grasp_state = 'OPEN'
        # Publishers
        self.DMP_goal_pub =         rospy.Publisher('/DMP/target_goal', Pose, queue_size=5) 
        self.autograsp_pub =        rospy.Publisher('/autograsp_controller/command', AutoGrasp, queue_size=5)
        self.optoforce_reset_pub =  rospy.Publisher('/reset_OptoForce_absolute', Bool, queue_size=5) 
        self.action_state_pub = rospy.Publisher('/state_machine/action_state', String, queue_size=5)
        self.system_state_pub = rospy.Publisher('/state_machine/system_state', String, queue_size=5)
        # Test publisher, can be used to test functions
        self.test_publisher = rospy.Publisher('/release_controller/test', Point, queue_size=5)
        # Subscribers
        self.external_wrench_sub = rospy.Subscriber('/DMP/external_wrench', ExternalWrench, callback=self.update_external_wrench, queue_size=5)
        self.action_state_sub           = rospy.Subscriber('/state_machine/action_state', String, callback=self.update_action_state, queue_size=5)
        self.system_state_sub           = rospy.Subscriber('/state_machine/system_state', String, callback=self.update_system_state, queue_size=5)
        # Dunamic reconfigure clients
        self.ExtEffDynClient = DynRClient(name='/DMP_controller/set_parameters_ext_effects', timeout=None)
        self.dyn_srv = Server(ReleaseControllerConfig, self.dyn_reconfigure_callback)


    def update_external_wrench(self, external_wrench_msg):
        # Signal is filtered with the low-pass discrete filters defined by self.lp_filters (that also compute the derivative of the fltered signal)
        t = external_wrench_msg.time.data
        dt = t - self.w_ext['t']
        self.w_ext['t']     = t
        self.w_ext_d['t']   = t
        self.w_ext_rel['t']     = t
        self.w_ext_rel_d['t']   = t
        if (dt > 100.0) :
            return 
        self.w_ext['lin'], self.w_ext_d['lin'] = self.lp_filters['w_ext']['lin'].filter(
            dt,
            np.array([external_wrench_msg.absolute.force.x,  external_wrench_msg.absolute.force.y,   external_wrench_msg.absolute.force.z])
        )
        self.w_ext['ang'], self.w_ext_d['ang'] = self.lp_filters['w_ext']['ang'].filter(
            dt,
            np.array([external_wrench_msg.absolute.torque.x, external_wrench_msg.absolute.torque.y,  external_wrench_msg.absolute.torque.z])
        )

        self.w_ext_rel['lin'], self.w_ext_rel_d['lin'] = self.lp_filters['w_ext_rel']['lin'].filter(
            dt,
            np.array([external_wrench_msg.relative.force.x,  external_wrench_msg.relative.force.y,   external_wrench_msg.relative.force.z])
        )
        self.w_ext_rel['ang'], self.w_ext_rel_d['ang'] = self.lp_filters['w_ext_rel']['ang'].filter(
            dt,
            np.array([external_wrench_msg.relative.torque.x, external_wrench_msg.relative.torque.y,  external_wrench_msg.relative.torque.z])
        )

        # msg = Point()
        # msg.x = self.w_ext['lin'][0]
        # msg.y = self.w_ext['lin'][1]
        # msg.z = self.w_ext['lin'][2]
        # self.test_publisher.publish(msg)
    
    def update_action_state(self, string_msg):
        if string_msg.data in self.allowed_action_state_list:
            self.action_state = string_msg.data
        else:
            rospy.logwarn('update_action_state: \'%s\' value not allowed for action state' %(string_msg.data))
    
    def update_system_state(self, string_msg):
        if string_msg.data in self.allowed_system_state_list:
            self.system_state = string_msg.data
        else:
            rospy.logwarn('update_system_state: \'%s\' value not allowed for system state' %(string_msg.data))

    def reset_OptoForce_absolute(self):
        msg = Bool()
        msg.data = True
        self.optoforce_reset_pub.publish(msg)
    
    def set_virtual_compliance(self, **kwargs):
        new_params = kwargs
        self.ExtEffDynClient.update_configuration(new_params)

    def send_grasp_pose(self, grasp_type, grasp_step, grasp_force):
        if (grasp_step>100) or (grasp_force>255):
            rospy.logwarn("send_grasp_pose() error: grasp_step>100 or grasp_force>255. Not publishing")
            return
        if not(grasp_type in ['RLX', 'LAT', 'PIN', 'TRI', 'CYL']):
            rospy.logwarn("send_grasp_pose() error: selected grasp_type (\'%s\') doesn't exist. Not publishing" %(grasp_type))
            return
        msg = AutoGrasp()
        msg.grasp_type = grasp_type
        msg.grasp_step = int(grasp_step)
        msg.grasp_force = int(grasp_force)
        self.autograsp_pub.publish(msg)
    
    def set_grasp_state(self, state):
        if not (state in self.allowed_grasp_state):
            rospy.logwarn('set_grasp_state: selected state (\'%s\') is not allowed' %(state))
        self.grasp_state = state
    
    def release(self):
        Kf_ext_z_old = self.ExtEffDynClient.get_configuration()['Kf_ext_z']
        # Remove z-axis compliance and coupling effect?
        self.set_virtual_compliance(Kf_ext_z=0.0)
        # Open hand
        self.send_grasp_pose(self.grasp_type, 0, 200)
        # state_msg = String()
        # state_msg.data = 'GRASP_NOT'
        # self.system_state_pub.publish(state_msg)
        # Sleep for 0.5 seconds
        rospy.sleep(rospy.Duration(secs=0, nsecs=5e8))
        self.set_grasp_state('OPEN')
        # Reset OptoForce on current value (object should have already been taken by the human)
        self.reset_OptoForce_absolute()
        # Rectivate z-axis compliance with the same value as before
        self.set_virtual_compliance(Kf_ext_z=Kf_ext_z_old)
    
    def grab(self):
        Kf_ext_z_old = self.ExtEffDynClient.get_configuration()['Kf_ext_z']
        # Remove z-axis compliance and coupling effect?
        self.set_virtual_compliance(Kf_ext_z=0.0)
        # Open hand
        self.send_grasp_pose(self.grasp_type, 80, 200)
        rospy.sleep(rospy.Duration(secs=2, nsecs=0))
        self.set_grasp_state('CLOSED')
        # Reset OptoForce on current value (object should have already been taken by the human)
        self.reset_OptoForce_absolute()
        # Rectivate z-axis compliance with the same value as before
        self.set_virtual_compliance(Kf_ext_z=Kf_ext_z_old)
    
    def check_release_condition(self):
        return (np.linalg.norm(self.w_ext['lin']) > self.F_TRIGGER) and (self.action_state=='GRASP_ACTIVE') and (self.grasp_state=='CLOSED') and (self.system_state=='READY')
    
    def check_closing_condition(self):
        return (np.linalg.norm(self.w_ext['lin']) > self.F_TRIGGER) and (self.action_state=='GRASP_ACTIVE') and (self.grasp_state=='OPEN') and (self.system_state=='READY')
        
    def publish_DMP_goal(self, pose):
        msg = Pose()
        msg.position.x = pose['lin'][0]
        msg.position.y = pose['lin'][1]
        msg.position.z = pose['lin'][2]
        msg.orientation.w = pose['ang'].w
        msg.orientation.x = pose['ang'].x
        msg.orientation.y = pose['ang'].y
        msg.orientation.z = pose['ang'].z
        self.DMP_goal_pub.publish(msg)

    def dyn_reconfigure_callback(self, config, level):
        self.F_TRIGGER = config['F_TRIGGER']
        return config


if __name__ == '__main__':
    try:
        NODE_NAME='DMP_controller'
        # Start node and publishers
        rospy.init_node(NODE_NAME)
        # Set rate
        HZ = rospy.get_param('/Release_Controller/hz', default=100.0)
        rate = rospy.Rate(HZ)
        ### Params
        grasp_params_dict = {'grasp_type': "CYL"}
        w_ext_params_dict = {'fc': 25.0}
        ### Set control tester
        rc = ReleaseController(grasp_params_dict, w_ext_params_dict)
        t = 0.0
        t_curr = rospy.get_time()
        t_last = rospy.get_time()
        dt = 1.0/HZ
        # # Set correct initial state, to avoid initial motion when starting up the robot
        # while (t<10.0) and (not rospy.is_shutdown()):
        #     # Manage time
        #     t_curr = rospy.get_time()
        #     dt_last = t_curr - t_last
        #     t += dt_last
        #     t_last = t_curr
        #     print(t)
        #     rate.sleep()
        while not rospy.is_shutdown():
            # Manage time
            t_curr = rospy.get_time()
            dt_last = t_curr - t_last
            t += dt_last
            t_last = t_curr
            # Check release condition
            if rc.check_release_condition():
                rc.release()
            if rc.check_closing_condition():
                rc.grab()
            # Sleep
            t_last = t_curr
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()


