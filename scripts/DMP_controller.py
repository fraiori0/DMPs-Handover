#!/usr/bin/env python 
import os
import numpy as np
import scipy as sp
import quaternion as quat
import rbdl
import joblib as jbl
from math                   import pi, sin, cos
from urdf_parser_py.urdf    import URDF
#
import roslib
import rospy
import tf2_ros
from std_msgs.msg           import Float64, Bool, String
from geometry_msgs.msg      import WrenchStamped, Pose, Point, Transform, Twist
from sensor_msgs.msg        import JointState
from nav_msgs.msg           import Path
from trajectory_msgs.msg    import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
from ur5_control.msg        import ExternalWrench
#

#
from DMP_R2H import DMP_R2H
from OrientationDynamics import OrientationDynamics

# Dynamic reconfigure
from dynamic_reconfigure.server import Server
from ur5_control.cfg import DMPParamsConfig, DMPExternalEffectsConfig
# from ur5_control.cfg import ControlTesterConfig

### geometry_msgs/WrenchStamped
# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# geometry_msgs/Wrench wrench
#   geometry_msgs/Vector3 force
#     float64 x
#     float64 y
#     float64 z
#   geometry_msgs/Vector3 torque
#     float64 x
#     float64 y
#     float64 z

### sensor_msgs/JointState
# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# string[] name
# float64[] position
# float64[] velocity
# float64[] effort
#### Order ---> name: [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]

### trajectory_msgs/JointTrajectoryPoint
# float64[] positions
# float64[] velocities
# float64[] accelerations
# float64[] effort
# duration time_from_start

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
    def filter(self, dt, signal):
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
                rospy.logwarn('Discrete_Low_Pass_VariableStep : dt too big for the selected cut-off frequency, aliasing may occur')
        # input signal should be a NUMPY ARRAY
        self.x = (1-dt*self.fc)*self.x + self.K*dt*self.fc * signal
        self.xd = (signal-self.x)/float(dt)
        return self.x, self.xd

class DMP_Controller:
    """Generate a trajectory using dynamical systems.
        A goal pose (psotion and orientation) is given through the appropriate topic, and the sistem generates 
        a trajectory that smoothly converges toward the goal
    """
    def __init__(self, URDF_PATH, DMP_params_dict, DMP_gains_dict, DMP_initial_state_dict, DMP_sigmoid_dict, DMP_obstacles_tuple, OD_params_dict, w_ext_params_dict, frame_id='base', joint_id=6):
        ### Parameters
        self.frame_id = frame_id
        ### Create internal robot model for kinematics and dynamics
        #self.model      = rbdl.loadModel(URDF_PATH)
        #print(self.model)
        self.NQ         = 6 #self.model.q_size
        self.JOINT_ID   = joint_id
        # EE position w/ respect to JOINT_ID link, in local coordinates
        self.EE_LOCAL_POS = np.array((0.0, 0.0, 0.045))
        ### State
        self.q  = np.zeros(self.NQ)
        self.qd = np.zeros(self.NQ)
        #Ra = rbdl.CalcBodyWorldOrientation(self.model,self.q,self.JOINT_ID)
        # xa is a quaternion describing the orientation of the end effector
        xa = np.quaternion(1,0,0,0) #quat.from_rotation_matrix(Ra)
        # xl describe the current position of the end effector
        xl = np.zeros(3) #rbdl.CalcBodyToBaseCoordinates(self.model,self.q,self.JOINT_ID,self.EE_LOCAL_POS)
        self.x = {
            'lin': xl,
            'ang': xa, # quaternion (w,x,y,z)
            't'  : 0.0
        }
        self.xd = {
            'lin':np.zeros(3),
            'ang':np.zeros(3),
            't'  : 0.0
        }
        self.xdd = {
            'lin':np.zeros(3),
            'ang':np.zeros(3),
            't'  : 0.0
        }
        self.xd_lp_filters = {
            'lin': Discrete_Low_Pass_VariableStep(dim=3, fc=20.0),
            'ang': Discrete_Low_Pass_VariableStep(dim=3, fc=20.0)
        }
        self.q_des  = np.zeros(self.NQ)
        self.qd_des = np.zeros(self.NQ)
        self.eq = self.q_des - self.q
        self.ex = {
            'lin':np.zeros(3),
            'ang':np.zeros(3)
        }
        # Wrench
        self.f = np.zeros(6)
        # Mass of the Azzurra hand (or everything that is permanently mounted after the OptoForce)
        self.HAND_MASS = w_ext_params_dict['hand_mass']
        # Estimated mass attached to the OptoForce sensor (from the gravity component of the Optoforce + mass of the Azzurra hand)
        self.EE_mass = self.HAND_MASS
        print(self.EE_mass)
        ### DMP (params are overwritten by the dyn config server)
        self.DMP = DMP_R2H(**DMP_params_dict)
        self.DMP.set_gains(**DMP_gains_dict)
        self.DMP.set_sigmoid(**DMP_sigmoid_dict)
        self.y = self.DMP.set_initial_state(**DMP_initial_state_dict)
        self.yd = np.zeros(self.y.shape[0])
        fnl_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'forcing_term_models',
            'fnl_KNNR_minimum_jerk.joblib'
        )
        self.forcing_term_model = jbl.load(fnl_path)
        self.Kf_ext_xy = 0.0 # force scaling on x-y plane
        self.Kf_ext_z = 0.0 # force scaling on z axis
        self.obstacles_tuple = obstacles_tuple # ({'position': np.array(), 'threshold':0.5, 'K':80.0}, ...)
        ### Dynamic system for orientation
        self.OD = OrientationDynamics(**OD_params_dict)
        self.o = OD_params_dict['q0']
        ### External wrench low_pass filters (one for force and one for torque)
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
        # External wrench variables, referred to an absolute (Base) and a relative (EE) reference frame
        self.w_ext = {
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.w_ext_rel = {
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        self.w_ext_rel_total = { # force measured by the sensor
            'lin': np.zeros(3),
            'ang': np.zeros(3),
            't'  : 0.0
        }
        # torque removed to zero-out OptoForce reading, constant in base reference frame (see self.update_external_wrench())
        self.w_ext0_abs = { 
            'lin': np.zeros(3),
            'ang': np.zeros(3)
        } 
        # torque removed to zero-out OptoForce reading, constant in OptoForce reference frame (see self.update_external_wrench())
        self.w_ext0_rel = {
            'lin': np.zeros(3),
            'ang': np.zeros(3)
        } 
        ### Rotation matrix
        self.REEOpt = quat.as_rotation_matrix(np.quaternion( cos(pi/4.0), 0, 0, sin(pi/4) )) #fixed, to be set manually, depends on how the OptoForce sensor is mounted
        self.REE = quat.as_rotation_matrix(np.quaternion(1,0,0,0))
        ### Release system state
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
        ### Publishers
        # self.target_wrench_pub = rospy.Publisher('/my_cartesian_vforce_controller/target_wrench', WrenchStamped, queue_size=1)
        self.target_traj_pub = rospy.Publisher('/my_cartesian_eik_controller/target_traj', MultiDOFJointTrajectoryPoint, queue_size=10)
        self.DMP_internal_goal_pub = rospy.Publisher('/DMP/internal_goal', Point, queue_size=5)
        self.DMP_internal_pos_pub = rospy.Publisher('/DMP/internal_position', Point, queue_size=5)
        self.DMP_internal_hh_dist_pub = rospy.Publisher('/DMP/internal_human_hand_dist', Float64, queue_size=5)
        self.external_wrench_pub = rospy.Publisher('/DMP/external_wrench', ExternalWrench, queue_size=5)
        self.action_state_pub = rospy.Publisher('/state_machine/action_state', String, queue_size=5)
        self.system_state_pub = rospy.Publisher('/state_machine/system_state', String, queue_size=5)
        # Test publisher, can be used to test functions
        self.test_publisher   = rospy.Publisher('/DMP/test', Point, queue_size=5)
        self.test_publisher_2 = rospy.Publisher('/DMP/test2', Point, queue_size=5)
        # Path for RVIZ visualization
        self.xdes_path_pub = rospy.Publisher(name='/DMP_controller/xdes_path',data_class=Path,queue_size=5)
        self.xdes_path_msg = Path()
        self.xdes_path_msg.header.frame_id = 'base'
        ### Subscribers
        self.robot_state_sub            = rospy.Subscriber('/joint_states', JointState, self.update_robot_state)
        self.robot_des_state_sub        = rospy.Subscriber('/my_cartesian_eik_controller/joint_des_cmd', JointTrajectoryPoint, self.update_robot_q_des_state)
        self.robot_ee_des_state_sub     = rospy.Subscriber('/my_cartesian_eik_controller/ee_des_traj', MultiDOFJointTrajectoryPoint, self.update_robot_ee_des_state)
        self.hmn_dist_sub               = rospy.Subscriber(name='/human_data/goal_distance', data_class=Float64, callback=self.update_human_hand_distance, queue_size=2)
        self.DMP_target_goal_sub        = rospy.Subscriber(name='/DMP/target_goal', data_class=Pose, callback=self.update_goal, queue_size=2)
        self.DMP_external_force_sub     = rospy.Subscriber(name='/ethdaq_data', data_class=WrenchStamped, callback=self.update_external_wrench, queue_size=1)
        self.OptoForce_reset_sub        = rospy.Subscriber(name='/reset_OptoForce_absolute', data_class=Bool, callback=self.reset_OptoForce_absolute, queue_size=2)
        self.action_state_sub           = rospy.Subscriber('/state_machine/action_state', String, callback=self.update_action_state, queue_size=5)
        self.system_state_sub           = rospy.Subscriber('/state_machine/system_state', String, callback=self.update_system_state, queue_size=5)
        # self.tfBuffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        ### Dynamic reconfigure server
        self.dyn_srv = Server(DMPParamsConfig, self.dyn_reconfigure_callback)
        self.dyn_srv_ext_effects = Server(DMPExternalEffectsConfig, self.dyn_reconfigure_callback_external_effects, namespace='/DMP_controller/set_parameters_ext_effects')
    
    def update_robot_state(self, joint_state):
        ### Order of joints, from topic joint_state
        # name: [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        self.q = np.array((
                joint_state.position[joint_state.name.index('shoulder_pan_joint')],
                joint_state.position[joint_state.name.index('shoulder_lift_joint')],
                joint_state.position[joint_state.name.index('elbow_joint')],
                joint_state.position[joint_state.name.index('wrist_1_joint')],
                joint_state.position[joint_state.name.index('wrist_2_joint')],
                joint_state.position[joint_state.name.index('wrist_3_joint')],
        ))
        self.qd = np.array((
                joint_state.velocity[joint_state.name.index('shoulder_pan_joint')],
                joint_state.velocity[joint_state.name.index('shoulder_lift_joint')],
                joint_state.velocity[joint_state.name.index('elbow_joint')],
                joint_state.velocity[joint_state.name.index('wrist_1_joint')],
                joint_state.velocity[joint_state.name.index('wrist_2_joint')],
                joint_state.velocity[joint_state.name.index('wrist_3_joint')],
        ))
        self.update_error_q()
    
    def update_robot_q_des_state(self, joint_traj_point):
        #### Order of the message ---> name: [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        self.q_des = np.array((joint_traj_point.positions))
        self.qd_des = np.array((joint_traj_point.velocities))
        ### Update end effector desired position and task-space error
        self.update_error_q()
    
    def update_robot_ee_des_state(self, ee_traj_point):
        #t = float(ee_traj_point.header.stamp.secs) + 1.0e-9 * float(ee_traj_point.header.stamp.nsecs)
        t = rospy.get_time()
        dt = t - self.x['t']
        #
        self.x['lin'] = np.array((
            ee_traj_point.transforms[0].translation.x,
            ee_traj_point.transforms[0].translation.y,
            ee_traj_point.transforms[0].translation.z
        ))
        self.x['ang'] = np.quaternion(
            ee_traj_point.transforms[0].rotation.w,
            ee_traj_point.transforms[0].rotation.x,
            ee_traj_point.transforms[0].rotation.y,
            ee_traj_point.transforms[0].rotation.z
        )
        self.x['t'] = t
        self.REE = quat.as_rotation_matrix(self.x['ang'])
        ###
        self.xd['lin'] = np.array((
            ee_traj_point.velocities[0].linear.x,
            ee_traj_point.velocities[0].linear.y,
            ee_traj_point.velocities[0].linear.z
        ))
        self.xd['ang'] = np.array((
            ee_traj_point.velocities[0].angular.x,
            ee_traj_point.velocities[0].angular.y,
            ee_traj_point.velocities[0].angular.z
        ))
        self.xd['t'] = t
        ###
        _, self.xdd['lin'] = self.xd_lp_filters['lin'].filter(dt, self.x['lin'])
        _, self.xdd['ang'] = self.xd_lp_filters['ang'].filter(dt, self.x['ang'])
        self.xdd['t'] = t
        test_msg = Point()
        test_msg.x = self.xdd['lin'][0]
        test_msg.y = self.xdd['lin'][1]
        test_msg.z = self.xdd['lin'][2]
        self.test_publisher.publish(test_msg)
        # Update error and DMP internal error
        self.ex['lin'] = self.y[self.DMP.X_INDEX] - self.x['lin']
        self.ex['ang'] = np.zeros(3)
        self.DMP.EX = self.ex['lin'].copy()
        # Change system state if EE's position is close to the final DMP's goal
        if (np.linalg.norm(self.x['lin'] - self.y[self.DMP.X_INDEX]) < self.d_thresh) and (self.action_state=='GRASP_NOT_ACTIVE') and (self.system_state=='READY'):
            state_msg = String()
            state_msg.data = 'GRASP_ACTIVE'
            self.action_state_pub.publish(state_msg)
    
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
    
    def stay(self):
        self.DMP.set_goal(self.x['lin'])
    
    def hold(self):
        target_xd = {
            'lin': np.zeros(3),
            'ang': np.zeros(3)
        }
        self.publish_target_traj(self.x, target_xd)

    def update_error_q(self):
        self.eq = self.q_des - self.q

    def update_human_hand_distance(self, msg):
        self.DMP.D = msg.data
    
    def update_goal(self, msg):
        state_msg = String()
        state_msg.data = 'GRASP_NOT_ACTIVE'
        self.action_state_pub.publish(state_msg)
        ### DMP
        G = np.array((msg.position.x, msg.position.y, msg.position.z))
        self.DMP.set_goal(G)
        # reset x0 to the current position
        self.DMP.x0 = self.y[self.DMP.X_INDEX].copy()
        # also reset DMP phase
        self.y[self.DMP.S_INDEX] = 1.0
        # and hand distance
        self.DMP.D0 = self.y[self.DMP.D_INDEX]
        ### OD
        qg = (np.quaternion(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)).normalized()
        self.OD.set_goal(qg)
    
    def update_external_wrench(self, wrench_msg):
        # w_ext_rel_total is the current reading of the OptoForce, given in the reference frame of the OptoForce sensor
        t  = float(wrench_msg.header.stamp.secs) + 1.0e-9 * float(wrench_msg.header.stamp.nsecs)
        dt = t - self.w_ext_rel_total['t']
        self.w_ext_rel_total = {
            'lin': np.array((wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z)),
            'ang': np.array((wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z)),
            't'  : t
        }
        self.w_ext['t']   = t
        if (dt > 100.0) :
            return 
        #print(dt)
        # Predict inertial forces from the mass attached at the end of the OptoForce
        # inertial_force_abs = self.predict_inertial_force()
        # test_msg = Point()
        # test_msg.x = inertial_force_abs[0]
        # test_msg.y = inertial_force_abs[1]
        # test_msg.z = inertial_force_abs[2]
        # self.test_publisher.publish(test_msg)
        # change of coordinates to express the wrench in base coordinates, and remove w_etx0 components (both w_ext0 rel. and abs.)
        self.w_ext['lin'] = ( self.REE.dot(self.REEOpt.dot(self.w_ext_rel_total['lin'] - self.w_ext0_rel['lin']))) - self.w_ext0_abs['lin']# - inertial_force_abs
        self.w_ext['ang'] = ( self.REE.dot(self.REEOpt.dot(self.w_ext_rel_total['ang'] - self.w_ext0_rel['ang']))) - self.w_ext0_abs['ang']
        self.w_ext['t']   = t

        self.w_ext_rel['lin'] = self.REEOpt.dot(self.w_ext_rel_total['lin'] - self.w_ext0_rel['lin']) - self.REE.T.dot(self.w_ext0_abs['lin'])# + inertial_force_abs)
        self.w_ext_rel['ang'] = self.REEOpt.dot(self.w_ext_rel_total['ang'] - self.w_ext0_rel['ang']) - self.REE.T.dot(self.w_ext0_abs['ang'])
        self.w_ext_rel['t']   = t

        # Publish unfiltered signal
        self.publish_external_wrench(self.w_ext_rel, self.w_ext)

        # Filter signals for internal use
        self.w_ext['lin'], _ = self.lp_filters['w_ext']['lin'].filter(dt, self.w_ext['lin'])
        self.w_ext['ang'], _ = self.lp_filters['w_ext']['ang'].filter(dt, self.w_ext['ang'])
        self.w_ext_rel['lin'], _ = self.lp_filters['w_ext_rel']['lin'].filter(dt, self.w_ext_rel['lin'])
        self.w_ext_rel['ang'], _ = self.lp_filters['w_ext_rel']['ang'].filter(dt, self.w_ext_rel['ang'])

        #print(self.EE_mass * np.array([0.0, 0.0, -9.81]))
        #print(self.REEOpt.T.dot(self.REE.T.dot(np.eye(3))))


    def reset_zero_external_wrench_relative(self):
        # Set current value of the external wrench as zero [RELATIVE]
        # Should be done only ONCE, at the start.
        # The purpose it's to zero-out forces generated by a tight mounting of the optoforce sensor
        self.w_ext0_rel['lin'] = self.w_ext_rel_total['lin'].copy() - self.REEOpt.T.dot(self.REE.T.dot(self.EE_mass * np.array([0.0, 0.0, -9.81])))
        self.w_ext0_rel['ang'] = self.w_ext_rel_total['ang'].copy()
        print(self.w_ext0_rel['lin'])

    def reset_zero_external_wrench_absolute(self):
        # CALL THE "RELATIVE" VERSION OF THIS FUNCTION AT LEAST ONCE BEFORE CALLING THIS!
        # Set current value of the external wrench as zero [ABSOLUTE]
        self.w_ext0_abs['lin'] = self.w_ext0_abs['lin'] + self.w_ext['lin'].copy()
        self.w_ext0_abs['ang'] = self.w_ext0_abs['ang'] + self.w_ext['ang'].copy()
        self.EE_mass = (-self.w_ext0_abs['lin'][2] / 9.81)
        print("Reset absolute OptoForce reading to: ")
        print(self.w_ext0_abs['lin'])
        print("Estimated EE mass: %f" %self.EE_mass)
    
    def reset_OptoForce_absolute(self, msg):
        if msg.data:
            self.reset_zero_external_wrench_absolute()
    
    def publish_external_wrench(self, wrench_rel, wrench_abs):
        wrench_msg = ExternalWrench()

        wrench_msg.absolute.force.x = wrench_abs['lin'][0]
        wrench_msg.absolute.force.y = wrench_abs['lin'][1]
        wrench_msg.absolute.force.z = wrench_abs['lin'][2]
        wrench_msg.absolute.torque.x = wrench_abs['ang'][0]
        wrench_msg.absolute.torque.y = wrench_abs['ang'][1]
        wrench_msg.absolute.torque.z = wrench_abs['ang'][2]

        wrench_msg.relative.force.x = wrench_rel['lin'][0]
        wrench_msg.relative.force.y = wrench_rel['lin'][1]
        wrench_msg.relative.force.z = wrench_rel['lin'][2]
        wrench_msg.relative.torque.x = wrench_rel['ang'][0]
        wrench_msg.relative.torque.y = wrench_rel['ang'][1]
        wrench_msg.relative.torque.z = wrench_rel['ang'][2]

        wrench_msg.time.data = float(wrench_abs['t'])

        self.external_wrench_pub.publish(wrench_msg)
    
    def publish_target_traj(self, x_des, xd_des):
        traj_msg = MultiDOFJointTrajectoryPoint()
        trans = Transform()
        vel = Twist()
        ###
        trans.translation.x = x_des['lin'][0]
        trans.translation.y = x_des['lin'][1]
        trans.translation.z = x_des['lin'][2]
        trans.rotation.w = x_des['ang'].w
        trans.rotation.x = x_des['ang'].x
        trans.rotation.y = x_des['ang'].y
        trans.rotation.z = x_des['ang'].z
        ### xd_des
        vel.linear.x = xd_des['lin'][0]
        vel.linear.y = xd_des['lin'][1]
        vel.linear.z = xd_des['lin'][2]
        vel.angular.x = xd_des['ang'][0]
        vel.angular.y = xd_des['ang'][1]
        vel.angular.z = xd_des['ang'][2]
        ###
        traj_msg.transforms.append(trans)
        traj_msg.velocities.append(vel)
        self.target_traj_pub.publish(traj_msg)
    
    def publish_DMP_internal_goal(self):
        int_g = Point()
        int_g.x = self.y[self.DMP.G_INDEX][0]
        int_g.y = self.y[self.DMP.G_INDEX][1]
        int_g.z = self.y[self.DMP.G_INDEX][2]
        self.DMP_internal_goal_pub.publish(int_g)

    def publish_DMP_internal_pos(self):
        int_pos = Point()
        int_pos.x = self.y[self.DMP.X_INDEX][0]
        int_pos.y = self.y[self.DMP.X_INDEX][1]
        int_pos.z = self.y[self.DMP.X_INDEX][2]
        self.DMP_internal_pos_pub.publish(int_pos)
    
    def publish_DMP_internal_human_hand_dist(self):
        dist = Float64()
        dist.data = self.y[self.DMP.D_INDEX]
        self.DMP_internal_hh_dist_pub.publish(dist)

    def set_lim_cartesian(self, xlim, ylim, zlim):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
    
    def set_frame_id(self, frame_id):
        # Not useful, but can help in keeping code clean
        self.frame_id = frame_id
    
    def step_DMP(self, t, dt):
        # DMP should be stepped as usual, but setting x and xd to the current actual values? Maybe not really
        # self.y[self.DMP.X_INDEX]  = self.x['lin']
        # self.y[self.DMP.XD_INDEX] = self.xd['lin']
        f_ext = np.zeros(3)
        if self.virt_comp_active :
            f_ext[0] = self.Kf_ext_xy * self.w_ext['lin'][0]
            f_ext[1] = self.Kf_ext_xy * self.w_ext['lin'][1]
            f_ext[2] = self.Kf_ext_z  * self.w_ext['lin'][2]
        for obs in self.obstacle_tuple:
            f_ext = f_ext + self.DMP.obstacle_avoidance_force_Skeletor(obs['position'], self.y, threshold=obs['threshold'], K=obs['k'])
        self.y, self.yd, self.f[0:3] = self.DMP.integrate_Euler_feq(self.y, t, dt, f_ext=f_ext)
        return self.y, self.f[0:3]
    
    def predict_inertial_force(self):
        # NOTE the minus sign, we want to predict the force received by the OptoForce
        return -self.EE_mass * self.yd[self.DMP.XD_INDEX]
    
    def step_OD(self, dt):
        self.o, w = self.OD.step(self.o, dt)
        return self.o.copy(), w

    def dyn_reconfigure_callback(self, config, level):
        self.DMP.set_parameters(
            TAU_0=config['TAU_0'], 
            ALPHA_X=config['ALPHA_X'],
            ALPHA_S=config['ALPHA_S'],
            ALPHA_G=config['ALPHA_G'],
            ALPHA_D=config['ALPHA_D'],
            ALPHA_E=config['ALPHA_E']
        )
        self.OD.set_parameters(
            K = config['K_OrntDyn'],
            fc = config['fc_OrntDyn']
        )
        return config
    
    def dyn_reconfigure_callback_external_effects(self, config, level):
        self.DMP.set_gains(
            Kcd = config['Kcd'],
            Kce = config['Kce'],
            Ktd = config['Ktd'],
            Kte = config['Kte']
        )
        self.d_thresh = config['d_thresh']
        self.Kf_ext_xy = config['Kf_ext_xy']
        self.Kf_ext_z = config['Kf_ext_z']
        self.DMP.set_sigmoid(
            steepness=config['sigm_steep'],
            offset=config['sigm_off']
        )
        self.virt_comp_active = config['Virt_Comp']
        return config


if __name__ == '__main__':
    try:
        NODE_NAME='DMP_controller'
        ### Start node and publishers
        rospy.init_node(NODE_NAME)
        ### Set rate
        HZ = rospy.get_param('/DMP_controller/hz', default=100.0)
        rate = rospy.Rate(HZ)
        ### Generate a urdf fro the robot description
        # The robot_description is uploaded directly from xacro file, but a URDF is needed for the RBDL internal model
        robot_description = rospy.get_param('/robot_description')
        URDF_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'urdf', 'tmp_model_no_world.urdf')
        # with open(URDF_PATH, "w") as urdf_file:
        #     urdf_file.write(robot_description)
        #     urdf_file.close()
        ### DMP Parameters
        DMP_params_dict = {
            'TAU_0':    0.3,
            'ALPHA_X':  0.0,
            'ALPHA_S':  1.0,
            'ALPHA_G':  5.0,
            'ALPHA_D':  10.0,
            'ALPHA_E':  5.0
        }
        DMP_gains_dict={
            'Kcd':  0.0, #10.0,
            'Kce':  0.0, # 1000.0,#1000.0,
            'Ktd':  0.0, #10.0,
            'Kte':  0.0, #100.0 #10.0,#1000.0
        }
        DMP_sigmoid_dict = {
            'steepness' : 1.0,
            'offset'    : 0.0
        }
        DMP_initial_state_dict={
            's0':   1.0,
            'x0':   np.zeros(3),
            'xd0':  np.zeros(3),#np.random.rand(3)-0.5,
            'D0':   1.0,
            'Dd0':  0.0,
            'EX0':  np.zeros(3)
        }
        DMP_obstacles_tuple = (
            {'position': np.array((0.0,0.0,0.4)), 'threshold': 0.5, 'K':80.0}
        )
        OD_params_dict = {
            'dt': 1.0/HZ,
            'K' : 6.0,
            'fc': 8.0,
            'q0': np.quaternion(1.0, 0.0, 0.0, 0.0)
        }
        w_ext_params_dict = {
            'fc': 20.0,
            'hand_mass': 0.64
        }
        ### Set control tester
        ct = DMP_Controller(
            URDF_PATH=              URDF_PATH,
            DMP_params_dict=        DMP_params_dict,
            DMP_gains_dict=         DMP_gains_dict,
            DMP_initial_state_dict= DMP_initial_state_dict,
            DMP_sigmoid_dict=       DMP_sigmoid_dict,
            DMP_obstacles_tuple=    DMP_obstacles_tuple,
            OD_params_dict=         OD_params_dict,
            w_ext_params_dict=      w_ext_params_dict,
            frame_id=               'base',
            joint_id=               6
        )
        ### DMP Forcing term and (initial) goal
        # Set initial goal, then it will be internally updated from received messages
        G = np.array((-0.4,0.4,0.7))
        ct.DMP.set_goal(G)
        my_fnl = lambda t: 0.0 * 0.02 * np.sin(np.array((0.5,0.5,0.5))*t)
        my_fnl = lambda s: ct.forcing_term_model.predict(np.array(s).reshape(-1,1))
        ct.DMP.set_fnl(my_fnl)
        ct.DMP.force_fun_gen()
        ### Orientation dynamic controller
        ct.OD.set_goal(np.quaternion(1.0, 0.0, 0.0, 0.0))
        ### Loop node
        my_wrench = {
            'lin': np.zeros(3),
            'ang': np.zeros(3)
        }
        target_x = {
            'lin': np.array((0.0,0.0,2.0)),
            'ang': np.quaternion(1,0,0,0)
        }
        target_xd = {
            'lin': np.zeros(3),
            'ang': np.zeros(3)
        }
        t = 0.0
        t_curr = rospy.get_time()
        t_last = rospy.get_time()
        dt = 1.0/HZ
        # Set correct DMP initial state, to avoid initial motion when starting up the robot
        while (t<10.0) and (not rospy.is_shutdown()):
            # Manage time
            t_curr = rospy.get_time()
            dt_last = t_curr - t_last
            t += dt_last
            t_last = t_curr
            if (t%0.2) < 0.02 :
                print(t)
            ct.y[ct.DMP.X_INDEX] = ct.x['lin'].copy()
            #print(ct.y[ct.DMP.X_INDEX])
            ct.y[ct.DMP.XD_INDEX]= np.zeros(3)
            #target_x['ang'] = ct.x['ang']
            ct.o = ct.x['ang']
            # Set goal to the real robot position, to avoid initial movements at the start-up
            ct.DMP.set_goal(ct.y[ct.DMP.X_INDEX])
            ct.DMP.x0 = ct.y[ct.DMP.X_INDEX].copy()
            ct.OD.set_goal(ct.o.copy())
            #
            ct.hold()
            ct.publish_DMP_internal_goal()
            ct.publish_DMP_internal_pos()
            ct.publish_DMP_internal_human_hand_dist()
            rate.sleep()
        # Reset OptoForce readings
        print('\n\n Reset OptoForce reading to')
        print('Relative')
        ct.reset_zero_external_wrench_relative()
        # Wait for self.w_ext to be updated
        tmp = 0.0
        while (tmp<1.0) and (not rospy.is_shutdown()):
            t_curr = rospy.get_time()
            dt_last = t_curr - t_last
            tmp += dt_last
            t_last = t_curr
        print('Absolute')
        ct.reset_zero_external_wrench_absolute()
        system_state_msg = String()
        system_state_msg.data = 'READY'
        ct.system_state_pub.publish(system_state_msg)
        while not rospy.is_shutdown():
            # Manage time
            t_curr = rospy.get_time()
            dt_last = t_curr - t_last
            t += dt_last
            t_last = t_curr
            # Compute and apply control inputs
            # NOTE: THERE IS A MINUS SIGN
            y, my_wrench['lin'] = ct.step_DMP(t, dt)
            o, w = ct.step_OD(dt)
            #ct.publish_target_wrench(my_wrench)
            target_x['lin']  = y[ct.DMP.X_INDEX]
            target_xd['lin'] = y[ct.DMP.XD_INDEX]
            target_x['ang']  = o
            target_xd['ang'] = w
            ct.publish_target_traj(target_x, target_xd)
            ct.publish_DMP_internal_goal()
            ct.publish_DMP_internal_pos()
            ct.publish_DMP_internal_human_hand_dist()
            #pose_msg = generate_random_PoseStamped(x_lim, y_lim, z_lim, 0.5, 0.8, frame_id)
            # Sleep
            t_last = t_curr
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
