#!/usr/bin/env python 

import roslib
import rospy
import numpy as np
import scipy as sp
import quaternion as quat
from geometry_msgs.msg import PoseStamped, Pose
from math import pi
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

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

def vec_to_0quat(v):
    #v is supposed to be a np.array with shape (3,)
    return quat.as_quat_array(np.insert(v,0,0.))

def normalize(v):
    v_norm = np.linalg.norm(v)
    if (v_norm < 1e-4):
        return np.array((0.0,0.0,0.0)) 
    else:
        return v/v_norm

def quat_prod(p,q):
    pq = np.quaternion()
    pq.w = p.w * q.w - (p.vec).dot(q.vec)
    pq.vec = p.w * q.vec + q.w * p.vec + np.cross(p.vec, q.vec)
    return pq

def quat_exp(q):
    # Note: returns a quaternion
    print(q.vec)
    v_norm = np.linalg.norm(q.vec)
    n = np.nan_to_num(q.vec/v_norm)
    print(n)
    print(q.w)
    vec_part =  np.exp(q.w) * n * np.sin(v_norm)
    sc_part = np.exp(q.w) * np.cos(v_norm)
    print(vec_part)
    print(sc_part)
    q_exp = np.quaternion(
        sc_part,
        vec_part[0],
        vec_part[1],
        vec_part[2]
    )
    return q_exp

def quat_log(q):
    # Note: Returns a quaternion
    v_norm = np.linalg.norm(q.vec)
    n = np.nan_to_num(q.vec/v_norm)
    sc_part = np.log(q.norm())
    vec_part = np.nan_to_num(n * np.log(q.w / q.norm()))
    q_log = np.quaternion(
        sc_part,
        vec_part[0],
        vec_part[1],
        vec_part[2]
    )
    return q_log

def quat_pw(q,rho):
    theta = np.nan_to_num(np.linalg.norm(quat.as_rotation_vector(q.normalized())))
    n = normalize(q.vec)
    scale = (np.sqrt(q.norm()))**rho
    #theta = np.arccos(q.w)
    sc_part = scale * np.cos(theta*rho)
    vec_part = scale* n * np.sin(theta*rho)
    q_pw = np.quaternion(
        sc_part,
        vec_part[0],
        vec_part[1],
        vec_part[2]
    )
    return q_pw

class OrientationDynamics:
    def __init__(self, dt, K, fc, q0):
        self.low_pass = Discrete_Low_Pass(3, dt, fc)
        self.q = q0.copy()
        self.K = K

    def set_goal(self, qg):
        self.qg = qg.copy()
    
    def step(self, q, dt):
        eq = self.qg * q.conjugate()
        w_des = self.K * eq.vec
        w = self.low_pass.filter(w_des)
        qd = 0.5 * vec_to_0quat(w) * q
        q = (q + qd * dt).normalized()
        return q.copy(), w
    
    def set_parameters(self, **params):
        try :
            self.K = params['K']
        except KeyError as e:
            print(e)
        try :
            self.low_pass.fc = params['fc']
        except KeyError as e:
            print(e)

if __name__ == '__main__':
    ### Parameters
    HZ = 240.0
    OD_params_dict = {
            'dt': 1.0/HZ,
            'K' : 6.0,
            'fc': 8.0,
            'q0': np.quaternion(1.0, 0.0, 0.0, 0.0)
        }
    dt = OD_params_dict['dt']
    q = OD_params_dict['q0'].copy()
    ### Initialize Orientation Dynamics
    od = OrientationDynamics(**OD_params_dict)
    qg = np.quaternion(*list(np.random.rand(4))) # np.quaternion(1,0,0,0) # 
    od.set_goal(qg)
    ### Simulate
    t = 0.0
    tf = 2.0
    e_array = []
    t_array = []
    w_array = []
    while (t<tf) :
        q, w = od.step(q, dt)
        #
        e_array.append((qg * q.conjugate()).vec)
        w_array.append(w)
        t_array.append(t)
        #
        t += dt
    ### Plot simulation
    e_array = np.array(e_array)
    w_array = np.array(w_array)
    t_array = np.array(t_array)
    fig_t,axs_t = plt.subplots(2,3)
    for index in range(0,3):
        axs_t[0,index].plot(
            t_array, e_array[:,index],
            color="xkcd:teal"
            )
        # axs_t[0,index].set(
        #     #ylabel="x[%d] [m]"%index,
        #     xlabel="t [s]",
        #     title='e[%d]'%index
        #     )
        axs_t[1,index].plot(
            t_array, w_array[:,index],
            color="xkcd:dark teal"
            )
        # axs_t[1,index].set(
        #     #ylabel="xd[%d] [m/s]"%index,
        #     xlabel="t [s]",
        #     title='w[%d]'%index
        #     )
    plt.show()

# print("\n")
# #print(quat_exp(q))
# print("\n")
# print(quat_log(q))
# print("\n")
# print(quat_exp(quat_log(q)))
########
