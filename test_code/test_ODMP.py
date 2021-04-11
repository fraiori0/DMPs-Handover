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

# print("\n")
# #print(quat_exp(q))
# print("\n")
# print(quat_log(q))
# print("\n")
# print(quat_exp(quat_log(q)))
########
e_array=[]
w_array=[]
t_array=[]
########
dt = 1.0/1000.0
t = 0.0
tf = 2.0
#
w  = np.zeros(3)
#q0 = np.quaternion(1,2,3,4).normalized()
q0 = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()).normalized()
#qg = np.quaternion(1,0,0,0).normalized()
qg = np.quaternion(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()).normalized()
q  = q0.copy()
wq = np.quaternion(1,0,0,0)
#qd = np.quaternion(3,0,0,0)
# Per imporre una qd che abbia senso
w = np.random.rand(3)
qd = 0.5 * vec_to_0quat(w) * q
qdv = qd.vec
#
q_old = q0.copy()
e_curr  = np.zeros(3)
e_old   = np.zeros(3)
ed_curr = np.zeros(3)
ed_old  = np.zeros(3)
#
tau = 100.0
alpha = 0.0
#
k = 6.0
d = np.sqrt(k/4)
###

w_lowpass = Discrete_Low_Pass(3, dt, 8)
while (t<tf) :
    eq = qg * q.conjugate()
    w_des = k * eq.vec
    w = w_lowpass.filter(w_des)
    ### Integrate
    #wq = 2* eq * q.conjugate()
    #d_q = quat.from_rotation_vector(0.5 * k * dt * quat.as_rotation_vector(wq))
    qd = 0.5 * vec_to_0quat(w) * q
    q = (q + qd * dt).normalized()
    ###
    e_array.append(eq.vec)
    t_array.append(t)
    #w_array.append(quat.as_rotation_vector(wq))
    w_array.append(w)
    #w_array.append(q.vec)
    ###
    t+=dt
qg = np.quaternion(-np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()).normalized()
while (t<2*tf) :
    eq = qg * q.conjugate()
    w_des = k * eq.vec
    w = w_lowpass.filter(w_des)
    ### Integrate
    #wq = 2* eq * q.conjugate()
    #d_q = quat.from_rotation_vector(0.5 * k * dt * quat.as_rotation_vector(wq))
    qd = 0.5 * vec_to_0quat(w) * q
    q = (q + qd * dt).normalized()
    ###
    e_array.append(eq.vec)
    t_array.append(t)
    #w_array.append(quat.as_rotation_vector(wq))
    w_array.append(w)
    #w_array.append(q.vec)
    ###
    t+=dt

### BASE CONCEPT
# while (t<tf) :
#     eq = qg * q.conjugate()
#     ew = 2.0 * qd * q.conjugate()
#     #qdd = quat_pw(ew ,-d) *quat_pw(eq,-k)
#     qdd = quat_pw(eq,-k) * quat_pw(ew ,-d) 
#     #qdd = quat_pw(ew,-d)
#     d_qd = quat_pw(qdd,dt)
#     print(q.norm())
#     qd = d_qd * qd # Check multiplication order
#     d_q = (quat_pw(qd,dt)).normalized()
#     q = q * d_q
#     ###
#     wq = 2 * qd * q.conjugate()
#     ###
#     e_array.append(eq.vec)
#     t_array.append(t)
#     #w_array.append(quat.as_rotation_vector(wq))
#     w_array.append(wq.vec)
#     #w_array.append(q.vec)
#     ###
#     t+=dt
### THis is the same but more elegant, using just the quaternion power (uh yesssss)
# while (t<tf) :
#     eq = qg*(q.conjugate())
#     d_q = quat_pw(eq,dt*k)
#     q = d_q * q
#     ###
#     #wq = 2* qd * q.conjugate()
#     ###
#     e_array.append(eq.vec)
#     t_array.append(t)
#     #w_array.append(quat.as_rotation_vector(wq))
#     ###
#     t+=dt
### THIS WORKS
# while (t<tf) :
#     eq = qg*(q.conjugate())
#     spring_term = quat.from_rotation_vector(k*quat.as_rotation_vector(eq))
#     damping_term = quat.from_rotation_vector(d*quat.as_rotation_vector(qd.conjugate()))
#     qdd = spring_term * damping_term #NOTE: is multiplication correct, to substitute the sum?
#     #wq = qg*q.conjugate()*q.conjugate()
#     #logv = k * quat.as_rotation_vector(wq)
#     d_qd = quat.from_rotation_vector(dt * quat.as_rotation_vector(qdd))
#     qd = qd * d_qd
#     qd = spring_term
#     wq = 2* eq * q.conjugate()
#     d_q = quat.from_rotation_vector(0.5 * k * dt * quat.as_rotation_vector(wq))
#     q = q * d_q
#     ###
#     #wq = 2* qd * q.conjugate()
#     ###
#     e_array.append(eq.vec)
#     t_array.append(t)
#     w_array.append(quat.as_rotation_vector(wq))
#     ###
#     t+=dt
# while (t<tf) :
#     eq = qg*(q.conjugate())
#     spring_term = quat.from_rotation_vector(k*quat.as_rotation_vector(eq))
#     damping_term = quat.from_rotation_vector(d*quat.as_rotation_vector(qd.conjugate()))
#     qdd = spring_term * damping_term #NOTE: is multiplication correct, to substitute the sum?
#     #wq = qg*q.conjugate()*q.conjugate()
#     #logv = k * quat.as_rotation_vector(wq)
#     d_qd = quat.from_rotation_vector(dt * quat.as_rotation_vector(qdd))
#     qd = qd * d_qd
#     qd = spring_term
#     d_q = quat.from_rotation_vector(dt * quat.as_rotation_vector(qd))
#     q = q * d_q
#     ###
#     wq = 2* qd * q.conjugate()
#     ###
#     e_array.append(eq.vec)
#     t_array.append(t)
#     w_array.append(quat.as_rotation_vector(wq))
#     ###
#     t+=dt
# while (t<tf) :
#     e = qg*(q.conjugate())
#     spring_term = quat.from_rotation_vector(k*quat.as_rotation_vector(2.0*e*q.conjugate()))
#     damping_term = quat.from_rotation_vector(d*quat.as_rotation_vector(wq.conjugate()))
#     wqd = spring_term + damping_term + 0.5*wq*wq
#     d_wq = quat.from_rotation_vector(0.5*dt*quat.as_rotation_vector(wqd))
#     wq = wq * d_wq
#     #wq = qg*q.conjugate()*q.conjugate()
#     #logv = k * quat.as_rotation_vector(wq)
#     logv = quat.as_rotation_vector(wq)
#     d_q = quat.from_rotation_vector(0.5*dt*quat.as_rotation_vector(wq))
#     q = q * d_q
#     ###
#     e_array.append(e.vec)
#     t_array.append(t)
#     w_array.append(quat.as_rotation_vector(wq))
#     ###
#     t+=dt
# while (t<tf) :
#     eq = qg*(q.conjugate())
#     e_curr = eq.vec
#     ed_curr = (e_curr - e_old)/dt
#     edd = alpha*((alpha/4)*e_curr + tau*ed_curr)/(tau**2)
#     # Store and update
#     e_old = e_curr.copy()
#     ed_old = ed_curr.copy()
#     #
#     ed_curr = ed_curr + edd * dt
#     e_curr = e_curr + ed_curr*dt
#     # Compute new q and qd, then store q in q_old for next computation
#     q = (quat.from_rotation_vector(e_curr)).conjugate() * qg
#     qd = quat.from_rotation_vector(quat.as_rotation_vector(q*q_old)/dt)
#     q_old = q.copy()
#     wq = 0.5 * qd * q.conjugate()
#     ###
#     e_array.append(eq.vec)
#     t_array.append(t)
#     w_array.append(quat.as_rotation_vector(wq))
#     ###
#     t+=dt

e_array = np.array(e_array)
w_array = np.array(w_array)
t_array = np.array(t_array)
# Plot results
fig_t,axs_t = plt.subplots(2,3)
for index in range(0,3):
    axs_t[0,index].plot(
        t_array, e_array[:,index],
        color="xkcd:teal"
        )
    # axs_t[0,index].set(
    #     ylabel="x[%d] [m]"%index,
    #     xlabel="t [s]",
    #     title='Position'
    #     )
    axs_t[1,index].plot(
        t_array, w_array[:,index],
        color="xkcd:dark teal"
        )
# axs_t[1,0].plot(
#         t_array, np.linalg.norm(e_array,axis=1),
#         color="xkcd:red"
#         )
plt.show()