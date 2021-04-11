import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
from math import pi, cos, sin, tan
import time
import rbdl

np.set_printoptions(precision=2)

def SVD(A):
    U,s,Vh = sp.linalg.svd(A)
    S = np.zeros(A.shape)
    Sinv = np.zeros(A.T.shape)
    for i in range(len(s)):
        S[i,i] = s[i]
        Sinv[i,i] = 1.0/(s[i]) #a small value is added, which doesn't modify the algorithm but avoids a division by an (exact) zero and getting an "inf" from NumPy
    return U, S, Sinv, Vh

# 8 DOF (q)
# 3 constraint equations
# x is in R4, so the system is also redundant
# Then let's try with 5DOF

J = np.random.rand(4,8)
A = np.random.rand(3,8)

U, S, Sinv, Vh = SVD(A)
Ainv = Vh.T.dot(Sinv.dot(U.T))
P = np.eye(Ainv.shape[0],A.shape[1]) - Ainv.dot(A)
print(Vh.T.dot(Sinv.dot(U.T.dot(U.dot(S.dot(Vh))))))
print(Vh.T.dot(Vh))
#print(A.dot(Ainv))
Uj, Sj, Sjinv, Vjh = SVD(A)

urdf_path = './urdf/iiwa_gripper_no_world.urdf'
model = rbdl.loadModel(urdf_path)
NQ = model.q_size
q = np.zeros(NQ+1)
qd = np.zeros(NQ+1)
JOINT_ID = 7
EE_LOCAL_POS = np.array((0.0, 0.0, 0.0))


x = 2.0*(np.random.rand(4)-0.5)
x_des = 2.0*(np.random.rand(4)-0.5)
dt = 1.0/240.0
t = 0.0
tf = 10.0
while (t<tf):

    t+=dt



