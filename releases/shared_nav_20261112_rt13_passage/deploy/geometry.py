"""Minimal attitude transforms; no SciPy runtime dependency on the robot."""
import numpy as np


def quaternion_matrix_xyzw(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('Invalid quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def yaw_matrix(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
