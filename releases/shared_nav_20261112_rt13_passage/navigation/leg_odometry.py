"""Short-range Pi Plus Pro support-foot odometry from leg encoders and AHRS.

All estimates are relative to the first accepted sample, in AHRS horizontal
axes. Box or lane observations should anchor this local displacement because
foot slip is unobservable here. No simulator or ROS dependency.
"""
from collections import deque
import math
import numpy as np

LEG_JOINTS = tuple(f'{side}_{joint}_joint' for side in ('l','r')
                   for joint in ('hip_pitch','hip_roll','thigh','calf','ankle_pitch','ankle_roll'))

# PiPlusPro_SF_12L8A0G2H0W_Z_261010 URDF leg origins and revolute axes.
_AXES = (1,0,2,1,1,0)
_ORIGINS = np.array([[0,.0474,-.0435],[0,.0341,0],[-.00035,0,-.0924],
                     [0,0,-.05],[0,0,-.14],[.051,0,0]])
# CAD soles have four parallel horizontal capsules of radius 10 mm. In the
# ankle-roll frame their center lines are 40 mm below the ankle.
def _sole_points(side):
    middle = (.077,.0765) if side=='l' else (.0765,.077)
    strips = zip((-.017,-.0272,-.0272,-.017),(-.025,-.009,.009,.025),(.065,*middle,.065))
    return np.array([(cx+dx,cy,-.04) for cx,cy,h in strips for dx in (-h,h)])

_SOLES = [_sole_points(side) for side in ('l','r')]


def _rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis==0: return np.array(((1,0,0),(0,c,-s),(0,s,c)))
    if axis==1: return np.array(((c,0,s),(0,1,0),(-s,0,c)))
    return np.array(((c,-s,0),(s,c,0),(0,0,1)))


def _feet(q, attitude):
    points=[]; rotations=[]
    for side in range(2):
        orient=attitude
        pos=np.zeros(3)
        for i,(axis,xyz) in enumerate(zip(_AXES,_ORIGINS)):
            if side and xyz[1]: xyz=xyz*np.array([1.,-1.,1.])
            pos+=orient@xyz
            orient=orient@_rotation(axis,q[side*6+i])
        points.append(_SOLES[side]@orient.T+pos)
        rotations.append(orient)
    return np.array(points),np.array(rotations)


class ProLegOdometry:
    def __init__(self):
        self.xy=np.zeros(2)
        self.velocity=np.zeros(2)
        self.stance_velocity=np.zeros(2)
        self.last_stance=None
        self._previous=None
        self.last_time=None
        self.samples=deque(maxlen=128)
        self.segment=0

    def update(self,t,positions,world_from_body):
        """Consume one complete synchronised leg JointState and AHRS rotation.

        positions maps each LEG_JOINTS name to its raw angle in radians. Submit
        each unique encoder timestamp once, using its closest measured attitude.
        """
        if self.last_time is not None and t<=self.last_time: return False
        try: q=np.array([positions[n] for n in LEG_JOINTS],dtype=float)
        except (KeyError,TypeError,ValueError): return False
        rot=np.asarray(world_from_body,dtype=float)
        if (not np.isfinite(t) or not np.isfinite(q).all() or rot.shape!=(3,3)
                or not np.isfinite(rot).all() or np.max(abs(rot.T@rot-np.eye(3)))>.02):
            return False
        points,rots=_feet(q,rot)
        dt=None if self.last_time is None else t-self.last_time
        if self._previous is not None and dt<=.13:
            previous,old_rot=self._previous
            # Choose the lowest foot across both endpoints, then the same
            # contacting sole material point at both timestamps. Integrating
            # ankle motion itself would count foot roll as trunk translation.
            z=(previous[:,:,2]+points[:,:,2])*.5
            side=int(np.argmin(z.min(axis=1)))
            point=int(np.argmin(z[side]))
            down=-.005*(old_rot[side,2]+rots[side,2])
            delta=(previous[side,point]-points[side,point]+(old_rot[side]-rots[side])@down)[:2]
            # Ground compliance and touchdown slip cause a jump precisely
            # when the feet exchange support. Bridge this short interval with
            # the last firmly planted foot velocity until the new one settles.
            gap0=previous[1-side,:,2].min()-previous[side,:,2].min()
            gap1=points[1-side,:,2].min()-points[side,:,2].min()
            bridging=(min(gap0,gap1)<.003 and self.last_stance is not None
                      and t-self.last_stance<.12 and np.linalg.norm(self.stance_velocity)>.06)
            if bridging:
                delta=self.stance_velocity*dt
            else:
                self.stance_velocity=delta/dt
                self.last_stance=t
            if np.linalg.norm(delta)<1.5*dt+.005:
                self.xy+=delta
                blend=1.-math.exp(-dt/.14)
                self.velocity+=blend*(delta/dt-self.velocity)
        else:
            self.samples.clear();self.velocity.fill(0.);self.segment+=1
        self._previous=(points,rots);self.last_time=t
        self.samples.append((t,self.xy.copy()))
        return True

    def at(self,t,max_age=.10):
        """Return interpolated XY at image capture, or None across a data gap."""
        if not self.samples or t<self.samples[0][0]-max_age or t>self.last_time+max_age:
            return None
        if t<=self.samples[0][0]: return self.samples[0][1].copy()
        if t>=self.last_time: return self.xy.copy()
        for i in range(len(self.samples)-1,0,-1):
            t0,a=self.samples[i-1]
            if t0<=t:
                t1,b=self.samples[i]
                return a+(b-a)*((t-t0)/(t1-t0))
        return None
