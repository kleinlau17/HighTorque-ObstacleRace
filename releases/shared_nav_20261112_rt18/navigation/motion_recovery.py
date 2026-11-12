"""Bounded standing recovery for commanded walking without visual progress."""
import numpy as np


class MotionRecovery:
    def __init__(self):
        self.phase='ready'
        self.stalled_since=None
        self.settle_until=None
        self.recoveries=0
        self.anchor=None

    def update(self,time,desired,pose,velocity,enabled):
        if self.phase=='failed':return 'failed'
        if not enabled:
            self.stalled_since=None
            return self.phase
        if self.anchor is None:self.anchor=np.asarray(pose[:2]).copy()
        if np.linalg.norm(pose[:2]-self.anchor)>.10:
            self.anchor=np.asarray(pose[:2]).copy();self.recoveries=0
        if self.phase=='settle':
            if time<self.settle_until:return 'settle'
            self.phase='ready';self.stalled_since=None
        requested=np.linalg.norm(desired[:2])>.12
        stationary=np.linalg.norm(velocity)<.02
        if requested and stationary:
            if self.stalled_since is None:self.stalled_since=time
            elif time-self.stalled_since>.8:
                self.recoveries+=1
                self.phase='settle' if self.recoveries<=2 else 'failed'
                self.settle_until=time+.5
        else:self.stalled_since=None
        return self.phase
