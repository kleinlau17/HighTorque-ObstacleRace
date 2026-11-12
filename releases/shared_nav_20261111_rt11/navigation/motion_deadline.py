"""Limit continuous zero output during an active RT11 navigation mission."""
import numpy as np


class MotionDeadline:
    # The 50 Hz publisher has 120 ms of scheduling margin before two seconds.
    WAIT_S = 1.88
    TURN_S = .40
    RAW_YAW = .8

    def __init__(self):
        self.started=None
        self.zero_since=None
        self.until=None
        self.direction=1.
        self.next_direction=None
        self.recovering=False

    def output(self,stamp,raw,active,direction=0.):
        """Called by the sole /cmd_vel owner, after the factory-input adapter."""
        vector=np.asarray(raw,dtype=float)
        if not active:
            self.started=None
            self.zero_since=None
            self.until=None
            self.next_direction=None
            self.recovering=False
            return vector
        if self.started is None:
            self.started=stamp
            self.zero_since=stamp
        if np.any(vector):
            # A current lane, side-step, or look command always takes priority.
            self.until=None
            self.next_direction=None
            self.recovering=False
            return vector
        if self.until is not None and stamp>=self.until:
            self.until=None
        if self.until is None and self.zero_since is not None and stamp-self.zero_since>=self.WAIT_S:
            self.direction=self.next_direction or np.sign(direction) or self.direction
            self.next_direction=-self.direction
            self.until=stamp+self.TURN_S
        self.recovering=self.until is not None
        if self.recovering:
            return np.array([0.,0.,self.direction*self.RAW_YAW])
        return vector

    def published(self,stamp,actual):
        # Only the actual motor publication can reset the stationary timer.
        if self.started is None:
            return
        if np.any(actual):
            self.zero_since=None
        elif self.zero_since is None:
            self.zero_since=stamp
