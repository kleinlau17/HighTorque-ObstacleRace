"""Verify physical motion and request bounded, floor-authorized in-place restarts."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MotionStatus:
    yaw_request: float
    reason: str
    still_s: float
    recovery_due: bool


def _angle(value):
    return (value+math.pi)%(2.*math.pi)-math.pi


def _fresh(now,stamp,lease):
    return stamp is not None and math.isfinite(stamp) and 0.<=now-stamp<=lease


def _xy(value):
    if value is None:
        return None
    v=np.asarray(value,dtype=float)
    return v.copy() if v.shape==(2,) and np.isfinite(v).all() else None


class BoxMotionWatch:
    """One instance per run. Call from the serialized box control path, including zero plans."""
    stall_s=1.20
    retry_s=.72
    burst_s=.32
    burst_angle=.14
    yaw_speed=.76
    view_lease=.38
    imu_lease=.15

    def __init__(self):
        self.last_motion=None
        self.confirmed=False
        self.last_call=None
        self.last_capture=-math.inf
        self.last_direct_stamp=-math.inf
        self.exposures=deque(maxlen=16)
        self.last_burst=-math.inf
        self.burst_yaw=None
        self.burst_sign=0.
        self.attempts=0
        self.legs=deque(maxlen=90)
        self.attitudes=deque(maxlen=90)
        self.captures=deque(maxlen=12)

    @staticmethod
    def _add(history,stamp,value):
        # Bound history by time even on a fast IMU or JointState stream.
        if not history or stamp-history[-1][0]>=.012:
            history.append((stamp,value))
        while history and stamp-history[0][0]>1.08:
            history.popleft()

    def _capture(self,now,stamp,landmark,relative):
        if not _fresh(now,stamp,.55) or stamp<=self.last_direct_stamp or landmark is None:
            return None
        rel=_xy(relative)
        if rel is None:
            return None
        self.last_direct_stamp=stamp
        sample=min(self.legs,key=lambda p:abs(stamp-p[0]),default=None)
        leg=sample[1] if sample is not None and abs(stamp-sample[0])<=.13 else None
        self.captures.append((stamp,landmark,rel,leg))
        if leg is None:
            return None
        # The observed stationary crate moves oppositely to the stance-leg
        # displacement. Neither the estimated map pose nor a speed request is evidence.
        captures=list(self.captures)
        last=len(captures)-1
        for i in range(last-1,-1,-1):
            prior,key,old_rel,old_leg=captures[i]
            dt=stamp-prior
            if key!=landmark or old_rel is None or dt>.72 or (self.confirmed and prior<self.last_motion):
                break
            if last-i<2 or dt<.18 or old_leg is None:
                continue
            # The middle capture must independently witness part of the step.
            # Otherwise one clipped final box face could appear to agree with
            # a drifting encoder even when all other pictures show no motion.
            _,_,mid_rel,mid_leg=captures[(i+last)//2]
            if mid_rel is None or mid_leg is None:
                continue
            mid_visual=old_rel-mid_rel
            mid_encoder=mid_leg[0]-old_leg[0]
            mid_dist=float(np.linalg.norm(mid_visual))
            if (mid_dist<.008 or np.dot(mid_visual,mid_encoder)<=0. or
                    np.linalg.norm(mid_visual-mid_encoder)>min(.030,.7*mid_dist)):
                continue
            visual=old_rel-rel
            enc=leg[0]-old_leg[0]
            distance=float(np.linalg.norm(visual))
            if (.024<=distance<=.62*dt+.012 and np.linalg.norm(enc)>=.018
                    and np.dot(visual,enc)>.65*distance*np.linalg.norm(enc)
                    and np.linalg.norm(visual-enc)<min(.045,.55*distance)
                    and max(old_leg[1],leg[1])>=.035):
                return stamp
        return None

    def _turned(self,now,stamp,yaw):
        if (not self.attitudes or not _fresh(now,self.last_capture,self.view_lease)
                or (self.confirmed and self.last_capture<=self.last_motion)
                or len(self.exposures)<2 or self.last_capture-self.exposures[-2]>.55):
            return False
        # IMU yaw must change across real time while the observer is receiving
        # new exposures. A stale camera or repeated IMU message cannot rearm a plan.
        for prior,angle in reversed(self.attitudes):
            dt=stamp-prior
            if dt>.65 or (self.confirmed and prior<self.last_motion):
                break
            if dt>=.15 and .085<=abs(_angle(yaw-angle))<=1.9*dt:
                return True
        return False

    def update(self,now,*,capture_stamp=None,relative=None,landmark=None,
               yaw=None,imu_stamp=None,leg_xy=None,leg_velocity=None,leg_stamp=None,
               pivot_clear=False,pivot_stamp=None,scan_direction=0.,turning=False,
               passage_valid=False,active=True,zero_since=None):
        """Use monotonic acquisition times. Both XY inputs must use the same fixed axes.

        relative is a directly matched box center, never an extrapolated position.
        landmark identifies that physical box across the two measurements. leg_xy and
        leg_velocity are encoder-based. pivot_clear is permission to turn on the
        floor observed at pivot_stamp, independent of translation passage_valid.
        zero_since is the forwarding guard's continuously all-zero interval, if known.
        """
        if not active:
            if self.last_call is not None:
                self.__init__()
            return MotionStatus(0.,'inactive',0.,False)
        if not math.isfinite(now) or (self.last_call is not None and now<self.last_call):
            return MotionStatus(0.,'invalid_clock',0.,True)
        self.last_call=now
        if self.last_motion is None:
            self.last_motion=now
        if not self.confirmed and _fresh(now,zero_since,math.inf):
            self.last_motion=min(self.last_motion,zero_since)

        if _fresh(now,leg_stamp,.55):
            xy,vel=_xy(leg_xy),_xy(leg_velocity)
            if xy is not None and vel is not None:
                self._add(self.legs,leg_stamp,(xy,float(np.linalg.norm(vel))))
        imu_ok=(_fresh(now,imu_stamp,self.imu_lease) and yaw is not None and math.isfinite(yaw))
        new_imu=imu_ok and (not self.attitudes or imu_stamp>self.attitudes[-1][0])
        if new_imu:
            self._add(self.attitudes,imu_stamp,yaw)
        motion=self._capture(now,capture_stamp,landmark,relative)
        if pivot_stamp is None:
            pivot_stamp=capture_stamp
        clear_view=pivot_clear and _fresh(now,pivot_stamp,self.view_lease)
        direct_view=(landmark is not None and _xy(relative) is not None
                     and _fresh(now,capture_stamp,self.view_lease))
        # Floor inspection may finish before crate association for the same
        # exposure. A newer floor report must never discard a late landmark.
        for stamp,eligible in ((capture_stamp,direct_view),(pivot_stamp,clear_view)):
            if eligible and stamp>self.last_capture:
                self.exposures.append(stamp)
                self.last_capture=stamp
        turning=turning or now-self.last_burst<=self.burst_s+.18
        reason='watching_corridor' if passage_valid else 'watching'
        if motion is not None:
            self.last_motion=motion
            self.confirmed=True
            self.burst_yaw=None
            reason='measured_translation'
        elif new_imu and turning and (direct_view or clear_view) and self._turned(now,imu_stamp,yaw):
            self.last_motion=imu_stamp
            self.confirmed=True
            self.burst_yaw=None
            reason='measured_turn'
        still=max(0.,now-self.last_motion)
        if still<self.stall_s:
            return MotionStatus(0.,reason,still,False)

        if not imu_ok or not clear_view:
            self.burst_yaw=None
            return MotionStatus(0.,'awaiting_turn_evidence',still,True)
        if now-self.last_burst>=self.retry_s:
            self.last_burst=now
            self.burst_yaw=yaw
            self.burst_sign=(math.copysign(1.,scan_direction) if scan_direction
                             else (1. if self.attempts%2==0 else -1.))
            self.attempts+=1
        if (self.burst_yaw is not None and now-self.last_burst<self.burst_s
                and abs(_angle(yaw-self.burst_yaw))<self.burst_angle):
            return MotionStatus(self.burst_sign*self.yaw_speed,'restart_turn',still,True)
        self.burst_yaw=None
        return MotionStatus(0.,'restart_observe',still,True)
