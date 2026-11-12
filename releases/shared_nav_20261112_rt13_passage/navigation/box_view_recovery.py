"""Reobserve a blocked box from a safe heading or a recently walked, observed path.

All coordinates of floor cells are relative to the *measured* crate; poses for
retracing use the box servo's local metric coordinates. No course map is used.
"""
from collections import deque
from dataclasses import dataclass
from navigation.gap_controller import GapController
import math

import numpy as np

# One processed capture reaches the planner about .40 s after its exposure on
# this Orin (capture hand-off plus RGB-D processing, measured .44 s median for
# held frames). A view therefore stays the newest evidence for a full
# perception cycle, not for .38 s, which is shorter than the cycle itself.
VIEW_LEASE_S = .8


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


STEP=.06
X0,Y0=-1.8,-.9
NX,NY=32,30


class ViewFloor:
    """Depth evidence in the vicinity of one measured crate."""
    def __init__(self):
        self.ground=np.full((NX,NY),-np.inf)
        self.hazard=np.full((NX,NY),-np.inf)
        self.turn_space=False

    def observe(self,t,xy,h,center):
        # Current near-field evidence is separate from the historical retreat map.
        distance2=np.einsum('ij,ij->i',xy,xy)
        ground=np.abs(h)<.033
        high=(h>.10)&(h<.68)
        # Our forward camera cannot see behind the feet. A short pivot uses
        # the occupied stance; newly seen solid returns constrain its swept rim.
        near_floor=ground&(xy[:,0]>.20)&(xy[:,0]<.72)&(np.abs(xy[:,1])<.32)
        self.turn_space=(np.count_nonzero(near_floor)>=36 and
                         np.count_nonzero(high&(distance2<.32**2))<5)
        rel=xy-center
        ix=np.floor((rel[:,0]-X0)/STEP).astype(int)
        iy=np.floor((rel[:,1]-Y0)/STEP).astype(int)
        used=(ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)
        group=ix[used]*NY+iy[used]
        for mask,field,minimum in ((ground,self.ground,3),(high,self.hazard,3),(h<-.055,self.hazard,4)):
            counts=np.bincount(group[mask[used]],minlength=NX*NY)
            field[counts.reshape(NX,NY)>=minimum]=t

    def supports(self,t,start,end):
        # Require a checked strip beneath the previously traversed path,
        # and reject raised obstacles across the wider body sweep.
        samples=np.linspace(start,end,5)
        for lateral in (-.055,0.,.055):
            points=samples+[0.,lateral]
            x=np.floor((points[:,0]-X0)/STEP).astype(int)
            y=np.floor((points[:,1]-Y0)/STEP).astype(int)
            if (np.any(x<0) or np.any(x>=NX) or np.any(y<0) or np.any(y>=NY)
                    or np.any(t-self.ground[x,y]>12.)):
                return False
        bounds=np.vstack((start,end))
        # Translation is admitted within 0.10 radians of the measured axis.
        # Enclose the full projected body across that permitted rotation.
        margin=.01
        front=GapController.front_reach+.10*GapController.side_reach+margin
        rear=GapController.rear_reach+.10*GapController.side_reach+margin
        lateral=GapController.side_reach+.10*max(GapController.front_reach,GapController.rear_reach)+margin
        lo=bounds.min(axis=0)-[rear,lateral]
        hi=bounds.max(axis=0)+[front,lateral]
        i=np.floor((lo-[X0,Y0])/STEP).astype(int)
        j=np.floor((hi-[X0,Y0])/STEP).astype(int)
        if np.any(i<0) or np.any(j>=[NX,NY]):
            return False
        return not np.any(t-self.hazard[i[0]:j[0]+1,i[1]:j[1]+1]<8.)


@dataclass(frozen=True)
class View:
    stamp:float
    index:int
    relative:object
    side:int
    turn_space:bool
    retreat:object
    identity:object=None
    blocked_by:object=None


class ViewTrack:
    def __init__(self):
        self.index=None
        self.passage=None
        self.path=deque(maxlen=100)
        self.last_stamp=None
        self.snapshot=None

    def capture(self,t,pos,seen_at,relative,passage,offset,index,phase,turn_clear=False,turn_block=None):
        if index!=self.index or (passage is not None and passage is not self.passage):
            self.index=index
            self.passage=passage
            self.path.clear()
            self.snapshot=None
            self.last_stamp=None
        if (passage is None or relative is None or seen_at is None or
                not np.isfinite(seen_at) or abs(t-seen_at)>1e-6):
            # This says only that *this camera exposure* saw room to pivot.
            # No old crate coordinates or return-path cells authorize translation.
            return View(t,index,None,0,bool(turn_clear),None,id(passage) if passage is not None else None,
                        None if turn_clear else (turn_block or 'near_field_unconfirmed'))
        if self.last_stamp==seen_at:
            return self.snapshot
        self.last_stamp=seen_at
        rel=np.asarray(relative,dtype=float)
        pos=np.asarray(pos,dtype=float)
        floor=passage.view_floor
        # Odometry and independently measured displacement of this same crate
        # must agree. Reject relocation jumps and lateral shortcuts when reversing.
        retreat=None
        if turn_clear and floor.turn_space and rel[0]>-.10:
            for stamp,old,old_rel,side in self.path:
                delta=pos-old
                optical_distance=old_rel[0]-rel[0]
                if (0<t-stamp<=12. and .085<=delta[0]<=.24 and optical_distance>.025 and abs(delta[1])<.052 and
                        abs(delta[1]-(old_rel-rel)[1])<.047 and
                        abs(delta[0]-(old_rel-rel)[0])<.14 and
                        floor.supports(t,-rel,-old_rel) and
                        floor.supports(t,-old_rel+delta,-old_rel)):
                    retreat=(float(pos[0]-min(delta[0],optical_distance)),float(pos[1]))
                    break
        if offset and floor.turn_space and abs(rel[0])<1.65 and (not self.path or np.linalg.norm(pos-self.path[-1][1])>.022):
            self.path.append((t,pos.copy(),rel.copy(),int(np.sign(offset))))
        last_side=int(np.sign(offset)) if offset else (self.path[-1][3] if self.path else 0)
        # Check both the whole-camera body-frame near field and the current
        # crate in the track frame. The return-path certificate permits only
        # an aligned reverse when the crate is inside the pivot envelope.
        # Distance to the enlarged square is tighter than a circle around
        # its centre. This permits an on-the-spot sweep beside a clear face
        # while protecting a projecting corner through the entire rotation.
        crate_clearance=np.linalg.norm(np.maximum(np.abs(rel)-.125,0.))
        pivot_radius=math.hypot(max(GapController.front_reach,GapController.rear_reach),
                                GapController.side_reach)+.01
        blocked=None
        if not turn_clear:
            blocked=turn_block or 'near_field_unconfirmed'
        elif not floor.turn_space:
            blocked='local_floor_or_pivot_rim_unconfirmed'
        elif crate_clearance<pivot_radius:
            blocked='crate_inside_pivot_envelope'
        self.snapshot=View(seen_at,index,tuple(rel),last_side,blocked is None,retreat,id(passage),blocked)
        return self.snapshot


class BoxViewRecovery:
    def __init__(self):
        self.action='idle'
        self.started=None
        self.index=None
        self.identity=None
        self.last_side=0
        self.aim=0.
        self.retreat_target=None
        self.scans=0
        self.retreats=0
        self.view_changed=False
        self.next_retreat=0.
        self.retreat_deadline=0.
        self.retreat_goal_range=0.
        self.hold_reason=None
        self.scan_cursor=0
        self.scan_direction=None
        self.scan_leg=None
        self.scan_progress=None
        self.scan_pulse_until=None
        self.scan_pulses=0
        self.imu_response='idle'

    def clear(self):
        self.action='idle'
        self.started=None
        self.retreat_target=None
        self.view_changed=False
        self.stop_scan()
        self.hold_reason=None

    def stop_scan(self):
        # Keep the next viewpoint across a brief handoff in the same box.
        self.scan_leg=None
        self.scan_progress=None
        self.scan_pulse_until=None
        self.imu_response='idle'

    def hold(self,reason):
        self.stop_scan()
        self.action='hold'
        self.retreat_target=None
        self.hold_reason=reason
        return np.zeros(3)

    def scan(self,t,yaw,heading,direction):
        # Drive the next viewpoint immediately when the IMU reaches this one.
        # A short stronger pulse addresses a gait that received yaw commands
        # without changing heading. If it remains unresponsive, try the other
        # direction, still restricted by each new camera's pivot certificate.
        angles=(.25,-.27,.40,-.39,.18,-.20)
        if self.scan_direction is None:
            self.scan_direction=direction
        if self.scan_leg is not None:
            err=wrap(self.aim-yaw)
            elapsed=t-self.scan_leg
            witness_t,witness_yaw=self.scan_progress
            if math.copysign(1.,err)*wrap(yaw-witness_yaw)>=.025:
                self.scan_progress=(t,yaw)
                self.scan_pulse_until=None
                self.imu_response='measured_turn'
            elif self.scan_pulse_until is None and t-witness_t>=.36:
                self.scan_pulse_until=t+.26
                self.scan_pulses+=1
                self.imu_response='strong_rescan'
            tried=self.scan_pulse_until is not None and t>=self.scan_pulse_until
            if abs(err)<=.055 or elapsed>=1.45 or tried:
                self.scan_leg=None
        if self.scan_leg is None:
            # At most one adjacent target can already coincide with this yaw.
            for _ in range(2):
                self.aim=heading+self.scan_direction*angles[self.scan_cursor%len(angles)]
                self.scan_cursor+=1
                if abs(wrap(self.aim-yaw))>.055:
                    break
            self.scan_leg=t
            self.scans+=1
            self.scan_progress=(t,yaw)
            self.scan_pulse_until=None
            self.imu_response='turn_requested'
        self.action='scan'
        self.view_changed=True
        err=wrap(self.aim-yaw)
        rate=.72 if self.scan_pulse_until is not None else min(.58,max(.40,2.4*abs(err)))
        return np.array([0.,0.,math.copysign(rate,err)])

    def diagnostic(self):
        return dict(crate_view_action=self.action,crate_view_scans=self.scans,
                    crate_view_retreats=self.retreats,crate_view_aim=self.aim,
                    crate_view_hold_reason=self.hold_reason,crate_view_imu_response=self.imu_response,
                    crate_view_scan_pulses=self.scan_pulses)

    def command(self,t,yaw,heading,pos,index,phase,offset,waiting,view):
        target=index+int(phase=='shift')
        identity=view.identity if view is not None and view.index==target else None
        if target!=self.index or (identity is not None and identity!=self.identity):
            self.clear();self.index=target;self.identity=identity;self.last_side=0
            self.scan_cursor=0;self.scan_direction=None
        if yaw is None or heading is None or not np.isfinite(yaw) or not np.isfinite(heading):
            return self.hold('imu_heading_unavailable')
        self.hold_reason=None
        if phase=='lane_entry' and not waiting:
            self.clear();return None
        fresh=view is not None and view.index==target and 0<=t-view.stamp<=VIEW_LEASE_S
        rel=np.asarray(view.relative) if fresh and view.relative is not None else None
        side=int(np.sign(offset))
        switch=bool(side and self.last_side and side!=self.last_side)
        # A new sign next to an already approached crate is not a sideways
        # route through that crate. Require room in front before changing sides.
        close_switch=switch and (rel is None or rel[0]<.55)
        continuing_return=(self.action=='retreat' and fresh and rel is not None and view.retreat is not None
                           and t<self.retreat_deadline and rel[0]<self.retreat_goal_range-.02)
        if not waiting and not close_switch and not continuing_return:
            if self.view_changed and abs(wrap(heading-yaw))>.11:
                if not fresh:
                    return self.hold('fresh_view_required_to_realign')
                if not view.turn_space:
                    return self.hold(view.blocked_by or 'pivot_space_unconfirmed')
                self.stop_scan()
                self.action='realign'
                return self.turn_to(t,yaw,heading,view)
            self.clear()
            if side:self.last_side=side
            return None
        # Passage evidence has been withdrawn. The native hold may request
        # heading correction as well; take ownership here to enforce fresh space.
        if self.started is None:
            self.started=t;self.next_retreat=t+1.10
        if not fresh:
            reason='target_view_mismatch' if view is not None and view.index!=target else 'fresh_depth_view_required'
            return self.hold(reason)
        if not view.turn_space and view.retreat is None:
            return self.hold(view.blocked_by or 'pivot_space_and_return_path_unconfirmed')
        if phase!='lane_entry' and t>=self.next_retreat and view.retreat is not None and self.retreat_target is None:
            self.stop_scan()
            self.retreat_target=view.retreat
            self.retreats+=1
            self.action='retreat'
            self.retreat_deadline=t+2.
            self.view_changed=True
            self.retreat_goal_range=rel[0]+min(.19,np.asarray(pos)[0]-view.retreat[0])
        if self.action=='retreat':
            # Translation is restricted to an aligned, re-certified return.
            eligible=(view.retreat is not None and rel is not None and rel[0]<self.retreat_goal_range-.015
                      and t<self.retreat_deadline and np.asarray(pos)[0]-view.retreat[0]>.018)
            if eligible:
                delta=wrap(heading-yaw)
                if abs(delta)>.10:
                    if not view.turn_space:
                        return self.hold(view.blocked_by or 'cannot_align_with_return_path')
                    return self.turn_to(t,yaw,heading,view)
                speed=min(.27,max(.12,2.6*(self.retreat_goal_range-rel[0])))
                return np.array([-speed*math.cos(delta),-speed*math.sin(delta),0.])
            self.retreat_target=None
            self.next_retreat=t+2.5
        if not view.turn_space:
            return self.hold(view.blocked_by or 'pivot_space_unconfirmed')
        focus_y=rel[1]+(view.side or self.last_side)*.23 if rel is not None else float(self.last_side or 1)
        return self.scan(t,yaw,heading,1 if focus_y>=0 else -1)

    @staticmethod
    def turn_to(t,yaw,target,view):
        if not view.turn_space or not 0<=t-view.stamp<=VIEW_LEASE_S:
            return np.zeros(3)
        err=wrap(target-yaw)
        angular=0. if abs(err)<.045 else math.copysign(min(.50,max(.24,2.4*abs(err))),err)
        return np.array([0.,0.,angular])
