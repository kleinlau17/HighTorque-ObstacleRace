"""Identical sensor-to-command pipeline for simulation, replay and ROS.

All positions are estimated from RGB-D. Attitude is world_from_body from an
AHRS measurement. No world position or simulator object data is accepted.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from navigation import params
from navigation.command import AMPCommandAdapter
from navigation.motion_recovery import MotionRecovery
from navigation.visual import CameraPerception, Perception, rotation2, cluster_points, fit_box


def rotation_matrix(value):
    if (value.shape != (3, 3) or not np.isfinite(value).all() or
            not np.allclose(value.T@value, np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(value), 1, atol=1e-4)):
        raise ValueError('Expected a proper rotation matrix')


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray


def core_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in ('.py', '.json')}


class LaneTurn:
    """Bound an in-place correction using signed, unwrapped AHRS rotation."""
    angle = np.deg2rad(6.)

    def __init__(self):
        self.started = None
        self.origin = 0.
        self.direction = 0
        self.settle_until = -np.inf
        self.no_response = 0
        self.fault = None

    def pause(self, now):
        if self.started is not None:
            self.settle_until = now+.5
            self.started = None

    def request(self, now, turn, heading, frame_time, needs_turn, measured_limit=None):
        """Return a yaw target, reason and absolute publisher cutoff."""
        if not needs_turn:
            self.pause(now)
            if frame_time<=self.settle_until:
                return 0., 'turn_reobserving', now
            self.fault = None
            self.no_response = 0
            return 0., 'ready', np.inf
        if self.fault:
            if self.direction*(turn-self.origin)<self.angle:
                return 0., self.fault, now
            self.fault = None
            self.no_response = 0
        if frame_time<=self.settle_until:
            return 0., 'turn_reobserving', now
        direction = 1 if heading>0 else -1
        if self.started is None:
            if self.direction!=direction:
                self.no_response = 0
            self.started, self.origin, self.direction = now, turn, direction
        progress = self.direction*(turn-self.origin)
        if measured_limit is not None and measured_limit[0]==self.started:
            # The 50 Hz publisher may see the limit and cut the turn while a
            # camera frame is processing, even if the body subsequently sways back.
            progress = measured_limit[1]
        cutoff = self.started+1.8
        if progress<=-self.angle:
            self.fault = 'turn_direction_mismatch'
        elif direction!=self.direction or progress>=self.angle or now>=cutoff:
            if progress>=np.deg2rad(1.):
                self.no_response = 0
            elif now>=cutoff:
                self.no_response += 1
                if self.no_response>=2:
                    self.fault = 'turn_stalled'
        else:
            # The factory policy showed a signed response to +/-1.0 on hardware.
            return float(direction), 'ready', cutoff
        self.pause(now)
        return 0., self.fault or 'turn_reobserving', now


class OnboardNavigation:
    """The camera loop owns steering. One worker owns optional tracking on clear lanes.

    On box/finish entry, drain the one in-flight observation and hand the tracker
    back to the camera loop. No mutable tracker data are read while the worker runs.

    RT12 keeps avoid mode through a few consecutive unseen frames instead of
    dropping out on a single occlusion.
    """
    exit_frames = 3
    lane_recall_frames = 3
    lane_recall_timeout = 3.
    lane_recall_gain = 1.5
    lane_recall_limit = .25

    def __init__(self, width, height, intrinsics, position, rotation, speed, observation_lease):
        self.local = CameraPerception(width, height, 80., position, rotation, intrinsics)
        # Avoidance owns a separate perception so the two modes never share
        # lane/ground state.
        self.avoid_local = CameraPerception(width, height, 80., position, rotation, intrinsics)
        self.missed_previous = False
        # The tracker was removed; keep these defined so the optional motion-
        # recovery block skips instead of raising AttributeError.
        self.snapshot_time = None
        self.snapshot = None
        self.recovery_stamp = self.recovery_sample = None
        self.lease = observation_lease
        self.speed = speed
        self.exit_frames = params.get('lane','exit_frames')
        self.lane_recall_frames = params.get('lane','recall_frames')
        self.lane_recall_timeout = params.get('lane','recall_timeout_s')
        self.lane_recall_gain = params.get('lane','recall_gain')
        self.lane_recall_limit = params.get('lane','recall_limit')
        self.adapter = AMPCommandAdapter()
        self.recovery = MotionRecovery()
        self.initial_yaw = self.yaw = self.last_imu_time = None
        self.turn = 0.
        self.last_frame_time = self.lane_time = self.lane_yaw = None
        self.scene = 'unknown_depth'
        self.reference_time = None
        self.inside_lane_start = True
        self.lane = self.support = None
        self.steering_heading = None
        self.turning_to_lane = False
        self.lane_turn = LaneTurn()
        self.tracking_control = self.holding_lane = False
        self.boxes_seen = False
        self.missed_frames = 0
        self.no_lane_frames = 0
        self.avoid_phase = 'off'
        self.avoid_frames = 0
        self.avoid_since = 0.
        self.box = None
        self.boxes = []
        self.observation = None
        self.avoid_obs_time = -1.
        # Committed lateral step (direction fixed when the step starts) and
        # whether the lane has been usable as a distance reference during it.
        self.strafe_dir = 0.
        self.strafe_lane_seen = False
        self.coast_since = None
        self.strafe_ref = None
        self.last_control_time = self.deadline = None
        self.desired = np.zeros(3)
        # Last translational request. RT10 never publishes zero once walking:
        # a missing plan holds this velocity instead of standing still.
        self.last_drive = np.zeros(2)
        self.reason = 'initializing'

    def set_attitude(self, timestamp, rotation):
        rotation_matrix(rotation)
        if not np.isfinite(timestamp) or (self.last_imu_time is not None and timestamp<self.last_imu_time):
            raise ValueError('Nonmonotonic IMU timestamp')
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        if self.yaw is not None:
            self.turn += (yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
        self.yaw, self.last_imu_time = yaw, timestamp
        self.attitude = rotation.copy()

    def close(self, path):
        pass

    def dispatch_mapping(self, tick, now, frame_period):
        pass

    def observe(self, frame, attitude, imu_time, inhibit, defer_mapping=False):
        if not np.isfinite(frame.time) or abs(frame.time-imu_time)>.15 or self.last_imu_time is None:
            raise ValueError('No time-aligned attitude for camera frame')
        if self.last_frame_time is not None and frame.time<=self.last_frame_time:
            raise ValueError('Nonmonotonic camera timestamp')
        if frame.depth.shape!=self.local.rays.shape[:2] or frame.rgb.shape!=(*frame.depth.shape,3):
            raise ValueError('Camera resolution differs from calibrated intrinsics')
        self.last_frame_time = frame.time
        fresh_lane = self.reference_time is not None and frame.time-self.reference_time<self.lease
        capture_yaw = np.arctan2(attitude[1,0], attitude[0,0])
        reference = None
        if fresh_lane:
            # Rotate the previous centre into the new heading before identifying a single edge.
            delta = self.lane_yaw-capture_yaw
            if abs(np.arctan2(np.sin(delta),np.cos(delta)))<.30:
                x = np.linspace(*self.support, 12)
                points = np.column_stack([x, np.polyval(self.lane, x)])@rotation2(delta).T
                if np.ptp(points[:,0])>.12:
                    reference = np.polyfit(points[:,0], points[:,1], 2)
        observation, self.scene = self.local.inspect_lane(
            frame, attitude, reference, inside_start=self.inside_lane_start)
        image_ok = np.count_nonzero((frame.rgb.max(axis=2)>5)&(frame.rgb.min(axis=2)<250))>=frame.depth.size*.01
        if not image_ok:
            self.scene = 'unusable_rgb'
        was_tracking = self.tracking_control
        raised = self.scene=='raised_object'
        self.holding_lane = False
        # RT12: avoid mode exists only while a box is actually seen. The cheap
        # raised test runs every frame, so a new box is picked up here; once the
        # tracker has no box in the current frame, control returns to the lane.
        # The avoidance view is computed every frame from the shared analysis;
        # it does not depend on the lane scene or its floor coverage.
        avoid = observation; avoid_scene = 'clear'
        if self.local.analysis is not None:
            avoid, avoid_scene = self.avoid_local.inspect_avoid(self.local.analysis)
        seen = avoid_scene=='raised_object'
        self.observation = avoid if (seen or self.tracking_control) else observation
        if seen and self.avoid_local.raised_sites is not None and len(self.avoid_local.raised_sites):
            sites=self.avoid_local.raised_sites
            self.boxes=[b for b in (fit_box(c) for c in cluster_points(sites[:,:2], min_pts=params.get('box','cluster_min_pts'))) if b is not None]
            self.box=(np.array([self.boxes[0]['center'][0],self.boxes[0]['center'][1],.2,.2])
                      if self.boxes else None)
        else:
            self.box=None; self.boxes=[]
        # Only a fitted 20x20x40 box counts as an obstacle. A few raised pixels
        # without a box must not stop the walk or latch the avoid mode.
        raised = seen and self.box is not None
        if raised:
            self.scene = 'raised_object'
        if raised:
            if self.box is not None and self.box[0]<1.0:
                self.inside_lane_start=False
            self.missed_frames=0
            if not was_tracking:
                self.tracking_control=True
                self.avoid_phase='approach'
        else:
            self.missed_frames+=1
        paired = self.local.lane_basis=='paired_edges'
        if was_tracking and not raised and self.missed_frames>=self.exit_frames and paired:
            self.tracking_control=False
            self.avoid_phase='off'
            if abs(self.last_drive[0])<.05:
                self.last_drive=np.array([min(self.speed,.10),0.])
        if self.tracking_control and not was_tracking:
            self._enter_avoid()
        if was_tracking and not self.tracking_control:
            self._enter_lane()
        if self.tracking_control:
            self.lane_time=None
            self.no_lane_frames=0
            return observation
        if self.scene in ('clear','turn_clear') and observation.lane_observed:
            self.steering_heading = observation.steering_heading
            self.lane, self.support = observation.lane.copy(), observation.lane_support
            self.reference_time = self.lane_time = frame.time
            self.lane_yaw = capture_yaw
            self.no_lane_frames = 0
        elif (self.scene=='clear' and self.lane_time is not None and reference is not None
              and not self.turning_to_lane and not self.missed_previous):
            self.holding_lane = True
        else:
            self.lane_time = None
            self.no_lane_frames += 1
            inhibit()
        self.missed_previous = not observation.lane_observed
        return observation

    def _near_white(self):
        obs=self.observation
        pts=getattr(obs,'white',None) if obs is not None else None
        if pts is None or len(pts)<5:
            return None
        pts=np.asarray(pts,dtype=float)
        near=pts[(pts[:,0]>.15)&(pts[:,0]<.65)]
        if len(near)<5:
            return None
        return float(np.median(near[:,1]))

    def _strafe_done(self, timestamp):
        """Distance stop for the committed lateral step.

        The lane fit is not updated while avoid mode owns the scene, so the
        paint itself is the ruler: how far the near white points moved
        sideways since the step started.
        """
        n=self._near_white()
        if n is not None:
            if self.strafe_ref is None:
                self.strafe_ref=n
                self.strafe_lane_seen=True
                return False
            return abs(n-self.strafe_ref)>=params.get('avoid','strafe_offset')
        # No paint to measure with: keep the blind step short.
        return (not self.strafe_lane_seen) and timestamp-self.avoid_since>.9

    def _avoid_step(self, obs, box, timestamp):
        """Memory-free avoidance built directly on the white points."""
        pts=None
        if obs is not None and getattr(obs,'white',None) is not None and len(obs.white):
            pts=np.asarray(obs.white,dtype=float)
        def medians():
            if pts is None: return None,None
            x,y=pts[:,0],pts[:,1]
            near=y[(x>=.30)&(x<.55)]; far=y[(x>=.60)&(x<.95)]
            n=float(np.median(near)) if len(near)>=3 else None
            f=float(np.median(far)) if len(far)>=3 else None
            return n,f
        def follow():
            n,f=medians()
            if n is None:
                self.reason='avoid_wait'; return np.zeros(3)
            side=params.get('avoid','follow_side')
            ride=self.strafe_dir
            near=pts[(pts[:,0]>=.30)&(pts[:,0]<.55)] if pts is not None else None
            spread=float(np.ptp(near[:,1])) if near is not None and len(near)>=6 else 0.
            if ride and spread>=.30:
                # Both tapes in view: their median is the lane centre. Target
                # the body 'side' metres away from it, on the ride side.
                err=n-(-ride*side)
            elif ride:
                # Only one tape left: it has already passed under/beside the
                # body, so hold the line instead of re-centring on it.
                err=0.
            else:
                err=n-(side if n>0 else -side)
            vy=float(np.clip(params.get('avoid','follow_ky')*err,
                             -params.get('avoid','follow_vy_max'),params.get('avoid','follow_vy_max')))
            if abs(err)>params.get('avoid','follow_align_tol'):
                self.reason='avoid_align'
                return np.array([0.,vy,0.])
            heading=float(np.arctan2(f-n,.30)) if f is not None else 0.
            self.reason='avoid_follow'
            return np.array([params.get('avoid','follow_vx'),vy,
                             float(np.clip(params.get('avoid','follow_kw')*heading,
                                           -params.get('avoid','follow_w_max'),params.get('avoid','follow_w_max')))])
        if box is None:
            if (self.avoid_phase=='strafe' and self.strafe_dir
                    and timestamp-self.avoid_since <= params.get('avoid','strafe_s')):
                if self._strafe_done(timestamp):
                    self.avoid_phase='follow'; self.avoid_frames=0
                    return np.zeros(3)
                self.reason='avoid_strafe'
                return np.array([0., self.strafe_dir*params.get('avoid','strafe_speed'), 0.])
            if pts is not None: return follow()
            self.reason='avoid_wait'; return np.zeros(3)
        bx=float(box[0]); by=float(box[1])
        # A nearer box on the other side re-locks the ride side at once; the
        # next crate is only ~0.5 m away, so waiting for this segment to end
        # leaves no time to switch.
        want=1. if by>0 else -1.
        if bx<.90 and want!=self.strafe_dir and self.avoid_phase in ('follow','off','approach'):
            self.strafe_dir=want
            self.avoid_phase='stop'; self.avoid_frames=0
            return np.zeros(3)
        connected=False
        if pts is not None:
            connected=bool(np.any((np.abs(pts[:,0]-bx)<.08)&(np.abs(pts[:,1]-by)<.06)))
        if self.avoid_phase in ('off','approach'):
            self.avoid_phase='stop'; self.avoid_frames=0
        if self.avoid_phase=='stop':
            if obs is not None and obs.time!=self.avoid_obs_time:
                self.avoid_frames+=1
                self.avoid_obs_time=obs.time
            self.reason='avoid_stop'
            if self.avoid_frames>=params.get('avoid','stop_frames'):
                lateral=abs(by); body_half=params.get('avoid','body_half')
                # Re-lock the ride side for THIS box, even when the step was
                # done and the machine goes straight to follow.
                self.strafe_dir = 1. if by>0 else -1.
                if not connected and lateral>=body_half:
                    self.avoid_phase='follow'
                elif connected or lateral<body_half:
                    # Side away from the box. The box leaves the frontal
                    # window while stepping aside, so the direction is fixed
                    # here and the distance is measured on the lane instead.
                    self.avoid_phase='strafe'; self.avoid_since=timestamp
                    self.strafe_dir = 1. if by>0 else -1.
                    self.strafe_lane_seen = False
                    self.strafe_ref = None
                else:
                    self.avoid_phase='back'; self.avoid_since=timestamp
            return np.zeros(3)
        if self.avoid_phase=='strafe':
            if timestamp-self.avoid_since>params.get('avoid','strafe_s'):
                self.avoid_phase='stop'; self.avoid_frames=0
                return np.zeros(3)
            # Stop by measured displacement, not by time: the first few hundred
            # milliseconds of a lateral request barely move the body, so a fixed
            # duration lands anywhere between 5 and 40 cm.
            done = self._strafe_done(timestamp)
            if done:
                # Go straight to following the far line; returning to 'stop'
                # made the box-side gate (body_half) strafe again and overshoot.
                self.avoid_phase='follow'; self.avoid_frames=0
                return np.zeros(3)
            d=self.strafe_dir or (1. if by>0 else -1.)
            self.reason='avoid_strafe'
            return np.array([0.,d*params.get('avoid','strafe_speed'),0.])
        if self.avoid_phase=='back':
            if timestamp-self.avoid_since>params.get('avoid','back_s'):
                self.avoid_phase='stop'; self.avoid_frames=0
                return np.zeros(3)
            self.reason='avoid_back'
            return np.array([params.get('avoid','back_speed'),0.,0.])
        return follow()

    def _enter_avoid(self):
        self.avoid_phase='approach'; self.avoid_frames=0
        self.avoid_since=0.; self.avoid_obs_time=-1.
        self.box=None; self.observation=None

    def _enter_lane(self):
        self.avoid_phase='off'; self.avoid_frames=0
        self.avoid_since=0.; self.avoid_obs_time=-1.
        self.box=None; self.observation=None
        # Never carry an avoid command (possibly backward) into lane mode.
        self.last_drive=np.array([min(self.speed,.10),0.])

    def command(self, timestamp, enabled=True, measured_turn_limit=None):
        previous = self.last_control_time
        self.last_control_time = timestamp
        dt = .02 if previous is None else np.clip(timestamp-previous, 0., .10)
        self.deadline = self.last_frame_time+self.lease if self.last_frame_time is not None else timestamp
        fresh_lane = self.lane_time is not None and timestamp<=self.lane_time+self.lease
        # RT10: every reason below is telemetry. Nothing here zeroes the
        # request; a missing plan holds the previous translation and the
        # mission stops only on its 300 s limit or an operator stop.
        if not enabled:
            self.reason = 'disabled'
        elif self.last_imu_time is None or not 0<=timestamp-self.last_imu_time<=.15:
            self.reason = 'stale_imu'
        elif self.last_frame_time is None or timestamp>self.deadline:
            self.reason = 'stale_camera'
        elif self.scene in ('unknown_depth','unusable_rgb'):
            self.reason = self.scene
        elif self.tracking_control:
            self.reason = 'tracking'
        elif not fresh_lane:
            self.reason = 'lane_stale'
        else:
            self.reason = 'ready'
        if self.tracking_control:
            self.turning_to_lane = False
            box=self.box
            if box is not None and box[0]<params.get('avoid','approach_far'):
                # Inside the avoidance range the state machine owns the command;
                # there is no dead hold band that would swallow a reverse.
                target=self._avoid_step(self.observation,box,timestamp)
                self.desired+=np.clip(target-self.desired,-np.array([.4,.5,1.5])*dt,np.array([.4,.5,1.5])*dt)
            elif box is not None:
                target=np.array([.10,0.,0.])
                self.desired+=np.clip(target-self.desired,-np.array([.4,.5,1.5])*dt,np.array([.4,.5,1.5])*dt)
                self.reason='avoid_approach'
            else:
                target=self._avoid_step(self.observation,None,timestamp)
                self.desired+=np.clip(target-self.desired,-np.array([.4,.5,1.5])*dt,np.array([.4,.5,1.5])*dt)
        elif fresh_lane:
            self.deadline = min(self.deadline, self.lane_time+self.lease)
            near, far = self.support
            near = max(.15, near)
            look = np.clip(near+.25, near, max(near,far))
            heading = np.arctan(np.polyval(np.polyder(self.lane),look))
            if self.steering_heading is not None:
                heading = self.steering_heading
            heading += (self.lane_yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
            angle_limit = np.deg2rad(16 if self.turning_to_lane else 30)
            self.turning_to_lane = self.scene=='turn_clear' or abs(heading)>angle_limit
            slope = np.polyval(np.polyder(self.lane),near)
            side = (np.polyval(self.lane,near)-near*slope)/np.hypot(1.,slope)
            vx = min(self.speed, .20) if self.holding_lane else self.speed
            target = np.array([vx, np.clip(1.8*side,-.20,.20), np.clip(2.5*heading,-.8,.8)])
            if (self.scene=='turn_clear' and (abs(heading)<self.lane_turn.angle
                    or heading*self.steering_heading<=0)):
                self.lane_turn.pause(timestamp)
                yaw, self.reason, cutoff = 0., 'turn_reobserving', timestamp
            else:
                yaw, self.reason, cutoff = self.lane_turn.request(timestamp, self.turn, heading,
                                                                 self.lane_time, self.turning_to_lane,
                                                                 measured_turn_limit)
            self.deadline = min(self.deadline, cutoff)
            if self.turning_to_lane:
                target[:2] = 0.
                self.desired[:2] = 0.
                if yaw*self.desired[2]<0:
                    self.desired[2] = 0.
                target[2] = yaw
                yaw_lease = min(self.lease, max(.20, abs(heading)/.8))
                self.deadline = min(self.deadline,self.lane_time+yaw_lease)
                if timestamp>self.deadline and self.reason=='ready':
                    self.reason = 'turn_observation_expired'
            self.desired += np.clip(target-self.desired, -np.array([.4,.5,1.5])*dt, np.array([.4,.5,1.5])*dt)
            # Stall detection is reported for telemetry but never brakes RT10.
            if self.snapshot_time is not None and self.snapshot_time!=self.recovery_sample:
                self.recovery_sample = self.snapshot_time
                valid = self.snapshot['velocity_valid'] and 0<=timestamp-self.snapshot_time<=self.lease
                if self.recovery_stamp is None or self.snapshot_time-self.recovery_stamp>self.lease:
                    self.recovery.stalled_since = None
                self.recovery_stamp = self.snapshot_time
                self.recovery.update(self.snapshot_time,self.desired,np.asarray(self.snapshot['pose']),
                                     self.snapshot['measured_velocity'],valid)
            if self.recovery.phase=='failed':
                self.reason = 'motion_stalled'
            elif self.recovery.phase=='settle' and timestamp<self.recovery.settle_until:
                self.reason = 'standing_recovery'
        self.recovery.stalled_since = None
        if self.tracking_control:
            if np.any(self.desired[:2]):
                self.last_drive = self.desired[:2].copy()
            elif np.any(self.last_drive):
                self.desired[:2] = self.last_drive
                self.desired[2] = 0.
        else:
            # RT10 fallback, kept byte-for-byte in behaviour: without a fresh
            # lane the yaw is cleared and the last translation is held, so the
            # body keeps moving through in-place turns and stands at start-up
            # until a lane (or a box) is seen.
            if not fresh_lane:
                self.desired[2] = 0.
            if np.any(self.desired[:2]):
                self.last_drive = self.desired[:2].copy()
            elif np.any(self.last_drive):
                self.desired[:2] = self.last_drive
            if (not self.tracking_control and not fresh_lane
                    and self.no_lane_frames>=self.lane_recall_frames
                    and self.lane is not None and self.reference_time is not None
                    and timestamp-self.reference_time<=self.lane_recall_timeout):
                # Nothing has been seen for a few frames: turn slightly toward
                # the last known lane so the camera can find it again.
                heading = np.arctan(np.polyval(np.polyder(self.lane), .25))
                heading += (self.lane_yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
                self.desired[2] += float(np.clip(self.lane_recall_gain*heading,
                                                 -self.lane_recall_limit, self.lane_recall_limit))
        if (not fresh_lane and not self.tracking_control and self.lane_time is not None
                and self.coast_since is not None and timestamp-self.coast_since <= 1.50):
            # The near lane often disappears for a moment beside a box. Keep
            # walking straight and re-fit instead of standing until the mission
            # times out; only if 1.5 s of coasting fails does the fallback run.
            self.desired[:] = [min(.07, self.speed), 0., 0.]
            self.reason = 'lane_coast'
            return self.adapter.update(self.desired, enabled=True)
        return self.adapter.update(self.desired, enabled=True)

    def diagnostic(self):
        obs=self.observation
        box=None if self.box is None else self.box.tolist()
        return dict(reason=self.reason,
                    control_source='tracked' if self.tracking_control else 'current_lane',
                    mode='avoid' if self.tracking_control else ('turn_to_lane' if self.turning_to_lane else 'lane'),
                    scene=self.scene, lane_basis=self.local.lane_basis,
                    lane_observed=bool(obs is not None and obs.lane_observed),
                    boxes_seen=self.box is not None,
                    tracking_control=self.tracking_control,
                    avoid_phase=self.avoid_phase, avoid_frames=self.avoid_frames,
                    missed_frames=self.missed_frames, lane_held=self.holding_lane,
                    local_lane_time=self.lane_time, turn=self.turn,
                    turn_no_response=self.lane_turn.no_response,
                    lane_ground_reason=self.local.ground.result.reason,
                    desired_velocity=self.desired.tolist(),
                    target_policy_velocity=self.adapter.target.tolist(),
                    predicted_policy_velocity=self.adapter.filtered.tolist(),
                    tracking_frame_time=self.last_frame_time,
                    fit_ms=getattr(self.local,'last_fit_ms',None),
                    avoid_basis=getattr(self.avoid_local,'lane_basis','no_fit'),
                    avoid_raw=bool(getattr(self.avoid_local,'raw_lane',None) is not None),
                    avoid_edges=0 if getattr(self.avoid_local,'lane_edges',None) is None else int(len(self.avoid_local.lane_edges)),
                    avoid_sites=0 if getattr(self.avoid_local,'raised_sites',None) is None else int(len(self.avoid_local.raised_sites)))
