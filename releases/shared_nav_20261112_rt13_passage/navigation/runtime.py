"""Identical sensor-to-command pipeline for simulation, replay and ROS.

Positions are estimated from RGB-D and leg encoders between captures. Attitude is world_from_body from an
AHRS measurement. No world position or simulator object data is accepted.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np

from navigation.box_servo import BoxServo
from navigation.box_motion_watch import BoxMotionWatch
from navigation.command import AMPCommandAdapter
from navigation.environment import verify_dependencies
from navigation.floor_map import FloorMap
from navigation.motion_recovery import MotionRecovery
from navigation.visual import CameraPerception, Perception, RGBDOdometry, VisualNavigator, rotation2


# Local RT13 patch (2026-11-12): in the box zone the body rides the boundary
# tape opposite the box with that tape under the centre axis, so at most one
# foot is outside the lane. The tape sits half a lane width from the measured
# centre, and the side flips when the next box appears. The multi-crate
# passage servo is used only when two or more boxes are in the corridor.
RIDE_HALF_WIDTH_M = .35
RIDE_HOLD_S = 4.0
RIDE_SWITCH_S = 1.5
RIDE_DEADBAND = .08
# The factory AMP zeroes |v| below its own deadband (0.08 for vx/vy) and the
# adaptation has to push a floor of 0.10 through its .3 filter. A 0.12 lateral
# request therefore does not reliably move the body; the crate servo boost of
# 0.18-0.32 confirms it. Use 0.18 and step up while the body does not move.
BLIND_BOX_SPEED = .16
BLIND_BOX_LATERAL = .18
BLIND_BOX_LATERAL_MAX = .22
BLIND_BOX_S = 4.0
# A short drop in tape points does not invalidate the last fit immediately.
# While a fit is held the body keeps going straight: an old fit must not add
# lateral motion after the body has already moved.
TAPE_HOLD_S = 1.0
# On-board box-zone hold: the first frame that measures a raised cluster stops
# the body and keeps it stopped until the operator creates this file. No SSH
# round trip is involved, so the stop is immediate.
BOX_HOLD_FLAG = Path('/tmp/competition_nav_box_resume')
BOX_REARM_S = 3.0
# Right after the box leaves the frontal window the lane is usually still
# hidden. Creep straight for a moment instead of standing in the gap.
BOX_RECENT_S = 3.0
BLIND_CREEP_S = 2.0
LANE_COAST_S = 1.5


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray


def core_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in ('.py', '.json')}


class NavigationRuntime:
    def __init__(self, width, height, intrinsics, camera_position, camera_rotation, speed=.22, observation_lease=.35):
        verify_dependencies()
        if not np.isfinite(speed) or not 0 < speed <= .4:
            raise ValueError('Navigation target speed must be within (0, 0.4]')
        if not np.isfinite(observation_lease) or not 0 < observation_lease <= 2.:
            raise ValueError('Observation lease must be within (0, 1] s')
        self.observation_lease = float(observation_lease)
        self.position = np.asarray(camera_position, dtype=float)
        self.camera_rotation = np.asarray(camera_rotation, dtype=float)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError('Invalid camera position')
        self._rotation(self.camera_rotation)
        self.perception = CameraPerception(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.odometry = RGBDOdometry(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.floor_map = FloorMap(self.odometry.floor)
        self.navigator = VisualNavigator(speed, self.observation_lease)
        self.adapter = AMPCommandAdapter()
        self.initial_yaw = None
        self.yaw = 0.
        self.turn = 0.
        self.distance = 0.
        self.xy = np.zeros(2)
        self.attitude = np.eye(3)
        self.last_imu_time = None
        self.last_frame_time = None
        self.last_lane_observation_time = None
        self.rgb_usable = False
        self.last_control_time = None
        self.observation = None
        self.good_frames = 0
        self.reason = 'initializing'
        self.desired = np.zeros(3)
        self.lost_since = None
        self.localization_reseeds = 0
        self.motion_recovery = MotionRecovery()

    @staticmethod
    def _rotation(value):
        if (value.shape != (3, 3) or not np.isfinite(value).all() or
                not np.allclose(value.T@value, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(value), 1, atol=1e-4)):
            raise ValueError('Expected a proper rotation matrix')

    @property
    def pose(self):
        return np.r_[self.xy, self.yaw]

    def set_attitude(self, timestamp, world_from_body):
        matrix = np.asarray(world_from_body, dtype=float)
        self._rotation(matrix)
        if not np.isfinite(timestamp) or (self.last_imu_time is not None and timestamp < self.last_imu_time):
            raise ValueError('Nonmonotonic IMU timestamp')
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        initial = np.eye(3)
        initial[:2, :2] = rotation2(-self.initial_yaw)
        self.attitude = initial@matrix
        yaw = np.arctan2(self.attitude[1, 0], self.attitude[0, 0])
        self.turn += (yaw-self.yaw+np.pi) % (2*np.pi)-np.pi
        self.yaw = float(yaw)
        self.last_imu_time = float(timestamp)

    def _observed_map_white(self, timestamp):
        """Keep a stationary lane reference available while visual motion is lost.

        A corridor is created only from repeated observed straight lane fits near
        boxes. Its direction and center are measurements; the 50 cm width is the
        same task dimension used by perception and planning, not a field pose.
        """
        heading = self.navigator.corridor_heading
        point = self.navigator.corridor_point
        alongside = any(-.4 < local[0] < .85 and abs(local[1]) < .8
                        for local in ((box['center']-self.xy)@rotation2(self.yaw)
                                      for box in self.perception.obstacle_memory))
        if (alongside and heading is not None and point is not None
                and np.isfinite(heading) and np.isfinite(point).all()):
            tangent = np.array([np.cos(heading), np.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            # Recenter the finite reference along the corridor as the robot
            # advances. Keep its measured perpendicular offset unchanged.
            center = point+tangent*((self.xy-point)@tangent)-self.xy
            along = np.linspace(-.4, 2.7, 311)
            return np.vstack([center+along[:, None]*tangent+side*normal
                              for side in (-.25, .25)])
        # Once tracking is lost no observations are fused. Expiring the last
        # accepted lines by wall-clock time alone would prevent later recovery
        # even when the robot has stopped at precisely the same place.
        reference_time = timestamp if self.lost_since is None else self.lost_since
        whites = [points for stamp, points, _ in self.perception.white_memory
                  if reference_time-stamp < 8.]
        return np.vstack(whites)-self.xy if whites else np.empty((0, 2))

    def observe(self, frame, world_from_body_at_frame=None, imu_time=None):
        stamp = self.last_imu_time if imu_time is None else imu_time
        if self.last_imu_time is None or stamp is None or abs(frame.time-stamp) > .15:
            raise ValueError('No time-aligned attitude for camera frame')
        frame_attitude = self.attitude
        if world_from_body_at_frame is not None:
            matrix = np.asarray(world_from_body_at_frame, dtype=float)
            self._rotation(matrix)
            initial = np.eye(3)
            initial[:2, :2] = rotation2(-self.initial_yaw)
            frame_attitude = initial@matrix
        if self.last_frame_time is not None and frame.time <= self.last_frame_time:
            raise ValueError('Nonmonotonic camera timestamp')
        if frame.depth.shape != self.perception.rays.shape[:2] or frame.rgb.shape != (*frame.depth.shape, 3):
            raise ValueError('Camera resolution differs from calibrated intrinsics')
        usable_pixels = (frame.rgb.max(axis=2)>5) & (frame.rgb.min(axis=2)<250)
        self.rgb_usable = bool(np.count_nonzero(usable_pixels) >= frame.depth.size*.01)
        if self.last_frame_time is not None and frame.time-self.last_frame_time > self.observation_lease:
            self.good_frames = 0
        tracking_frame = frame if self.rgb_usable else CameraFrame(frame.time, np.zeros_like(frame.rgb), frame.depth)
        xy = self.odometry.update(tracking_frame, frame_attitude, self.xy)
        if not self.odometry.accepted and self.perception.obstacle_memory:
            # Relocalize against geometry previously measured by this robot.
            # A partial/blank keyframe must not make a known box disappear.
            white=self._observed_map_white(frame.time)
            near_edges=[];far_edges=[]
            along=np.linspace(-.10,.10,31)
            for box in self.perception.obstacle_memory:
                axes=box.get('axes',np.eye(2))
                for axis in (0,1):
                    for side in (-.10,.10):
                        local=np.zeros((len(along),2));local[:,axis]=side;local[:,1-axis]=along
                        points=local@axes+box['center']-self.xy
                        normal=np.sign(side)*axes[axis]
                        (near_edges if np.mean(points,axis=0)@normal<0 else far_edges).append(points)
            geometry=self.odometry.geometry
            reference=[geometry.sample(white),*[geometry.sample(np.vstack(edges)) if edges else np.empty((0,2),np.float32) for edges in (near_edges,far_edges)]]
            measured=geometry.project(tracking_frame,frame_attitude)
            correction=geometry.register(reference,measured)
            if correction is not None and np.linalg.norm(correction)<.09:
                xy=self.xy+correction
                self.odometry.accepted=True
                self.odometry.inliers=geometry.support
                self.odometry.method='observed_map'
                self.odometry.previous=None
                # Seed the next image tracker at the recovered metric pose.
                self.odometry.update(tracking_frame,frame_attitude,xy)
                self.odometry.accepted=True
                self.odometry.inliers=geometry.support
                self.odometry.method='observed_map'
                self.odometry.relocalized=True
        # Search recent textured keyframes after an interruption, before the
        # local tracker gives up its segment. Hidden boxes stay in odom.
        previous_time = self.odometry.previous_time
        if (not self.odometry.accepted and self.rgb_usable and previous_time is not None
                and frame.time-previous_time > .25):
            recovered = self.floor_map.update(frame, frame_attitude,
                self.odometry.geometry.ground.result, self.xy, self.distance, recovering=True)
            if recovered is not None:
                xy = recovered
                self.odometry.previous = None
                self.odometry.update(frame, frame_attitude, xy)
                self.odometry.accepted = True
                self.odometry.relocalized = True
                self.odometry.method = 'floor_keyframe'
        if self.odometry.accepted:
            self.lost_since = None
        else:
            if self.lost_since is None:self.lost_since=frame.time
            # In an empty corridor, restart a purely local coordinate segment
            # after stopping. Never move/reseed a map containing hidden boxes
            # or a tracked finish target; their relative positions matter.
            if (frame.time-self.lost_since>.6 and not self.perception.obstacle_memory
                    and self.navigator.finish_target is None and not np.any(self.adapter.target)):
                self.odometry.previous=None
                self.odometry.previous_time=None
                self.perception.white_memory.clear()
                self.perception.previous_lane=None
                self.lost_since=frame.time
                self.localization_reseeds+=1
                self.floor_map.start_segment()
        self.distance += float(np.linalg.norm(xy-self.xy))
        self.xy = xy
        frame_pose = np.r_[self.xy, np.arctan2(frame_attitude[1, 0], frame_attitude[0, 0])]
        if self.odometry.accepted and self.rgb_usable:
            self.floor_map.update(frame, frame_attitude, self.odometry.geometry.ground.result,
                                  self.xy, self.distance)
            self.observation = self.perception.update(frame, frame_pose, frame_attitude,
                ground_plane=self.odometry.geometry.ground.result)
            if self.observation.lane is not None and self.observation.lane_observed:
                self.last_lane_observation_time = float(frame.time)
            self.navigator.gap.translate_map(getattr(self.perception,'map_translation',np.zeros(2)))
        else:
            # Neither erase hidden boxes nor fuse pixels at an unknown pose.
            # Retain the accepted reference, inhibit motion, and recover first.
            obstacles = []
            for record in self.perception.obstacle_memory:
                local = (record['center']-self.xy)@rotation2(frame_pose[2])
                obstacles.append([*local, *record['size']])
            self.observation = Perception(frame.time, None, np.empty((0,2)),
                np.asarray(obstacles).reshape(-1,4), np.empty((0,2)), 0., frame.rgb.copy())
        if self.odometry.relocalized:
            # A pose correction is not physical body velocity. Reacquire
            # three trusted images before resuming the local motion servo.
            self.good_frames=0
            self.navigator.velocity_pose=frame_pose[:2].copy()
            self.navigator.velocity_time=frame.time
            self.navigator.measured_velocity[:]=0.
        self.navigator.observe(self.observation, frame_pose, self.turn,
                               motion_observed=self.odometry.accepted and not self.odometry.relocalized)
        has_route = self.observation.lane is not None or self.navigator.gap.phase!='idle'
        valid = has_route and self.odometry.accepted and self.rgb_usable
        self.good_frames = self.good_frames+1 if valid else 0
        self.last_frame_time = float(frame.time)
        return self.observation

    def command(self, timestamp, policy_feedback=None, enabled=True):
        if not np.isfinite(timestamp) or (self.last_control_time is not None and timestamp <= self.last_control_time):
            raise ValueError('Nonmonotonic control timestamp')
        dt = .02 if self.last_control_time is None else min(.1, timestamp-self.last_control_time)
        self.last_control_time = float(timestamp)
        # RT10: the strings below are telemetry only. No condition zeroes the
        # request; the mission's 300 s limit is enforced by the supervisor.
        if not enabled:
            self.reason = 'disabled'
        elif self.last_imu_time is None or not 0 <= timestamp-self.last_imu_time <= .15:
            self.reason = 'stale_imu'
        elif self.last_frame_time is None or not 0 <= timestamp-self.last_frame_time <= self.observation_lease:
            self.reason = 'stale_camera'
        elif not self.rgb_usable:
            self.reason = 'unusable_rgb'
        elif self.good_frames < 3:
            self.reason = 'waiting_for_trusted_visual_motion_and_lane'
        else:
            self.desired = self.navigator.update(timestamp, self.pose, dt)
            self.reason = (self.navigator.mode if self.navigator.mode in
                ('lost_vision', 'blocked', 'finished', 'initializing') else 'ready')
        self.motion_recovery.update(timestamp,self.desired,self.pose,
            self.navigator.measured_velocity,False)
        return self.adapter.update(self.desired, policy_feedback, True)

    def diagnostic(self):
        return dict(reason=self.reason, mode=self.navigator.mode, pose=self.pose.tolist(),
                    map_xy=(self.xy+self.floor_map.offset).tolist() if self.floor_map.attached else None,
                    floor_keyframes=len(self.floor_map.keys), map_revisits=self.floor_map.revisits,
                    map_tracking_updates=self.floor_map.local_updates,
                    map_recoveries=self.floor_map.recoveries,
                    turn=self.turn, distance=self.distance, vo_inliers=self.odometry.inliers,
                    vo_accepted=self.odometry.accepted, good_frames=self.good_frames,
                    velocity_valid=self.navigator.velocity_valid,
                    odometry_method=self.odometry.method,
                    relocalized=self.odometry.relocalized,
                    last_lane_observation_time=self.last_lane_observation_time,
                    rgb_usable=self.rgb_usable,
                    floor_tracking_reason=self.odometry.floor.reason,
                    floor_tracking_quality=dict(self.odometry.floor.quality),
                    gap_phase=self.navigator.gap.phase,gap_error=self.navigator.gap.error.tolist(),
                    gap_recoveries=self.navigator.gap.recoveries, gap_hold=self.navigator.gap.hold_reason,
                    motion_recovery=self.motion_recovery.phase,
                    motion_recoveries=self.motion_recovery.recoveries,
                    gap_target=None if self.navigator.gap.target is None else self.navigator.gap.target.tolist(),
                    measured_velocity=self.navigator.measured_velocity.tolist(),
                    effective_lane=None if getattr(self.navigator,'effective_lane',None) is None else self.navigator.effective_lane.tolist(),
                    corridor_heading=self.navigator.corridor_heading,
                    corridor_point=None if self.navigator.corridor_point is None else self.navigator.corridor_point.tolist(),
                    finish_target=None if self.navigator.finish_target is None else self.navigator.finish_target.tolist(),
                    finish_normal=None if self.navigator.finish_normal is None else self.navigator.finish_normal.tolist(),
                    ground=self.odometry.geometry.ground.result.diagnostic(),
                    localization_reseeds=self.localization_reseeds,
                    desired_velocity=self.desired.tolist(),
                    target_policy_velocity=self.adapter.target.tolist(),
                    predicted_policy_velocity=self.adapter.filtered.tolist())


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
    """
    def __init__(self, width, height, intrinsics, position, rotation, speed, observation_lease):
        self.crates=BoxServo()
        self._box_publication=None
        self.motion_watch=BoxMotionWatch()
        self.motion_survey=None
        self.motion_status={}
        self.tracked = NavigationRuntime(width, height, intrinsics, position, rotation, speed, observation_lease)
        self.local = CameraPerception(width, height, 80., position, rotation, intrinsics)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = self.map_offer = None
        self.mapping_defer_until = 0.
        self.mapping_budget_skips = 0
        self.snapshot = self.tracked.diagnostic()
        self.missed_previous = False
        self.recovery_stamp = self.recovery_sample = None
        self.snapshot_time = None
        self.lease = observation_lease
        self.speed = speed
        self.adapter = AMPCommandAdapter()
        self.recovery = MotionRecovery()
        self.initial_yaw = self.yaw = self.last_imu_time = None
        self.turn = 0.
        self.last_frame_time = self.lane_time = self.lane_yaw = None
        self.scene = 'unknown_depth'
        self.reference_time = None
        self.edge_correction = None
        self.inside_lane_start = True
        self.lane = self.support = None
        self.steering_heading = None
        self.turning_to_lane = False
        self.lane_turn = LaneTurn()
        self.unconfirmed_sites = {}
        self.clear_frames = 0
        self.box_latched = self.saw_box = self.finish_latched = False
        self.tracking_control = self.finished = self.holding_lane = False
        self.last_control_time = self.deadline = None
        self.desired = np.zeros(3)
        # Last translational request. RT10 never publishes zero once walking:
        # a missing plan holds this velocity instead of standing still.
        self.last_drive = np.zeros(2)
        self.reason = 'initializing'
        # Local RT13 patch: single-box avoidance, passage gating and the lane
        # plan timestamp used to bound blind coasting.
        self.ride_side = 0
        self.ride_y = None
        self.ride_meas_y = None
        self.ride_count = 0
        self.ride_now = 0
        self.white_points = None
        self.tape_counts = (0, 0)
        self.tape_fit = None
        self.box_hold = False
        self.box_hold_done = False
        self.blind_since = None
        self.ride_pending = 0
        self.ride_pending_since = 0.
        self.ride_stamp = None
        self.boxes_seen = ()
        self.raised_seen = None
        self.passage_required = False
        self.lane_plan_stamp = None

    def supervise_output(self,t,vector,reason,imu_stamp,yaw,zero_since=None):
        """Run once per actual 50 Hz output tick, after all ordinary plan cutoffs."""
        if self.box_hold:
            # Operator hold: no pivot pulse, no restart request, nothing.
            return vector, reason, None
        survey=self.motion_survey
        box=self.crates.frame
        # Local RT13 patch: an in-place pivot is only appropriate for the
        # multi-box passage, not while avoiding a single box on the lane.
        in_boxes=bool(box is not None and not self.crates.done and self.passage_required)
        capture=relative=landmark=pivot_stamp=None
        clear=False
        heading=0.
        if survey is not None:
            pivot_stamp,clear,wide_clear,heading=survey
            capture=pivot_stamp
        if in_boxes:
            view=box.view
            if clear and view is not None:
                # Lane inspection finishes before full crate association on the
                # Jetson. A newly measured wider empty sweep can bridge that
                # computation only with the previous safe stance still recent.
                same=view.stamp==pivot_stamp
                clear=view.turn_space and (same or (wide_clear and 0<pivot_stamp-view.stamp<=.55))
            else:
                clear=False
            if view is not None:
                if view.relative is not None:
                    capture=view.stamp
                    relative=np.asarray(view.relative)@rotation2(box.heading).T
                    landmark=view.identity
            if abs(vector[2])>.08:
                heading=vector[2]
        if relative is None and self.snapshot_time is not None and self.initial_yaw is not None and self.snapshot.get('vo_accepted') and 0<=t-self.snapshot_time<=.55:
            # The inverse RGB-D odometry displacement has the same sign as
            # movement of a stationary landmark relative to this robot.
            capture=self.snapshot_time
            relative=-np.asarray(self.snapshot['pose'][:2])@rotation2(self.initial_yaw).T
            landmark='lane_odometry'
        leg=self.crates.latest_leg
        leg_xy=leg_velocity=leg_stamp=None
        if leg is not None:
            leg_stamp,leg_xy,leg_velocity=leg
        turn=abs(vector[2])>.17 and np.linalg.norm(vector[:2])<.09
        feedback=self.motion_watch.update(t,capture_stamp=capture,relative=relative,landmark=landmark,
                yaw=yaw,imu_stamp=imu_stamp,leg_xy=leg_xy,leg_velocity=leg_velocity,leg_stamp=leg_stamp,
                pivot_clear=clear,pivot_stamp=pivot_stamp,scan_direction=float(np.sign(heading)),
                turning=turn,passage_valid=bool(in_boxes and not self.crates.passage_wait),zero_since=zero_since)
        # A walking lane follower can have no mapping frames when steering is
        # using the whole processing budget. Keep its observed gait in that case.
        lane_walking=(not in_boxes and zero_since is None and np.linalg.norm(vector[:2])>.08
                      and leg is not None and 0<=t-leg_stamp<.15 and np.linalg.norm(leg_velocity)>.065)
        self.motion_status=dict(motion_still_s=round(feedback.still_s,3),motion_watch_reason=feedback.reason,
                                guarded_zero_s=None if zero_since is None else round(max(0.,t-zero_since),3))
        if feedback.yaw_request and not lane_walking and clear and 0<=t-pivot_stamp<=.38:
            self.motion_status['motion_pivot_rad_s']=float(feedback.yaw_request)
            return (0.,0.,float(feedback.yaw_request)),'motion_reobserve',min(pivot_stamp+.38,imu_stamp+.15)
        return vector,reason,None

    def publish_velocity(self,t,planned,attitude):
        hook=self.freeze_publish_velocity()
        return planned if hook is None else hook(t,planned,attitude)

    def freeze_publish_velocity(self):
        return self._box_publication

    def update_legs(self, stamp, joints, attitude):
        return self.crates.update_legs(stamp,joints,attitude)

    def set_attitude(self, timestamp, rotation):
        NavigationRuntime._rotation(rotation)
        if not np.isfinite(timestamp) or (self.last_imu_time is not None and timestamp<self.last_imu_time):
            raise ValueError('Nonmonotonic IMU timestamp')
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        if self.yaw is not None:
            self.turn += (yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
        self.yaw, self.last_imu_time = yaw, timestamp
        self.attitude = rotation.copy()

    def _track(self, frame, attitude, turn):
        r = self.tracked
        if r.initial_yaw is None:
            r.initial_yaw = self.initial_yaw
        if r.last_imu_time is None or frame.time>=r.last_imu_time:
            r.set_attitude(frame.time, attitude)
        r.turn = turn
        r.observe(frame, attitude, frame.time)
        state = r.diagnostic()
        state['nearby_box'] = any(-.46<b[0]<1.15 and abs(b[1])<.85 for b in r.observation.obstacles)
        return frame.time, state

    def _collect(self):
        if self.pending is not None:
            self.snapshot_time, self.snapshot = self.pending.result()
            self.pending = None

    def close(self, path):
        try:
            self._collect()
        finally:
            self.worker.shutdown(wait=True)
        self.tracked.floor_map.save(path)

    def dispatch_mapping(self, tick, now, frame_period):
        """Offer spare CPU time after the guarded velocity request is available."""
        offer, self.map_offer = self.map_offer, None
        if now-tick > frame_period-.02:
            # Leave the following capture and a 50 Hz guard tick for steering.
            self.mapping_defer_until = now+frame_period
        if offer is not None:
            if now>=self.mapping_defer_until:
                self.pending = self.worker.submit(self._track, *offer)
            else:
                self.mapping_budget_skips += 1

    def _review_raised_sites(self, position, attitude, lane, timestamp):
        keys = list(self.unconfirmed_sites)
        sites = np.array([self.unconfirmed_sites[key][0] for key in keys])
        sites[:,:2] -= position
        visible = self.local.previous_sites_clear(sites, attitude, lane)
        self.clear_frames = 0
        for key, clear in zip(keys, visible):
            point, count, first = self.unconfirmed_sites[key]
            if timestamp-first>3*self.lease:
                count = 0
            if clear:
                if count==0:
                    first = timestamp
                count += 1
            if count>=3:
                del self.unconfirmed_sites[key]
            else:
                self.unconfirmed_sites[key] = (point, count, first)
                self.clear_frames = max(self.clear_frames, count)

    def observe(self, frame, attitude, imu_time, inhibit, defer_mapping=False):
        self.map_offer = None
        if not np.isfinite(frame.time) or abs(frame.time-imu_time)>.15 or self.last_imu_time is None:
            raise ValueError('No time-aligned attitude for camera frame')
        if self.last_frame_time is not None and frame.time<=self.last_frame_time:
            raise ValueError('Nonmonotonic camera timestamp')
        if frame.depth.shape!=self.local.rays.shape[:2] or frame.rgb.shape!=(*frame.depth.shape,3):
            raise ValueError('Camera resolution differs from calibrated intrinsics')
        self.last_frame_time = frame.time
        if self.pending is not None and self.pending.done():
            self._collect()
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
        self.motion_survey=(frame.time,bool(getattr(self.local,'pivot_clear',False)),
                            bool(getattr(self.local,'pivot_wide_clear',False)),observation.steering_heading or 0.)
        # A running Pro can drift off the inside of a bend while its depth
        # coverage falls short of the full straight-ahead band. The two
        # independently measured floor tapes still specify a direction back
        # toward the lane. Correct only sizeable observed offsets.
        previous_correction = self.edge_correction
        if (previous_correction is None or frame.time-previous_correction[0]>.85 or
                self.box_latched or self.unconfirmed_sites or self.scene=='raised_object'):
            self.edge_correction = None
        if (not self.box_latched and not self.unconfirmed_sites and
                self.local.lane_basis=='paired_edges' and observation.lane_observed and
                self.scene in ('clear','turn_clear','unknown_depth') and self.local.ground.result.valid):
            curve=observation.lane
            near,far=observation.lane_support
            slope=np.polyval(np.polyder(curve), np.clip(.40,near,far))
            side=curve[-1]/np.hypot(1.,curve[-2])
            trigger=.13 if previous_correction is not None else .22
            if abs(side)>trigger and abs(side)<.65:
                self.edge_correction=(frame.time,capture_yaw,np.arctan(slope),side)
                self.lane, self.support = curve.copy(), observation.lane_support
                self.reference_time, self.lane_yaw = frame.time, capture_yaw
            else:
                self.edge_correction=None
        image_ok = np.count_nonzero((frame.rgb.max(axis=2)>5)&(frame.rgb.min(axis=2)<250))>=frame.depth.size*.01
        if not image_ok:
            self.scene = 'unusable_rgb'
        was_tracking = self.tracking_control
        recent_box = (self.snapshot_time is not None and frame.time-self.snapshot_time<=self.lease
                      and self.snapshot.get('nearby_box', False))
        raised = self.scene=='raised_object'
        if recent_box:
            self.box_latched = self.saw_box = True
        if (self.turn>5.5 and len(observation.finish)) or self.snapshot.get('finish_target') is not None:
            self.finish_latched = True
        self.tracking_control = raised or bool(self.unconfirmed_sites) or self.box_latched or self.finish_latched
        # Local RT13 patch: only a genuine multi-box corridor keeps the
        # passage servo in charge. A single box is avoided while the lane is
        # still followed, so the body returns to the line by itself.
        self.boxes_seen = observation.obstacles
        # Local RT13 patch: the height map reports raised cells in the level
        # (AHRS) frame, while everything below reasons in heading-aligned body
        # coordinates. Rotate them by the same yaw the white points use,
        # otherwise a box ahead lands at negative x whenever the body is not
        # facing the initial heading.
        sites = self.local.raised_sites
        if sites is not None and len(sites) and attitude is not None:
            yaw = float(np.arctan2(attitude[1, 0], attitude[0, 0]))
            rotated = np.array(sites, dtype=float, copy=True)
            rotated[:, :2] = rotated[:, :2] @ rotation2(yaw)
            sites = rotated
        self.raised_seen = sites
        self.white_points = (observation.near_white if observation.near_white is not None
                             else observation.current_white)
        self.ride_count, self.ride_meas_y = self._cluster_measure()
        self.ride_y = self.ride_meas_y
        self.passage_required = sum(1 for b in self.boxes_seen
                                    if .15 < b[0] < 2.2 and abs(b[1]) < .95) >= 2
        self.holding_lane = False
        if self.tracking_control:
            if (not was_tracking or self.scene in ('unusable_rgb','unknown_depth')
                    or (self.scene=='raised_object' and not self.box_latched)):
                inhibit()
            self._collect()
            self.snapshot_time, self.snapshot = self._track(frame, attitude, self.turn)
            r = self.tracked
            # Tracker XY is expressed relative to the initial heading. Store
            # footprints in AHRS axes and account for measured braking drift.
            position = r.xy@rotation2(self.initial_yaw).T
            if raised:
                sites = self.local.raised_sites.copy()
                sites[:,:2] += position
                for point in sites:
                    key = tuple(np.floor(point[:2]/.04).astype(int))
                    # A repeat hit revokes prior clearance for this cell only.
                    self.unconfirmed_sites[key] = (point, 0, frame.time)
            if r.odometry.accepted and r.observation.lane_observed and r.observation.lane_support is not None and r.observation.lane is not None:
                self.lane, self.support = r.observation.lane.copy(), r.observation.lane_support
                self.reference_time, self.lane_yaw = frame.time, capture_yaw
            boxes = r.observation.obstacles
            # Count as passed only a tracked box that is actually behind the body.
            nearby = any(-.46<b[0]<1.25 and abs(b[1])<.85 for b in boxes)
            passed = any(b[0]<-.46 and abs(b[1])<.85 for b in boxes)
            if r.odometry.accepted and nearby:
                self.box_latched = self.saw_box = True
            if self.box_latched or self.finish_latched or r.navigator.gap.phase!='idle':
                self.inside_lane_start = False
            if nearby and self.saw_box:
                self.unconfirmed_sites.clear()
            if self.unconfirmed_sites and not self.box_latched:
                if (self.scene in ('clear','turn_clear') and observation.lane_observed
                        and not nearby and r.navigator.gap.phase=='idle'):
                    self._review_raised_sites(position, attitude, observation.lane, frame.time)
                else:
                    self.clear_frames = 0
            if not self.unconfirmed_sites:
                self.clear_frames = 0
            if (self.box_latched and self.saw_box and passed and not nearby and self.scene=='clear'
                    and r.odometry.accepted and r.navigator.gap.phase=='idle' and observation.lane_observed):
                self.box_latched = self.saw_box = False
            self.finish_latched |= r.navigator.finish_target is not None
            self.tracking_control = bool(self.unconfirmed_sites) or self.box_latched or self.finish_latched
        elif self.pending is None:
            # Drop optional mapping frames if busy. Never enqueue a backlog in front of steering.
            offer = (frame, attitude.copy(), self.turn)
            if defer_mapping:
                self.map_offer = offer
            else:
                self.pending = self.worker.submit(self._track, *offer)
        exit_lane_seen=bool(observation.lane_observed and observation.lane is not None
                            and self.scene in ('clear','turn_clear') and self.local.ground.result.valid)
        exit_lane_heading=None
        if exit_lane_seen and self.crates.phase=='lane_entry' and observation.lane_support is not None:
            near,far=observation.lane_support
            look=np.clip(.40,near,far)
            heading=capture_yaw+np.arctan(np.polyval(np.polyder(observation.lane),look))
            if np.isfinite(heading):
                exit_lane_heading=heading
        self.crates.observe(frame,attitude,self.local,self.tracked,self.initial_yaw,observation.lane,
                            exit_lane_seen=exit_lane_seen,exit_lane_heading=exit_lane_heading)
        if self.crates.done:
            self.tracking_control=False
            self.box_latched=self.saw_box=False
            self.unconfirmed_sites.clear()
        if self.tracking_control and self.passage_required and not (self.crates.phase=='lane_entry' and exit_lane_seen):
            self.lane_time = None
            return self.tracked.observation
        if self.scene in ('clear','turn_clear') or (self.scene=='raised_object' and not self.passage_required):
            # Local RT13 patch: a box in view no longer withdraws the lane.
            # The edge reconstruction still describes the lane, so keep
            # following it and ride the opposite tape with _ride_offset().
            if observation.lane_observed:
                self.steering_heading = observation.steering_heading
                self.lane, self.support = observation.lane.copy(), observation.lane_support
                self.reference_time = self.lane_time = frame.time
                self.lane_yaw = capture_yaw
            elif (self.lane_time is not None and reference is not None
                  and not self.turning_to_lane and not self.missed_previous):
                self.holding_lane = True
            else:
                self.lane_time = None
                inhibit()
        else:
            self.lane_time = None
            inhibit()
        self.missed_previous = not observation.lane_observed
        return observation

    def _tape_target(self, now):
        """Position and slope of the tape the body should ride, or None.

        Only the near segment is fitted: a corner further ahead must not bend
        the line. The white points are split into left/right clusters by their
        lateral gap instead of by sign, because a body riding a tape has that
        tape on both sides of its own axis. A fit with a poor residual or a
        short span is rejected, and the previous fit may be held for one second
        without contributing any lateral request.
        """
        if not self.ride_now or self.passage_required or self.ride_count < 8:
            return None
        points = self.white_points
        if points is None or len(points) < 10:
            return None
        points = np.asarray(points, dtype=float)
        near = points[(points[:,0] > .25) & (points[:,0] < .80)]
        self.tape_counts = (int(np.count_nonzero(np.sign(near[:,1]) == self.ride_now)),
                            int(np.count_nonzero(np.sign(near[:,1]) == -self.ride_now)))
        cluster = None
        if len(near) >= 10:
            order = np.sort(near[:,1])
            gaps = np.diff(order)
            split = None
            if len(gaps) and np.max(gaps) >= .06:
                k = int(np.argmax(gaps))
                split = .5*(order[k]+order[k+1])
            pieces = ([near] if split is None else
                      [near[near[:,1] <= split], near[near[:,1] > split]])
            for piece in pieces:
                if len(piece) < 10:
                    continue
                median = float(np.median(piece[:,1]))
                if abs(median) < .04 or np.sign(median) != self.ride_now:
                    continue
                cluster = piece
                break
        if cluster is None:
            if self.tape_fit is not None and now-self.tape_fit[2] <= TAPE_HOLD_S:
                return self.tape_fit[0], self.tape_fit[1], False
            return None
        slope, intercept = np.polyfit(cluster[:,0], cluster[:,1], 1)
        residual = cluster[:,1]-(slope*cluster[:,0]+intercept)
        if np.sqrt(np.mean(residual**2)) > .02 or np.ptp(cluster[:,0]) < .18:
            if self.tape_fit is not None and now-self.tape_fit[2] <= TAPE_HOLD_S:
                return self.tape_fit[0], self.tape_fit[1], False
            return None
        offset = float(slope*.45+intercept)
        self.tape_fit = (offset, float(slope), now)
        return offset, float(slope), True

    def _cluster_measure(self):
        """Lateral centre of the raised cluster in front, and how many cells it has.

        Measured every frame so the decision is visible even when the box hides
        the lane. The far half is used because the near face shifts sideways as
        the body passes.
        """
        sites = self.raised_seen
        if sites is None or not len(sites):
            return 0, None
        points = np.asarray(sites)
        # Match the perception window: anything the height map reports from
        # 0.22 m outwards counts. A 0.30 m floor silently discarded boxes that
        # were already close, which is exactly when the side decision matters.
        near = points[(points[:,0] > .22) & (points[:,0] < 1.80)
                      & (np.abs(points[:,1]) < .80)]
        # The side decision only needs to know which side the box is on, so a
        # weaker cluster is accepted; the lateral request still needs a body
        # that can move, which the caller checks.
        if len(near) < 8:
            return len(near), None
        far = near[near[:,0] >= np.percentile(near[:,0], 60)]
        if not len(far):
            return len(near), None
        return len(near), float(np.median(far[:,1]))

    def _ride_decision(self, now):
        """Give the boundary tape opposite the obstacle, or keep the current one.

        The box model is deliberately not used here: a raised cluster in the
        corridor is enough. A weak reading keeps the tape the body is already
        riding so it does not weave between decisions.
        """
        y = self.ride_meas_y
        if y is None:
            # Between obstacles keep riding the current tape; leave the box
            # zone only after a quiet period.
            if self.ride_stamp is not None and now-self.ride_stamp > RIDE_HOLD_S:
                self.ride_side = self.ride_pending = 0
                self.ride_stamp = None
            return self.ride_side
        self.ride_stamp = now
        if abs(y) <= RIDE_DEADBAND:
            # Boxes are placed against a tape, so this is only a weak or
            # ambiguous reading: hold the side already in use.
            if not self.ride_side:
                self.ride_side = 1
            self.ride_pending = 0
            return self.ride_side
        desired = -1 if y > 0 else 1
        if not self.ride_side:
            # First obstacle: adopt the opposite tape at once.
            self.ride_side = desired
            self.ride_pending = 0
        elif desired == self.ride_side:
            self.ride_pending = 0
        elif self.ride_pending != desired:
            self.ride_pending = desired
            self.ride_pending_since = now
        elif now-self.ride_pending_since >= RIDE_SWITCH_S:
            self.ride_side = desired
            self.ride_pending = 0
        return self.ride_side

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
        elif self.tracking_control and self.scene=='turn_clear':
            self.reason = ('checking_raised_object' if self.unconfirmed_sites and not self.box_latched
                           else 'unknown_depth')
        elif self.tracking_control:
            self.reason = 'waiting_for_box_geometry'
        elif not fresh_lane:
            self.reason = 'lost_vision'
        else:
            self.reason = 'ready'
        self.crates.commit_frame()
        guided=self.crates.command(timestamp,self.yaw)
        if not self.passage_required:
            # Local RT13 patch: the crates servo only steers the multi-box
            # passage; a single box is handled by lane following + avoidance.
            guided=None
        # Local RT13 patch: one operator confirmation per run, taken at the
        # moment the box-zone controller is about to take over (a box measured
        # and a side chosen). After the release the latch stays cleared, so it
        # never stops again in the same run.
        self.ride_now = self._ride_decision(timestamp)
        entering = bool(self.ride_now) and self.ride_meas_y is not None
        if entering and not self.box_hold_done and not self.box_hold:
            self.box_hold = True
        if self.box_hold:
            if os.path.exists(BOX_HOLD_FLAG):
                try:
                    os.unlink(BOX_HOLD_FLAG)
                except OSError:
                    pass
                self.box_hold = False
                self.box_hold_done = True
            else:
                self.desired[:] = 0.
                self.last_drive[:] = 0.
                self.reason = 'box_zone_hold'
                return self.adapter.update(self.desired, enabled=True)
        self._box_publication=None
        if self.crates.done and self.crates.exit_search_used and not getattr(self,'exit_drive_reset',False):
            self.exit_drive_reset=True
            self.desired=np.zeros(3)
            self.last_drive=np.zeros(2)
            self.lane_turn=LaneTurn()
        if self.crates.done and fresh_lane:
            self.tracking_control=False
        if guided is not None:
            frame=self.crates.frame
            self.deadline=self.crates.observation_deadline(frame,min(self.deadline,frame.stamp+self.lease))
            self._box_publication=self.crates.publication(frame,self.deadline)
            self.desired=guided
            self.lane_turn.pause(timestamp)
            self.reason=('box_passage_unconfirmed' if self.crates.passage_wait else
                         'box_entry_hold' if self.crates.entry_hold else 'box_'+self.crates.phase)
            if self.crates.view_override and self.crates.view_recovery.action in ('scan','retreat','realign'):
                self.reason='box_view_'+self.crates.view_recovery.action
            return self.adapter.update(self.desired,enabled=True)
        if (enabled and self.reason not in ('stale_imu', 'stale_camera', 'unusable_rgb') and
                not self.tracking_control and self.edge_correction is not None and
                0. <= timestamp - self.edge_correction[0] <= .85):
            _, observation_yaw, heading, side = self.edge_correction
            rotation_error=(observation_yaw+heading-self.yaw+np.pi)%(2*np.pi)-np.pi
            self.desired[:]=[min(self.speed,.18),np.clip(1.6*side,-.23,.23),
                             np.clip(2.4*rotation_error,-.72,.72)]
            self.last_drive=self.desired[:2].copy()
            self.lane_turn.pause(timestamp)
            self.reason='return_to_observed_lane'
            return self.adapter.update(self.desired,enabled=True)
        # Local RT13 patch: the box-zone tape follower may only take over when
        # the lane itself is unavailable. A raised blob outside the box zone
        # (a leg, a stray object) must not override a healthy lane plan.
        tape = self._tape_target(timestamp) if not fresh_lane else None
        if tape is not None:
            # Local RT13 patch: in the box zone follow the chosen tape itself,
            # half a body width each side of it, instead of the lane centre.
            # A held fit contributes no lateral request.
            offset, slope, fresh = tape
            lateral = float(np.clip(1.2*offset, -.18, .18)) if fresh else 0.
            self.desired[:] = [min(self.speed, .15), lateral,
                               float(np.clip(1.2*np.arctan(slope), -.25, .25))]
            self.last_drive = self.desired[:2].copy()
            self.lane_plan_stamp = timestamp
            self.blind_since = None
            self.reason = 'box_zone_tape'
            return self.adapter.update(self.desired, enabled=True)
        if self.tracking_control and self.passage_required:
            self.turning_to_lane = False
            r = self.tracked
            known = any(-.46<b[0]<1.35 and abs(b[1])<.85 for b in r.observation.obstacles)
            if (not known) and r.navigator.gap.phase=='idle':
                self.reason = ('checking_raised_object' if self.unconfirmed_sites
                               else 'waiting_for_box_geometry')
            else:
                if self.last_imu_time>=r.last_imu_time:
                    r.set_attitude(self.last_imu_time, self.attitude)
                r.command(timestamp)
                self.finished = r.navigator.finished
                self.desired = r.desired.copy()
                self.reason = r.reason
                if r.navigator.gap.drive_until is not None:
                    self.deadline = min(self.deadline, r.navigator.gap.drive_until)
        elif fresh_lane:
            self.deadline = min(self.deadline, self.lane_time+self.lease)
            self.lane_plan_stamp = timestamp
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
            side = float(np.clip(side + self.ride_now*RIDE_HALF_WIDTH_M, -.45, .45))
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
        # With no fresh lane plan the held yaw is cleared, so an interrupted
        # turn cannot become an unbounded spin.
        if not fresh_lane and not self.passage_required:
            self.desired[2] = 0.
        # Local RT13 patch: inside the box zone the box itself hides the near
        # lane, so waiting for a lane plan means waiting forever. With a stable
        # raised cluster in front, walk slowly and shift away from it until the
        # lane comes back. Bounded in time and only with a measured cluster.
        cluster_now = bool(self.ride_now and self.ride_meas_y is not None)
        recent = (self.ride_stamp is not None and timestamp-self.ride_stamp <= BOX_RECENT_S)
        if (cluster_now or recent) and not self.passage_required and not fresh_lane:
            if self.blind_since is None:
                self.blind_since = timestamp
            elapsed = timestamp-self.blind_since
            limit = BLIND_BOX_S if cluster_now else BLIND_CREEP_S
            if elapsed <= limit:
                lateral = 0.
                if cluster_now:
                    lateral = BLIND_BOX_LATERAL
                    measured = self.snapshot.get('measured_velocity') or (0., 0.)
                    moved = abs(float(np.asarray(measured, dtype=float)[1]))
                    if elapsed > .7 and moved < .05:
                        # The AMP deadband swallowed the request; step up once.
                        lateral = BLIND_BOX_LATERAL_MAX
                self.desired[:] = [BLIND_BOX_SPEED, self.ride_now*lateral, 0.]
                self.last_drive = self.desired[:2].copy()
                self.reason = 'box_zone_ride' if cluster_now else 'box_zone_creep'
                return self.adapter.update(self.desired, enabled=True)
        elif fresh_lane:
            self.blind_since = None
        # Local RT13 patch: replaying the last translation kept the walk going
        # for the rest of the mission with no lane (4 m blind at .22 m/s).
        # Coast for one perception cycle, then stand until a plan returns.
        lane_age = (timestamp-self.lane_plan_stamp if self.lane_plan_stamp is not None
                    else float('inf'))
        if not self.passage_required and not fresh_lane and lane_age > LANE_COAST_S:
            self.desired[:2] = 0.
        elif np.any(self.desired[:2]):
            self.last_drive = self.desired[:2].copy()
        elif not self.passage_required and np.any(self.last_drive) and lane_age <= LANE_COAST_S:
            self.desired[:2] = self.last_drive
        if not self.crates.exit_search_used and getattr(self,'launch_stamp',None) is None and self.desired[0]>=.12:
            self.launch_stamp=timestamp
            self.launch_direction=-1 if self.desired[2]<0 else 1
        if not self.crates.exit_search_used and self.desired[0]>.10 and timestamp-getattr(self,'launch_stamp',-100.)<.70:
            self.desired[:]=[.22,0.,.24*self.launch_direction]
        return self.adapter.update(self.desired, enabled=True)

    def diagnostic(self):
        state = dict(self.tracked.diagnostic() if self.tracking_control else self.snapshot)
        state.update(ride_side=self.ride_side, ride_y=self.ride_y, ride_cells=self.ride_count,
                     tape_points=self.tape_counts,
                     reason=self.reason, control_source='tracked' if self.tracking_control else 'current_lane',
                     mode=state['mode'] if self.tracking_control else ('turn_to_lane' if self.turning_to_lane else 'lane'), scene=self.scene,
                     mapping_budget_skips=self.mapping_budget_skips, lane_held=self.holding_lane, lane_basis=self.local.lane_basis, tracking_frame_time=self.snapshot_time,
                     desired_velocity=self.desired.tolist(), local_lane_time=self.lane_time,
                     turn_no_response=self.lane_turn.no_response, pending_raised_cells=len(self.unconfirmed_sites),
                     clearance_frames=self.clear_frames, lane_ground_reason=self.local.ground.result.reason,
                     target_policy_velocity=self.adapter.target.tolist())
        state.update(self.crates.diagnostic())
        state.update(self.motion_status)
        return state
