"""Identical sensor-to-command pipeline for simulation, replay and ROS.

Positions are estimated from RGB-D and leg encoders between captures. Attitude is world_from_body from an
AHRS measurement. No world position or simulator object data is accepted.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from navigation.box_servo import BoxServo
from navigation.command import AMPCommandAdapter
from navigation.environment import verify_dependencies
from navigation.floor_map import FloorMap
from navigation.motion_recovery import MotionRecovery
from navigation.visual import CameraPerception, Perception, RGBDOdometry, VisualNavigator, rotation2


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
        self.measured_lane = None
        self.search_origin = None
        self.search_side = 1.
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
        # Most recent lateral/forward request used by the inherited lane controller.
        self.last_drive = np.zeros(2)
        self.reason = 'initializing'

    def publish_velocity(self,t,planned,attitude):
        hook=self.freeze_publish_velocity()
        return planned if hook is None else hook(t,planned,attitude)

    def invalidate_box_projection(self, stamp):
        self.crates.invalidate_projection(stamp)

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
        self.crates.observe(frame,attitude,self.local,self.tracked,self.initial_yaw,observation.lane,image_ok)
        if self.crates.done:
            self.tracking_control=False
            self.box_latched=self.saw_box=False
            self.unconfirmed_sites.clear()
        if (image_ok and observation.lane_observed and observation.steering_heading is not None and
                observation.lane_support is not None and self.scene in ('clear','turn_clear','unknown_depth') and
                (self.scene!='unknown_depth' or self.local.lane_basis in ('paired_edges','edge_with_history'))):
            near=max(.15,observation.lane_support[0])
            slope=np.polyval(np.polyder(observation.lane),near)
            offset=(np.polyval(observation.lane,near)-near*slope)/np.hypot(1.,slope)
            self.measured_lane=(frame.time,capture_yaw+observation.steering_heading,float(offset))
            self.search_origin=None
        if self.tracking_control:
            self.lane_time = None
            return self.tracked.observation
        if self.scene in ('clear','turn_clear') and observation.lane_observed:
            self.steering_heading = observation.steering_heading
            self.lane, self.support = observation.lane.copy(), observation.lane_support
            self.reference_time = self.lane_time = frame.time
            self.lane_yaw = capture_yaw
        elif (self.scene=='clear' and self.lane_time is not None and reference is not None
              and not self.turning_to_lane and not self.missed_previous):
            self.holding_lane = True
        else:
            self.lane_time = None
            inhibit()
        self.missed_previous = not observation.lane_observed
        return observation

    def _recover_lane(self,t):
        # The tracker can keep an unconfirmed raised pixel for many seconds.
        # Use this camera's lane independently of that pending map review.
        lane=self.measured_lane
        self.lane_turn.pause(t)
        self.turning_to_lane=False
        if lane is not None and 0<=t-lane[0]<=.55 and self.scene in ('clear','turn_clear','unknown_depth'):
            error=(lane[1]-self.yaw+np.pi)%(2*np.pi)-np.pi
            turn_only=self.scene=='turn_clear' or abs(error)>np.deg2rad(20)
            self.desired[:]=[0. if turn_only else min(self.speed,.14),
                             0. if turn_only else np.clip(1.6*lane[2],-.18,.18),
                             np.clip(2.5*error,-.8,.8)]
            self.reason='follow_recovered_lane'
        else:
            if self.search_origin is None:
                self.search_origin=self.yaw if lane is None else lane[1]
                self.search_side=-1. if self.desired[2]<0 else 1.
            target=self.search_origin+self.search_side*np.deg2rad(23)
            error=(target-self.yaw+np.pi)%(2*np.pi)-np.pi
            if abs(error)<np.deg2rad(4):
                self.search_side*=-1.
                error=self.search_side*np.deg2rad(46)
            self.desired[:]=[0.,0.,np.copysign(.70,error)]
            self.reason='turn_to_recover_lane'
        self.last_drive=self.desired[:2].copy()
        return self.adapter.update(self.desired,enabled=True)

    def command(self, timestamp, enabled=True, measured_turn_limit=None):
        previous = self.last_control_time
        self.last_control_time = timestamp
        dt = .02 if previous is None else np.clip(timestamp-previous, 0., .10)
        self.deadline = self.last_frame_time+self.lease if self.last_frame_time is not None else timestamp
        fresh_lane = self.lane_time is not None and timestamp<=self.lane_time+self.lease
        # A pending obstacle classification may still use a current lane observation.
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
        self._box_publication=None
        if guided is not None:
            frame=self.crates.frame
            self.deadline=min(self.deadline,frame.stamp+self.lease)
            self._box_publication=self.crates.publication(frame,self.deadline)
            self.desired=guided
            self.lane_turn.pause(timestamp)
            self.reason='box_'+self.crates.phase
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
        if self.tracking_control:
            self.turning_to_lane = False
            r = self.tracked
            known = any(-.46<b[0]<1.35 and abs(b[1])<.85 for b in r.observation.obstacles)
            if (not known) and r.navigator.gap.phase=='idle':
                if (enabled and self.unconfirmed_sites and not self.box_latched and not self.finish_latched and
                        self.reason not in ('stale_imu','stale_camera','unusable_rgb')):
                    return self._recover_lane(timestamp)
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
        elif (enabled and not fresh_lane and self.measured_lane is not None and
                timestamp-self.measured_lane[0]>self.lease and
                self.reason in ('lost_vision','unknown_depth')):
            return self._recover_lane(timestamp)
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
        # With no fresh lane plan the held yaw is cleared, so an interrupted
        # turn cannot become an unbounded spin.
        if not fresh_lane and not self.tracking_control:
            self.desired[2] = 0.
        # Reuse translation only on the ordinary RT10 lane/tracker path.
        if np.any(self.desired[:2]):
            self.last_drive = self.desired[:2].copy()
        elif np.any(self.last_drive):
            self.desired[:2] = self.last_drive
        if getattr(self,'launch_stamp',None) is None and self.desired[0]>=.12:
            self.launch_stamp=timestamp
            self.launch_direction=-1 if self.desired[2]<0 else 1
        if self.desired[0]>.10 and timestamp-getattr(self,'launch_stamp',-100.)<.70:
            self.desired[:]=[.22,0.,.24*self.launch_direction]
        return self.adapter.update(self.desired, enabled=True)

    def diagnostic(self):
        state = dict(self.tracked.diagnostic() if self.tracking_control else self.snapshot)
        state.update(reason=self.reason, control_source='tracked' if self.tracking_control else 'current_lane',
                     mode=state['mode'] if self.tracking_control else ('turn_to_lane' if self.turning_to_lane else 'lane'), scene=self.scene,
                     mapping_budget_skips=self.mapping_budget_skips, lane_held=self.holding_lane, lane_basis=self.local.lane_basis, tracking_frame_time=self.snapshot_time,
                     desired_velocity=self.desired.tolist(), local_lane_time=self.lane_time,
                     turn_no_response=self.lane_turn.no_response, pending_raised_cells=len(self.unconfirmed_sites),
                     clearance_frames=self.clear_frames, lane_ground_reason=self.local.ground.result.reason,
                     target_policy_velocity=self.adapter.target.tolist())
        state.update(self.crates.diagnostic())
        return state
