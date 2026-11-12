"""Shared RGB-D navigation on the robot; standalone entry is always shadow.

The bounded supervisor may call run(..., publish=True) only after calibration
and immutable factory-contract checks. Published requests are desired policy
velocities; the sole 50 Hz guard applies navigation.command.AMPCommandAdapter.
"""
import argparse
from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
import platform
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from navigation.calibration import calibration_reasons
from navigation.command import verify_device_files
from navigation.runtime import CameraFrame, LaneTurn, OnboardNavigation, core_hashes
from navigation.release import verify_release
from navigation.environment import verify_dependencies
from deploy.geometry import quaternion_matrix_xyzw


def frame_head_reasons(config, head, timestamp):
    """Only head-pose errors invalidate the fixed camera projection in shadow."""
    fresh = {name: value[1] for name, value in head.items() if abs(value[0]-timestamp) < .2}
    return [reason for reason in calibration_reasons(config, fresh)
            if reason in ('fixed_head_reference_required', 'head_pose_not_at_calibrated_reference')]


def preflight(config):
    verify_dependencies()
    release=verify_release(ROOT)
    identity = verify_device_files('/home/nvidia/action_library/policy/amp',
        '/home/nvidia/sim2real/install/lib/sim2real_master/sim2real_master_node',
        '/home/nvidia/sim2real/src/sim2real/robot_config/pi_plus-S-12L8A0G2H0W/pd.yaml')
    imports = {}
    for name in ('rospy', 'rosgraph', 'rosnode', 'yaml', 'sensor_msgs.msg', 'geometry_msgs.msg', 'std_msgs.msg', 'livelybot_serial.msg', 'sim2real_msg.msg'):
        importlib.import_module(name)
        imports[name] = True
    return dict(status='code_and_factory_contract_verified', python=platform.python_version(),
                release_id=release['release_id'],
                numpy=np.__version__, opencv=cv2.__version__, imports=imports,
                factory_sha256=identity, navigation_core_sha256=core_hashes(),
                calibration_inhibit_reasons=calibration_reasons(config),
                geometry_basis=config.get('calibration_basis', 'calibration_flags'),
                geometry_assumptions=config.get('assumptions', []),
                motor_output_enabled=False)


class TurnCutoff:
    """Latch the first signed AHRS limit between camera updates."""
    def __init__(self):
        self.measurement = None

    def permits(self, pulse, yaw):
        start, origin, direction = pulse
        if self.measurement is not None and self.measurement[0]==start:
            return False
        delta = direction*((yaw-origin+np.pi)%(2*np.pi)-np.pi)
        if abs(delta)>=LaneTurn.angle:
            self.measurement = (start,delta)
            return False
        return True


@dataclass(frozen=True)
class PublishedPlan:
    vector: tuple = (0., 0., 0.)
    deadline: float = 0.
    pulse: object = None
    reason: str = 'initializing'
    scene: object = None
    frame_time: object = None
    fresh: bool = False
    box_servo: object = None

    def evaluate(self, stamp, imu, body_from_imu, cutoff):
        if self.box_servo is not None and stamp >= self.deadline:
            return (0., 0., 0.), 'box_observation_expired', False
        if self.pulse is None and self.box_servo is None:
            return self.vector, self.reason, self.fresh
        if imu is None or not 0. <= stamp-imu[0] <= .15:
            reason = 'no_fresh_turn_imu' if self.pulse is not None else 'no_fresh_box_imu'
            return (0., 0., 0.), reason, False
        attitude = quaternion_matrix_xyzw(imu[1]) @ body_from_imu.T
        if self.pulse is not None and not cutoff.permits(self.pulse, np.arctan2(attitude[1,0], attitude[0,0])):
            return (0., 0., 0.), 'turn_angle_limit', False
        vector = self.vector
        if self.box_servo is not None:
            vector = self.box_servo(stamp, np.array(self.vector), attitude)
            if np.shape(vector) != (3,) or not np.isfinite(vector).all():
                return (0., 0., 0.), 'invalid_box_request', False
        reason=self.reason
        if self.box_servo is not None and not np.any(vector) and any(self.vector):
            reason='box_plan_replaced_or_inhibited'
        return vector, reason, self.fresh


def supervised_request(plan,now,imu,body_from_imu,cutoff,runtime,guard):
    vector,reason,fresh=plan.evaluate(now,imu,body_from_imu,cutoff)
    deadline=plan.deadline if plan.box_servo is not None else None
    if runtime is not None:
        yaw=stamp=None
        if imu is not None and 0<=now-imu[0]<=.15:
            rotation=quaternion_matrix_xyzw(imu[1])@body_from_imu.T
            yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
            stamp=imu[0]
        zero=guard[1] if guard is not None and 0<=now-guard[0]<=.20 else None
        vector,reason,recovery_deadline=runtime.supervise_output(now,vector,reason,stamp,yaw,zero)
        if recovery_deadline is not None:
            deadline=recovery_deadline
    return vector,reason,fresh,deadline


def export_points(out, depth, intrinsics, raised, observation, rotation, ground_z):
    """Write the measured cloud so the operator page can render it in 3D."""
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics
    stride = 3
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[0:height:stride, 0:width:stride]
    good = np.isfinite(z) & (z > .20) & (z < 3.0)
    xs = (us-cx)/fx*z
    ys = (vs-cy)/fy*z
    cam = np.stack([xs[good], ys[good], z[good]], axis=1)
    if rotation is not None:
        body = cam @ np.asarray(rotation, dtype=float).T
    else:
        body = cam
    keep = (np.abs(body[:, 0]) < 3.2) & (np.abs(body[:, 1]) < 1.6) & np.isfinite(body).all(axis=1)
    body = body[keep]
    if len(body) > 2600:
        body = body[::int(np.ceil(len(body)/2600.))]
    payload = dict(
        floor=np.round(body, 3).tolist(),
        raised=[] if raised is None or not len(raised) else
               [[round(float(v[0]), 3), round(float(v[1]), 3), round(float(ground_z), 3)]
                for v in np.asarray(raised)],
        tape=[])
    tape = None
    if observation is not None:
        tape = getattr(observation, 'near_white', None)
        if tape is None or not len(tape):
            tape = getattr(observation, 'current_white', None)
    if tape is not None and len(tape):
        pts = np.asarray(tape, dtype=float)
        if len(pts) > 1500:
            pts = pts[::int(np.ceil(len(pts)/1500.))]
        payload['tape'] = [[round(float(v[0]), 3), round(float(v[1]), 3), round(float(ground_z), 3)]
                           for v in pts]
    try:
        (out/'latest_points.json').write_text(json.dumps(payload, separators=(',', ':')))
    except OSError:
        pass


def _write_perception_previews(out, depth, intrinsics, raised, observation=None,
                               rotation=None, ground_z=0.):
    """Depth colour map and a bird's-eye 3D view of the measured points."""
    import cv2
    import numpy as np
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics
    z = depth
    valid = np.isfinite(z) & (z > .15) & (z < 3.0)
    d8 = np.clip((np.nan_to_num(z, nan=3.0)-.2)/2.8*255., 0, 255).astype(np.uint8)
    cv2.imwrite(str(out/'latest_depth.jpg'), cv2.applyColorMap(255-d8, cv2.COLORMAP_TURBO))
    vs, us = np.mgrid[0:height, 0:width]
    forward = z
    lateral = (us-cx)/fx*z
    side = 1.25
    size = 360
    canvas = np.zeros((size, size, 3), np.uint8)
    sel = valid
    f = forward[sel]
    s_lat = lateral[sel]
    if not len(f):
        cv2.imwrite(str(out/'latest_3d.jpg'), canvas)
        export_points(out, depth, intrinsics, raised, observation, rotation, ground_z)
        return
    u = ((s_lat+side)/(2*side)*(size-1)).astype(int)
    v = (size-1-(f/3.0)*(size-1)).astype(int)
    keep = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    colour = cv2.applyColorMap(np.clip(f/3.0*255., 0, 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO).reshape(-1, 3)
    canvas[v[keep], u[keep]] = colour[keep]
    def to_pixels(xs, ys):
        uu = ((ys+side)/(2*side)*(size-1)).astype(int)
        vv = (size-1-(xs/3.0)*(size-1)).astype(int)
        return uu, vv
    tapes = None
    if observation is not None:
        tapes = getattr(observation, 'near_white', None)
        if tapes is None or not len(tapes):
            tapes = getattr(observation, 'current_white', None)
    if tapes is not None and len(tapes):
        tape_points = np.asarray(tapes, dtype=float)
        uu, vv = to_pixels(tape_points[:, 0], tape_points[:, 1])
        ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
        canvas[vv[ok], uu[ok]] = (255, 255, 0)
    if raised is not None and len(raised):
        pts = np.asarray(raised, dtype=float)
        uu, vv = to_pixels(pts[:, 0], pts[:, 1])
        ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
        canvas[vv[ok], uu[ok]] = (0, 140, 255)
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        if x1-x0 < .20:
            mid = .5*(x0+x1); x0, x1 = mid-.10, mid+.10
        if y1-y0 < .20:
            mid = .5*(y0+y1); y0, y1 = mid-.10, mid+.10
        a, b = to_pixels(np.array([x1, x1, x0, x0]), np.array([y0, y1, y1, y0]))
        cv2.polylines(canvas, [np.column_stack([a, b]).astype(np.int32)], True,
                      (0, 200, 255), 2)
        label = '%.2fx%.2fm @%.2f' % (x1-x0, y1-y0, .5*(x0+x1))
        cv2.putText(canvas, label, (8, size-10), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (0, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (size//2, size-1), 6, (255, 255, 255), -1)
    for value, colour in ((0., (90, 90, 90)), (-.25, (200, 120, 255)), (.25, (200, 120, 255))):
        pass
    cv2.line(canvas, (size//2, size-1), (size//2, 0), (80, 80, 80), 1)
    cv2.imwrite(str(out/'latest_3d.jpg'), canvas)
    export_points(out, depth, intrinsics, raised, observation, rotation, ground_z)


_FAULT_HANDLES = []


def _enable_fault_log(out):
    """Write a hard-crash traceback (segfault, aborted thread) to disk.

    The released supervisor only reports 'exited_without_report' when the
    vision process dies without writing its summary, which hides the cause.
    """
    import faulthandler
    try:
        out.mkdir(parents=True, exist_ok=True)
        handle = (out/'crash.log').open('a')
        handle.write('\n=== vision start %s ===\n' % time.strftime('%Y-%m-%dT%H:%M:%S'))
        handle.flush()
        faulthandler.enable(file=handle, all_threads=True)
        _FAULT_HANDLES.append(handle)
    except Exception as exc:
        print('fault log unavailable: %s' % exc, flush=True)


def run(args, publish=False):
    import rospy
    from geometry_msgs.msg import TwistStamped
    from deploy.shadow import Sensors, capture_worker, read_exact
    config = json.loads(args.config.read_text())
    initial_reasons = calibration_reasons(config)
    if publish and initial_reasons:
        raise ValueError('Motion inhibited: '+', '.join(initial_reasons))
    identity = preflight(config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _enable_fault_log(out)
    (out/'preflight.json').write_text(json.dumps(identity, indent=2))
    (out/'config.json').write_text(json.dumps(config, indent=2))
    rospy.init_node('competition_lane_trial' if publish else 'competition_nav_shadow', disable_signals=True)
    sensors = Sensors()
    publisher = rospy.Publisher('/competition/lane_request', TwistStamped, queue_size=1) if publish else None
    proc = None
    rows = []
    commands = []
    runtime = None
    status = 'starting'
    start = time.monotonic()

    archive = None
    guard_sub = None
    guard_state = None
    terminate = threading.Event()
    if publish:
        from deploy.stop_archive import StopArchive
        from std_msgs.msg import String
        archive = StopArchive(out)
        def guard_feedback(msg):
            nonlocal guard_state
            row=json.loads(msg.data)
            if row.get('command') is not None:
                guard_state=(time.monotonic(),row.get('stopped_at') if not any(row['command']) else None)
            archive.guard(row)
        guard_sub = rospy.Subscriber('/competition/line_guard', String,guard_feedback,queue_size=128)
        def stop_signal(signum, frame):
            terminate.set()
        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGHUP, stop_signal)
    def request(vector, reason='initializing', scene=None, frame_time=None, new_frame=False, box_deadline=None):
        if publisher is not None:
            msg = TwistStamped()
            msg.header.frame_id = args.token
            msg.header.stamp = rospy.Time.now()
            # Recheck after servo calculation, including the tick straddling a lease.
            if terminate.is_set():
                vector, reason, new_frame = (0., 0., 0.), 'termination_requested', False
            elif box_deadline is not None and time.monotonic() >= box_deadline:
                vector, reason, new_frame = (0., 0., 0.), 'box_observation_expired', False
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = map(float, vector)
            if archive is not None:
                archive.decision(msg.header.stamp.to_nsec(), reason, scene, frame_time, new_frame)
            publisher.publish(msg)

    # Perception produces one update per frame (about 4 Hz on this robot), but
    # the 50 Hz guard drops any request older than 0.2 s. Publishing straight
    # from the frame loop therefore left the guard with nothing to forward. The
    # publisher thread repeats the newest planned velocity and applies the
    # observation lease itself, so a slow frame never looks like a dead link.
    # Replace the vector and its capture-based deadline together.
    publish_state = PublishedPlan()
    turn_cutoff = TurnCutoff()
    body_from_imu = np.asarray(config['body_from_imu'])

    def inhibit(reason=None):
        nonlocal publish_state
        # RT10: an inhibit only relabels the request; it never zeroes it.
        publish_state = replace(publish_state, reason=reason or publish_state.scene or 'perception_inhibit')
    publish_stop = threading.Event()

    def publish_loop():
        next_tick = time.monotonic()
        while not publish_stop.is_set():
            plan = publish_state
            imu = sensors.latest_imu
            now = time.monotonic()
            if terminate.is_set():
                vector,reason,fresh,deadline=(0.,0.,0.),'termination_requested',False,None
            else:
                vector,reason,fresh,deadline=supervised_request(plan,now,imu,body_from_imu,turn_cutoff,runtime,guard_state)
                if plan is not publish_state:
                    continue
            request(vector,reason,plan.scene,plan.frame_time,fresh,deadline)
            next_tick += .02
            now = time.monotonic()
            if next_tick <= now:
                next_tick = now + .02
            publish_stop.wait(next_tick-now)

    publisher_thread = None
    if publish:
        publisher_thread = threading.Thread(target=publish_loop, daemon=True)
        publisher_thread.start()

    try:
        with (out/'camera.log').open('w') as log:
            proc = subprocess.Popen([str(ROOT/'deploy/realsense_stream')], stdout=subprocess.PIPE, stderr=log, bufsize=0)
            header = bytearray()
            while not header.endswith(b'\n'):
                header.extend(read_exact(proc.stdout, 1, 10.))
                if len(header) > 4096:
                    raise ValueError('Invalid camera header')
            camera = json.loads(header)
            (out/'camera_intrinsics.json').write_text(json.dumps(camera, indent=2))
            w, h = camera['width'], camera['height']
            frame_period = 1./float(camera['fps'])
            k = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1.]])
            coefficients = np.asarray(camera['coefficients'])
            if np.any(abs(coefficients) > 1e-9) and camera['distortion_model'] != 4:
                raise ValueError('Unsupported nonzero camera distortion')
            # Perception runs on a downscaled image. The pixel-centre convention is
            # preserved so every metric quantity (rays, ground, lanes) is unchanged.
            scale = float(config.get('perception_scale', .5))
            if not .25 <= scale <= .5:
                raise ValueError('perception_scale must be within [0.25, 0.5]')
            sw, sh = int(round(w*scale)), int(round(h*scale))
            intrinsics = [k[0, 0]*scale, k[1, 1]*scale, (k[0, 2]+.5)*scale-.5, (k[1, 2]+.5)*scale-.5]
            # Rectify straight into the perception resolution. Remapping the full
            # 640x480 pair and resizing afterwards spent about 0.25 s per frame on
            # the Jetson for pixels that were then discarded; the decimated image
            # is all that is ever used, so the maps target it directly.
            destination = np.array([[intrinsics[0], 0, intrinsics[2]],
                                    [0, intrinsics[1], intrinsics[3]], [0, 0, 1.]])
            maps = cv2.initUndistortRectifyMap(k, coefficients, None, destination, (sw, sh), cv2.CV_32FC1)
            runtime = OnboardNavigation(sw, sh, intrinsics, config['camera_position_body_m'],
                config['camera_rotation_optical_to_body'], args.speed,
                float(config.get('observation_lease_s', .35)))
            if hasattr(runtime, 'publish_velocity') and not hasattr(runtime, 'freeze_publish_velocity'):
                raise ValueError('A 50 Hz box controller must implement freeze_publish_velocity()')
            if hasattr(runtime, 'update_legs'):
                def update_legs(stamp, joints, quat):
                    return runtime.update_legs(stamp, joints, quaternion_matrix_xyzw(quat) @ body_from_imu.T)
                sensors.leg_callback = update_legs
            latest = queue.Queue(maxsize=1)
            # Frame arrival is recorded by the reader, not inferred from how long
            # our own processing took, so a slow frame cannot fake a dead camera.
            camera_status = {'frames': 0, 'last_received': time.monotonic(),
                             'last_frame_time': None, 'error': None}
            threading.Thread(target=capture_worker, args=(proc.stdout, w, h, latest, camera_status), daemon=True).start()
            last_report = -1.
            control_start = time.monotonic()
            while time.monotonic()-control_start < args.duration and not rospy.is_shutdown() and not terminate.is_set():
                tick = time.monotonic()
                current_reasons = list(initial_reasons)
                raw = None
                try:
                    raw = latest.get_nowait()
                except queue.Empty:
                    pass
                if isinstance(raw, Exception):
                    # RT10: a camera read error is telemetry; keep walking.
                    raw = None
                capture_imu, capture_head = None, {}
                imu = None
                if raw is not None:
                    try:
                        capture_imu, _, capture_head = sensors.snapshot(raw.time)
                    except ValueError:
                        pass
                    if archive is not None:
                        archive.frame(raw, capture_imu, capture_head)
                try:
                    imu, joints, head = sensors.snapshot(tick)
                    rotation = quaternion_matrix_xyzw(imu[1])@body_from_imu.T
                    runtime.set_attitude(imu[0], rotation)
                    fresh_head = {name: value[1] for name, value in head.items() if tick-value[0] < .2}
                    current_reasons += calibration_reasons(config, fresh_head)
                    if sum(tick-value[0] < .2 for value in joints.values()) != 20:
                        current_reasons.append('joint_feedback_incomplete')
                except ValueError:
                    current_reasons.append('no_fresh_imu')
                processing_ms = None
                if raw is not None:
                    if tick-raw.time > .2:
                        projection_reasons = ['stale_capture']
                    else:
                        if capture_imu is not None:
                            imu = capture_imu
                            projection_reasons = frame_head_reasons(config, capture_head, raw.time)
                        else:
                            # RT10: a missing capture attitude relabels only.
                            projection_reasons = []
                        if imu is None:
                            projection_reasons = ['no_attitude']
                    current_reasons += projection_reasons
                    if current_reasons:
                        inhibit(','.join(current_reasons))
                    if not projection_reasons:
                        try:
                            before = time.monotonic()
                            frame_rotation = quaternion_matrix_xyzw(imu[1])@body_from_imu.T
                            rgb = cv2.remap(raw.rgb, *maps, cv2.INTER_LINEAR)
                            depth = cv2.remap(raw.depth, *maps, cv2.INTER_NEAREST)
                            remap_ms = (time.monotonic()-before)*1000
                            depth[(depth < .15) | (depth > 4.)] = np.nan
                            observe_start = time.monotonic()
                            observation = runtime.observe(CameraFrame(raw.time, rgb, depth), frame_rotation, imu[0], inhibit, defer_mapping=True)
                            processing_ms = (time.monotonic()-observe_start)*1000
                        except Exception as exc:
                            # RT10: a rejected frame never ends the walk.
                            current_reasons.append('observe_failed:'+str(exc)[:80])
                            processing_ms = None
                now = time.monotonic()
                # RT10: a stalled camera no longer ends the walk; the last
                # request is republished until the 300 s mission limit.
                # Re-read the control attitude right before planning so the IMU
                # freshness gate measures the sensor link, not how long this loop
                # just spent on perception.
                try:
                    control_imu, _, _ = sensors.snapshot(now)
                    runtime.set_attitude(control_imu[0],
                        quaternion_matrix_xyzw(control_imu[1])@body_from_imu.T)
                except ValueError:
                    pass
                # Shadow calculates candidates with declared transforms, while
                # retaining every calibration inhibit in the diagnostic record.
                enabled = True
                candidate = runtime.command(now, enabled=enabled, measured_turn_limit=turn_cutoff.measurement)
                desired = runtime.desired.copy()
                computation_ms = (time.monotonic()-tick)*1000
                pulse = None
                if runtime.lane_turn.started is not None and runtime.reason=='ready':
                    turn = runtime.lane_turn
                    pulse = (turn.started, runtime.initial_yaw+turn.origin, turn.direction)
                reason = ','.join(current_reasons) if current_reasons else runtime.reason
                frame_time = raw.time if raw is not None else runtime.last_frame_time
                # The published hook owns this *same* command's plan parameters.
                # Its only live inputs are the current AHRS and completed encoder samples.
                freeze = getattr(runtime, 'freeze_publish_velocity', None)
                publish_state = PublishedPlan(tuple(desired), runtime.deadline, pulse,
                    reason, runtime.scene, frame_time, raw is not None, freeze() if freeze else None)
                if not publish:
                    request(desired)
                if processing_ms is not None:
                    runtime.dispatch_mapping(tick, time.monotonic(), frame_period)
                    row = dict(t=tick-start, camera_host_age_s=tick-raw.time, imu_host_delta_s=imu[0]-raw.time,
                        remap_ms=remap_ms, processing_ms=processing_ms, lane=None if observation.lane is None else observation.lane.tolist(),
                        boxes=observation.obstacles.tolist(), **runtime.diagnostic())
                    rows.append(row)
                    with (out/'telemetry.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
                    if tick-last_report > 1:
                        if archive is not None:
                            archive.preview(raw.time, rgb, observation.overlay)
                        else:
                            cv2.imwrite(str(out/'latest_rgb.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(out/'latest_overlay.jpg'), cv2.cvtColor(observation.overlay, cv2.COLOR_RGB2BGR))
                        _write_perception_previews(
                            out, depth, intrinsics,
                            getattr(runtime.local, 'raised_sites', None), observation,
                            config.get('camera_rotation_optical_to_body'),
                            float(observation.ground_height))
                        print(json.dumps(row), flush=True)
                        last_report = tick
                commands.append(dict(t=now-start, candidate_raw=candidate.tolist(),
                    requested_policy_velocity=desired.tolist(),
                    published=publish, computation_ms=computation_ms, has_camera_frame=processing_ms is not None,
                    inhibit_reasons=sorted(set(current_reasons)), **runtime.diagnostic()))
                # RT10: the finish line is recorded (runtime.finished) but does
                # not end the mission; the 300 s limit does.
                time.sleep(max(0., .02-(time.monotonic()-tick)))
            else:
                status = 'termination_requested' if terminate.is_set() else ('navigation_time_limit' if publish else 'shadow_complete')
    except (Exception, KeyboardInterrupt) as exc:
        status = 'error: '+str(exc)
    finally:
        sensors.leg_callback = None
        publish_stop.set()
        if publisher_thread is not None:
            publisher_thread.join(timeout=1.)
        for _ in range(5):
            request([0., 0., 0.], status, runtime.scene if runtime else None)
            if publish:time.sleep(.02)
        if proc is not None:
            proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if archive is not None:
            guard_sub.unregister()
            try:archive.close()
            except Exception as exc:print('Stop archive close failed: '+str(exc),flush=True)
        rospy.signal_shutdown('shared navigation stopped')
        if runtime is not None:
            runtime.close(out/'floor_map.npz')
        (out/'commands.json').write_text(json.dumps(commands))
        summary = dict(status=status, frames=len(rows), motor_publishers=0, perception_scale=config.get('perception_scale', .5), velocity_request_publishers=int(publish),
            navigation_core_sha256=core_hashes(), calibration_inhibit_reasons=initial_reasons,
            last_navigation=None if runtime is None else runtime.diagnostic(),
            remap_ms_p50=float(np.median([r['remap_ms'] for r in rows])) if rows else None,
            remap_ms_p95=float(np.percentile([r['remap_ms'] for r in rows], 95)) if rows else None,
            processing_ms_p50=float(np.median([r['processing_ms'] for r in rows])) if rows else None,
            processing_ms_p95=float(np.percentile([r['processing_ms'] for r in rows], 95)) if rows else None,
            camera_cycle_ms_p95=float(np.percentile([r['computation_ms'] for r in commands if r['has_camera_frame']],95)) if any(r['has_camera_frame'] for r in commands) else None,
            control_cycle_ms_p95=float(np.percentile([r['computation_ms'] for r in commands],95)) if commands else None,
            imu_camera_association='host receipt timestamps; exposure latency not calibrated')
        (out/('vision_summary.json' if publish else 'summary.json')).write_text(json.dumps(summary, indent=2))
        try:
            summary['rss_mb'] = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])/1024.
        except Exception:
            summary['rss_mb'] = None
        print(json.dumps(summary, indent=2), flush=True)
    if publish:
        return 0 if status=='visual_finish_detected' else 2
    return 2 if status.startswith('error:') else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/pro_config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=.22)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not 0 < args.duration <= 300 or not 0 < args.speed <= .4:
        parser.error('Duration/speed outside the bounded navigation contract')
    if args.preflight:
        result = preflight(json.loads(args.config.read_text()))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'preflight.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    return run(args, publish=False)


if __name__ == '__main__':
    raise SystemExit(main())
