"""Minimal headed MuJoCo driver for an RT16-lineage release package.

The RT11/RT13 harness (scripts/rt11_sim/run_validation.py) replays the recorded
RT10 publish schedule and needs published plans, the crate servo and a tracker.
The RT16 package on the robot dropped those interfaces, so this runner keeps the
same MuJoCo world, robot and sensors but drives the package directly: one RGB-D
capture every ``--cam-period`` seconds, one command per 50 Hz control tick.

    .venv-shared/bin/mjpython scripts/rt16_sim/run_headed.py \
        --source output/deployment/rt16_pull/shared_nav_20261112_rt16 \
        --case full_01 --gui --limit 300
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--case', default='full_01')
parser.add_argument('--out', type=Path)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--limit', type=float, default=300.)
parser.add_argument('--cam-period', type=float, default=.2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
SRC = args.source.resolve()
sys.path[:0] = [str(SRC), str(ROOT), str(ROOT/'scripts/rt11_sim')]

import cv2
import mujoco
import mujoco.viewer
from PIL import Image

from deploy.geometry import quaternion_matrix_xyzw
from navigation import params
import navigation.runtime as _nav_runtime
from navigation.command import AMPCommandAdapter
from navigation.runtime import CameraFrame, OnboardNavigation, core_hashes
from rgbd_view import FrameWriter, millimeters, sensor_panel, show_panel
from sim.navigation import Track
from sim.robot import Robot
from sim.scoring import Referee
from sim.sensors import SensorRig
# run_visual imports the RT13-only NavigationRuntime at module level; stub it so
# only its world builder is reused.
if not hasattr(_nav_runtime, 'NavigationRuntime'):
    _nav_runtime.NavigationRuntime = object
from scripts.run_visual import visual_world

cv2.setNumThreads(1)
LOAD = lambda p: json.loads(p.read_text())
BASE = ROOT/'reference/rt11_sim'
config = LOAD(BASE/'vision/config.json')
intrinsics_path = BASE/'vision/camera_intrinsics.json'
camera = dict(LOAD(intrinsics_path), processing_scale=config['perception_scale'], rectify=True)

cases = {'full_01': [.55, -1.25, 0.], 'boxes_center': [2.80, 1.25, np.pi],
         'boxes_oblique': [2.62, 1.40, np.pi-.20], 'small_entry': [1., 1.25, np.pi],
         'small_entry_offset': [1., 1.25, np.pi+.14], 'full_01_offset': [.55, -1.25, .14]}

stamp = time.strftime('%Y%m%d_%H%M%S')
out = (args.out or ROOT/f'output/sim/rt16_run_{stamp}').resolve()
out.mkdir(parents=True, exist_ok=False)

track = Track('corner')
spawn = cases[args.case]
start = track.config['starts'][0]
empty = args.case.startswith(('small', 'large_exit')) or args.case == 'sharp_entry'
model, xml = visual_world(out, start, no_obstacles=empty, camera_profile=camera, bend_profile='corner')
(out/'scene.xml').write_text(xml)
robot = Robot(model, spawn, from_standby=True)
rig = SensorRig(model, seed=args.seed, noise=True, camera_profile=camera)
referee = Referee(robot, track)
params.load(SRC/'deploy/tuning.json')
nav = OnboardNavigation(rig.width, rig.height, rig.intrinsics,
                        config['camera_position_body_m'], config['camera_rotation_optical_to_body'],
                        config.get('speed_m_s', .22), config['observation_lease_s'])
(out/'setup.json').write_text(json.dumps(dict(
    candidate=str(SRC), case=args.case, camera=camera, core_sha256=core_hashes(),
    spawn=spawn, track_length_m=track.length, tuning=params.as_dict(),
    camera_period_s=args.cam_period, seed=args.seed), indent=2))

view = None
if args.gui:
    view = mujoco.viewer.launch_passive(model, robot.data, show_left_ui=False, show_right_ui=False)
    with view.lock():
        view.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        view.cam.trackbodyid = robot.base
        view.cam.lookat[:] = robot.data.xpos[robot.base]
        view.cam.distance, view.cam.elevation, view.cam.azimuth = 2.5, -32., 85.
    view.sync()
    print('MuJoCo GUI open', flush=True)

recorder = FrameWriter(out)
panel = panel_key = None
latest = None
decisions = []
states = []
feedback_history = []
guard = AMPCommandAdapter()
START_S = 2.5
FEEDBACK_DELAY = .15


def feedback_at(when):
    """Latest measured policy command at or before ``when`` (computation delay)."""
    for stamp, value in reversed(feedback_history):
        if stamp <= when:
            return value
    return None


next_capture = 0.
t0 = time.monotonic()
frames = 0
try:
    while robot.data.time < args.limit + 1e-9:
        t = float(robot.data.time)
        measurement = rig.proprioception(robot.data)
        q = measurement.orientation_wxyz
        attitude = quaternion_matrix_xyzw(np.r_[q[1:], q[0]]) @ np.array(config['body_from_imu']).T
        nav.set_attitude(t, attitude)
        if t + 1e-9 >= next_capture:
            latest = rig.camera(robot.data)
            nav.observe(CameraFrame(t, latest.rgb, latest.depth), attitude, t, lambda *a: None)
            recorder.capture(t, t, latest.rgb, latest.depth)
            next_capture += args.cam_period
            frames += 1
        # Same factory-input chain as the RT11 validation harness: a separate
        # adapter is fed the measured policy command from FEEDBACK_DELAY ago.
        # Without that delayed feedback its feed-forward decays to the target,
        # the factory deadband drops the request and the body stays standing.
        nav.command(t)
        feedback = feedback_at(t - FEEDBACK_DELAY)
        cmd = guard.update(nav.desired, feedback, enabled=t >= START_S)
        decisions.append((t, nav.reason, *[round(float(v), 4) for v in cmd]))
        recorder.plan(t, nav.reason)
        robot.step(cmd, referee.contacts)
        feedback_history.append((t, robot.device.command.copy()))
        referee.boundary()
        states.append(robot.data.qpos.copy())
        if view is not None and view.is_running():
            if len(states) % 4 == 1:
                x, y, yaw = robot.pose
                view.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, 'rt16 | ' + args.case,
                                f'{t:.1f} s    {x:.2f},{y:.2f}    {nav.reason}'))
                key = (t, nav.reason)
                if panel_key != key and latest is not None:
                    panel = sensor_panel(latest.rgb, millimeters(latest.depth), t, nav.reason)
                    panel_key = key
                if panel is not None:
                    show_panel(view, panel)
                view.sync()
            if not view.is_running():
                break
        if int(t*2) % 2 == 0 and len(states) % 50 == 1:
            x, y, yaw = robot.pose
            print(f'{args.case} t={t:5.1f} xy=({x:5.2f},{y:5.2f}) yaw={yaw:5.2f} {nav.reason} cmd={np.round(cmd,2)}', flush=True)
        if referee.reason:
            print('referee:', referee.reason, referee.event_detail, 'at', round(t, 2), flush=True)
            break
except KeyboardInterrupt:
    pass
finally:
    if view is not None:
        try:
            view.close()
        except Exception:
            pass

summary = dict(case=args.case, candidate=str(SRC), seconds=round(float(robot.data.time), 2),
               frames=frames, referee=referee.reason, referee_detail=referee.event_detail,
               min_base_height=referee.min_base_height, gui=bool(args.gui),
               wall_s=round(time.monotonic()-t0, 1),
               end_pose=[round(float(v), 3) for v in robot.pose],
               desired=[round(float(v), 3) for v in nav.desired],
               reason=nav.reason)
(out/'summary.json').write_text(json.dumps(summary, indent=2))
with (out/'decisions.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'reason', 'vx', 'vy', 'wz'])
    w.writerows(decisions)
np.savez_compressed(out/'states.npz', t=np.arange(len(states))*.02, qpos=np.array(states))
Image.fromarray(latest.rgb).save(out/'last_camera.jpg') if latest is not None and hasattr(latest, 'rgb') else None
print(json.dumps(summary, indent=2), flush=True)
