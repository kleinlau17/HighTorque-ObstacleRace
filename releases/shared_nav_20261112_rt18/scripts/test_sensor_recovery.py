"""Sensor-fault injection through the shared RGB-D navigation entry point.

The analytic camera moves before each fault, normally holds during interruption,
and resumes its measured trajectory after recovery. One case adds measured
coasting displacement during the outage. This tests navigation
commands and state retention, without simulating motor or stopping dynamics.
"""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import sys
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from navigation.runtime import CameraFrame, NavigationRuntime, core_hashes
from scripts.test_ground_color_integration import (
    CAMERA_POSITION, CAMERA_ROTATION, HEIGHT, INTRINSICS, WIDTH,
    camera_perception, perspective_scene,
)


@dataclass(frozen=True)
class FaultCase:
    name: str
    fault: str
    duration: float
    hidden_box_after: bool = False
    coast_m: float = 0.
    box_angle: float = 0.
    texture_scale: float = 1.
    expect_recovery: bool = True


CASES = [
    FaultCase('rgb_black_0_4s', 'rgb', .4),
    FaultCase('rgb_black_0_6s_two_box_faces', 'rgb', .6, box_angle=np.pi / 4),
    FaultCase('rgb_saturated_0_4s_two_box_faces', 'saturated', .4, box_angle=np.pi / 4),
    FaultCase('depth_invalid_0_4s', 'depth', .4),
    FaultCase('rgb_depth_invalid_0_4s', 'both', .4),
    FaultCase('depth_holes_10_percent_0_6s', 'holes', .6),
    FaultCase('camera_pause_0_6s', 'camera', .6),
    FaultCase('camera_pause_2s', 'camera', 2.),
    FaultCase('imu_pause_0_6s', 'imu', .6),
    FaultCase('camera_pause_0_6s_box_still_occluded', 'camera', .6, True),
    FaultCase('camera_pause_0_6s_5cm_coast_box_occluded', 'camera', .6, True, .05),
    FaultCase('camera_pause_0_6s_10cm_coast_box_occluded', 'camera', .6, True, .10),
    FaultCase('camera_pause_0_6s_20cm_coast_box_occluded', 'camera', .6, True, .20,
              expect_recovery=False),
    FaultCase('camera_pause_0_6s_textureless_box_occluded', 'camera', .6, True,
              texture_scale=0., expect_recovery=False),
]


class SensorScene:
    def __init__(self, box_angle=0., texture_scale=1.):
        self.box_angle = box_angle
        self.rays = camera_perception().rays
        rng = np.random.default_rng(97)
        texture = rng.normal(0., 1., (1024, 1024)).astype(np.float32)
        self.texture = cv2.GaussianBlur(texture, (0, 0), .6) * 1.5 * texture_scale
        self.holes = np.random.default_rng(18).random((HEIGHT, WIDTH)) < .1

    def frame(self, time, position, hidden_box=False):
        frame, labels = perspective_scene([[220, 45, 45]],
            box_centers=np.array([[1.3 - position, .12]]), box_angle=self.box_angle)
        body = (self.rays * frame.depth[..., None]) @ CAMERA_ROTATION.T + CAMERA_POSITION
        grain = cv2.remap(self.texture,
            ((body[..., 0] + position + .1) / .002).astype(np.float32),
            ((body[..., 1] + 1.) / .002).astype(np.float32), cv2.INTER_LINEAR)
        unmarked_floor = (labels < 0) & (frame.rgb.min(axis=2) < 115)
        frame.rgb[unmarked_floor] = np.clip(
            frame.rgb[unmarked_floor].astype(float) + np.rint(grain[unmarked_floor, None]),
            0, 255).astype(np.uint8)
        if hidden_box:
            frame.depth[labels >= 0] = np.nan
            frame.rgb[labels >= 0] = 0
        return CameraFrame(time, frame.rgb, frame.depth)


def run_case(case):
    scene = SensorScene(case.box_angle, case.texture_scale)
    runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION)
    start, end = 1., 1. + case.duration
    rows, exceptions = [], []
    baseline_boxes = None
    # Fixture position is used to render/evaluate sensors, never by the core.
    for index in range(int(round((end + 1.4) / .02)) + 1):
        timestamp = round(index * .02, 6)
        faulty = start <= timestamp < end
        resumed_motion = .12 * max(0., timestamp - end - .3) if case.expect_recovery else 0.
        position = (.12 * min(timestamp, start) + resumed_motion +
                    case.coast_m * np.clip((timestamp - start) / .3, 0., 1.))
        if not (faulty and case.fault == 'imu'):
            runtime.set_attitude(timestamp, np.eye(3))
        observed = index % 5 == 0 and not (faulty and case.fault == 'camera')
        if observed:
            frame = scene.frame(timestamp, position,
                hidden_box=case.hidden_box_after and timestamp >= end)
            rgb, depth = frame.rgb, frame.depth
            if faulty:
                if case.fault in ('rgb', 'both'):
                    rgb = np.zeros_like(rgb)
                if case.fault == 'saturated':
                    rgb = np.full_like(rgb,255)
                if case.fault in ('depth', 'both'):
                    depth = np.full_like(depth, np.nan)
                if case.fault == 'holes':
                    depth[scene.holes] = np.nan
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', np.RankWarning)
                    runtime.observe(CameraFrame(timestamp, rgb, depth))
            except ValueError as error:
                if str(error) != 'No time-aligned attitude for camera frame':
                    raise
                exceptions.append(dict(time=timestamp, error=str(error)))
        command = runtime.command(timestamp)
        boxes = [record['center'].tolist() for record in runtime.perception.obstacle_memory]
        if timestamp < start:
            baseline_boxes = boxes
        current_white = None if runtime.observation is None else runtime.observation.current_white
        rows.append(dict(time=timestamp, fault=faulty, observed=observed,
            reason=runtime.reason, mode=runtime.navigator.mode, command=command.tolist(),
            accepted=runtime.odometry.accepted, method=runtime.odometry.method,
            floor_reason=runtime.odometry.floor.reason, good_frames=runtime.good_frames,
            boxes=boxes, pose=runtime.pose.tolist(),
            fixture_position_m=position,
            pose_error_m=float(np.linalg.norm(runtime.xy - [position, 0.])),
            fresh_white=0 if current_white is None else len(current_white),
            localization_reseeds=runtime.localization_reseeds))
    fault_rows = [row for row in rows if row['fault']]
    after = [row for row in rows if row['time'] >= end]
    stopped = [row for row in fault_rows if not np.any(row['command'])]
    ready_after = [row for row in after if row['reason'] == 'ready']
    resumed = [row for row in after if row['reason'] == 'ready' and np.any(row['command'])]
    tracked_after = [row for row in after if row['observed'] and row['accepted']]
    lost = [row for row in rows if row['time'] >= start and not row['accepted']]
    memory_errors = [float(np.max(abs(np.array(row['boxes']) - baseline_boxes)))
                     for row in lost if len(row['boxes']) == len(baseline_boxes)]
    summary = dict(name=case.name, fault=case.fault, duration_s=case.duration,
        fault_start_s=start, restore_s=end, baseline_box_count=len(baseline_boxes),
        fixture_coasting_distance_m=case.coast_m,
        expected_recovery=case.expect_recovery,
        baseline_ready=any(row['reason'] == 'ready' for row in rows if row['time'] < start),
        stop_delay_s=None if not stopped else round(stopped[0]['time'] - start, 6),
        first_stop_reason=None if not stopped else stopped[0]['reason'],
        resume_delay_s=None if not resumed else round(resumed[0]['time'] - end, 6),
        ready_delay_s=None if not ready_after else round(ready_after[0]['time'] - end, 6),
        ready_camera_frames=None if not ready_after else sum(
            row['observed'] and row['time'] <= ready_after[0]['time'] for row in after),
        last_reason=rows[-1]['reason'], last_floor_reason=rows[-1]['floor_reason'],
        first_reaccepted_pose_error_m=None if not tracked_after else tracked_after[0]['pose_error_m'],
        final_pose_error_m=rows[-1]['pose_error_m'],
        minimum_box_count_during_fault=min(len(row['boxes']) for row in fault_rows),
        minimum_box_count_when_unlocalized=min((len(row['boxes']) for row in lost), default=None),
        maximum_box_movement_when_unlocalized_m=max(memory_errors, default=0.),
        nonzero_commands_during_fault=sum(bool(np.any(row['command'])) for row in fault_rows),
        ready_without_fresh_white_during_fault=sum(row['reason'] == 'ready' and row['fresh_white'] == 0 for row in fault_rows),
        localization_reseeds=runtime.localization_reseeds,
        stale_attitude_frame_rejections=len(exceptions),
        assessment='recovered' if resumed else 'did_not_recover')
    if resumed and (summary['first_reaccepted_pose_error_m'] > .02 or
                    summary['final_pose_error_m'] > .02):
        summary['assessment'] = 'resumed_with_localization_error'
    elif not resumed and not case.expect_recovery:
        summary['assessment'] = 'held_for_unobserved_motion'
    return dict(summary=summary, observations=rows, frame_rejections=exceptions)


class SensorRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case.name: run_case(case) for case in CASES}

    def test_faults_preserve_previously_observed_boxes(self):
        for case in CASES:
            with self.subTest(case=case.name):
                value = self.results[case.name]['summary']
                self.assertTrue(value['baseline_ready'])
                self.assertEqual(value['baseline_box_count'], 1)
                self.assertEqual(value['minimum_box_count_during_fault'], 1)
                if value['minimum_box_count_when_unlocalized'] is not None:
                    self.assertEqual(value['minimum_box_count_when_unlocalized'], 1)
                self.assertLess(value['maximum_box_movement_when_unlocalized_m'], 1e-12)

    def test_depth_camera_and_imu_faults_stop_at_their_watchdogs(self):
        limits = {'depth': .02, 'both': .02, 'camera': .35, 'imu': .15}
        for case in CASES:
            if case.fault not in limits:
                continue
            with self.subTest(case=case.name):
                value = self.results[case.name]['summary']
                self.assertIsNotNone(value['stop_delay_s'])
                self.assertLessEqual(value['stop_delay_s'], limits[case.fault])

    def test_restored_sensors_resume_navigation(self):
        for case in CASES:
            with self.subTest(case=case.name):
                value = self.results[case.name]['summary']
                if case.expect_recovery:
                    self.assertIsNotNone(value['resume_delay_s'], value['last_floor_reason'])
                    self.assertLessEqual(value['resume_delay_s'], .8)
                else:
                    self.assertIsNone(value['resume_delay_s'])

    def test_occluded_recovery_uses_floor_map_and_retains_box(self):
        rows = self.results['camera_pause_0_6s_10cm_coast_box_occluded']['observations']
        recovered = next(row for row in rows if row['method'] == 'floor_keyframe')
        self.assertLess(recovered['pose_error_m'], .003)
        self.assertEqual(recovered['good_frames'], 1)
        self.assertEqual(len(recovered['boxes']), 1)
        self.assertFalse(np.any(recovered['command']))

    def test_unusable_rgb_stops_even_when_box_depth_supports_motion(self):
        for name in ('rgb_black_0_6s_two_box_faces', 'rgb_saturated_0_4s_two_box_faces'):
            value = self.results[name]
            for row in value['observations']:
                if row['fault']:
                    np.testing.assert_array_equal(row['command'], np.zeros(3))
            self.assertLessEqual(value['summary']['ready_camera_frames'], 3)

    def test_short_lane_occlusion_keeps_usable_observed_corridor(self):
        scene = SensorScene(box_angle=np.pi/4)
        runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION)
        for index in range(11):
            timestamp = index*.1
            frame = scene.frame(timestamp, 0.)
            if timestamp >= .4:
                white = frame.rgb.min(axis=2)>115
                frame.rgb[white] = [19,125,78]
            runtime.set_attitude(timestamp,np.eye(3))
            runtime.observe(frame)
            command = runtime.command(timestamp)
        self.assertFalse(runtime.observation.lane_observed)
        self.assertTrue(runtime.rgb_usable)
        self.assertEqual(runtime.reason,'ready')
        self.assertGreater(np.linalg.norm(command),0.)

    def test_restored_visual_pose_preserves_metric_displacement(self):
        for case in CASES:
            with self.subTest(case=case.name):
                value = self.results[case.name]['summary']
                if not case.expect_recovery:
                    self.assertIsNone(value['first_reaccepted_pose_error_m'])
                    continue
                self.assertIsNotNone(value['first_reaccepted_pose_error_m'])
                self.assertLess(value['first_reaccepted_pose_error_m'], .02)
                self.assertLess(value['final_pose_error_m'], .02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SensorRecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, value in SensorRecoveryTests.results.items():
            (args.output / (name + '.json')).write_text(json.dumps(value, indent=2))
        summary = dict(navigation_core_sha256=core_hashes(),
            metric_pose_tolerance_m=.02,
            input='analytic perspective RGB-D with measured attitude, no mocked tracking',
            motion_model='prescribed sensor trajectory, normally held during interruption; cases include 5, 10 and 20 cm of additional coasting',
            cases=[value['summary'] for value in SensorRecoveryTests.results.values()])
        (args.output / 'sensor_recovery_summary.json').write_text(json.dumps(summary, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
