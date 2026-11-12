"""Regressions for weak texture, curved markings, and geometry initialization."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from navigation.surface_odometry import SurfaceOdometry
from navigation.visual import RGBDOdometry
from navigation.runtime import CameraFrame


class RGBDOdometryTests(unittest.TestCase):
    def test_noisy_parallel_markings_do_not_supply_forward_position(self):
        along = np.linspace(-.8, .8, 220)
        line = np.c_[along, np.full_like(along, .25)]
        for noise in (0., .0007, .001, .003, .004):
            with self.subTest(noise_m=noise):
                rng = np.random.default_rng(7)
                reference = np.asarray(line+rng.normal(0, noise, line.shape), np.float32)
                current = np.asarray(line-[.04, .02]+rng.normal(0, noise, line.shape), np.float32)
                geometry = SurfaceOdometry(None, None, None)
                self.assertIsNone(geometry.register([reference], [current]))

    def test_shallow_curve_provides_measured_forward_constraint(self):
        along = np.linspace(-.8, .8, 260)
        reference = np.c_[along, .25+.08*along**2].astype(np.float32)
        translation = np.array([.045, -.025])
        rng = np.random.default_rng(11)
        current = np.asarray(reference-translation+rng.normal(0, .0007, reference.shape), np.float32)
        geometry = SurfaceOdometry(None, None, None)
        result = geometry.register([reference], [current])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, translation, atol=.004)

    def test_existing_pose_initializes_reference_with_longer_baseline(self):
        along = np.linspace(-.5, .5, 140)
        reference = [np.c_[along+.6, np.full_like(along, .25)].astype(np.float32),
                     np.c_[np.full_like(along, .8), along].astype(np.float32)]
        translation = np.array([.19, -.06])
        current = [np.asarray(points-translation, np.float32) for points in reference]
        geometry = SurfaceOdometry(None, None, None)
        self.assertIsNone(geometry.register(reference, current))
        np.testing.assert_allclose(geometry.register(reference, current, initial=[.17, -.05]),
                                   translation, atol=.002)

    def test_low_contrast_carpet_is_excluded_from_feature_candidates(self):
        rng = np.random.default_rng(5)
        weak = rng.integers(107, 115, (120, 160), dtype=np.uint8)
        depth = np.ones(weak.shape, np.float32)
        self.assertLess(RGBDOdometry.feature_mask(weak, depth).sum(), .01*weak.size)

    def test_strong_corners_survive_on_continuous_depth(self):
        y, x = np.indices((120, 160))
        image = np.asarray(((x//12+y//12)%2)*180+40, np.uint8)
        depth = np.ones(image.shape, np.float32)
        self.assertGreater(RGBDOdometry.feature_mask(image, depth).sum(), 100)
        depth[:, 60:] = .5
        mask = RGBDOdometry.feature_mask(image, depth)
        self.assertFalse(mask[:, 59:61].any())

    def test_rasterized_parallel_line_edges_are_not_metric_corners(self):
        from scripts.test_sensor_recovery import SensorScene
        from scripts.test_ground_color_integration import CAMERA_POSITION, CAMERA_ROTATION
        import cv2
        scene = SensorScene()
        frame = scene.frame(.9, .108)
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        body = (scene.rays * frame.depth[..., None]) @ CAMERA_ROTATION.T + CAMERA_POSITION
        line_edges = ((abs(abs(body[..., 1]) - .25) < .012) &
                      (body[..., 0] > .25) & (body[..., 0] < .65))
        raw_corners = cv2.cornerMinEigenVal(gray, 3) >= 1e-4
        self.assertGreater(np.count_nonzero(raw_corners & line_edges), 1000)
        self.assertFalse((RGBDOdometry.feature_mask(gray, frame.depth) & line_edges).any())

    def test_recent_geometry_bridges_an_unmatchable_old_keyframe(self):
        odometry = RGBDOdometry(32, 24, 80, np.zeros(3), np.eye(3))
        image = np.zeros((24, 32, 3), np.uint8)
        depth = np.ones((24, 32), np.float32)
        odometry.previous = (image[..., 0], depth, np.eye(3), np.zeros(2))
        odometry.previous_time = 1.
        along = np.linspace(-.4, .4, 100)
        reference = [np.c_[along+.7, np.full_like(along, .25)].astype(np.float32),
                     np.c_[np.full_like(along, .8), along].astype(np.float32),
                     np.empty((0, 2), np.float32)]
        displacement = np.array([.045, -.025])
        measured = [np.asarray(points-displacement, np.float32) for points in reference]
        odometry.surface_reference = ([np.empty((0, 2), np.float32)]*3, np.zeros(2), .5)
        odometry.recent_surface_reference = (reference, np.zeros(2), 1.)
        with patch.object(odometry.geometry, 'project', return_value=measured):
            result = odometry.update(CameraFrame(1.1, image, depth), np.eye(3), np.array([.03, -.02]))
        self.assertTrue(odometry.accepted)
        self.assertEqual(odometry.method, 'rgbd_surfaces')
        np.testing.assert_allclose(result, displacement, atol=.002)

    def test_trusted_reference_keeps_priority_over_floor_candidate(self):
        odometry = RGBDOdometry(32, 24, 80, np.zeros(3), np.eye(3))
        image = np.zeros((24, 32, 3), np.uint8)
        depth = np.ones((24, 32), np.float32)
        odometry.previous = (image[..., 0], depth, np.eye(3), np.zeros(2))
        odometry.previous_time = 1.
        along = np.linspace(-.4, .4, 100)
        reference = [np.c_[along+.7, np.full_like(along, .25)].astype(np.float32),
                     np.c_[np.full_like(along, .8), along].astype(np.float32),
                     np.empty((0, 2), np.float32)]
        displacement = np.array([.045, -.025])
        measured = [np.asarray(points-displacement, np.float32) for points in reference]
        trusted = (reference, np.zeros(2), 1.)
        odometry.surface_reference = trusted
        odometry.candidate_surface_reference = (reference, np.array([.05, 0.]), 1.)
        with patch.object(odometry.geometry, 'project', return_value=measured), \
                patch.object(odometry.geometry, 'register', wraps=odometry.geometry.register) as register:
            result = odometry.update(CameraFrame(1.1, image, depth), np.eye(3), np.array([.03, -.02]))
        self.assertEqual(register.call_count, 1)
        self.assertIs(odometry.surface_reference, trusted)
        self.assertIsNone(odometry.candidate_surface_reference)
        np.testing.assert_allclose(result, displacement, atol=.002)

    def test_unconfirmed_floor_candidate_cannot_become_an_anchor(self):
        odometry = RGBDOdometry(32, 24, 80, np.zeros(3), np.eye(3))
        image = np.zeros((24, 32, 3), np.uint8)
        depth = np.ones((24, 32), np.float32)
        odometry.previous = (image[..., 0], depth, np.eye(3), np.zeros(2))
        odometry.previous_time = 1.
        along = np.linspace(-.4, .4, 100)
        candidate = [np.c_[along+.7, np.full_like(along, .25)].astype(np.float32),
                     np.c_[np.full_like(along, .8), along].astype(np.float32),
                     np.empty((0, 2), np.float32)]
        odometry.candidate_surface_reference = (candidate, np.zeros(2), 1.)
        empty = [np.empty((0, 2), np.float32)] * 3
        with patch.object(odometry.geometry, 'project', return_value=empty):
            odometry.update(CameraFrame(1.1, image, depth), np.eye(3), np.zeros(2))
        self.assertFalse(odometry.accepted)
        self.assertIsNone(odometry.surface_reference)
        self.assertIsNone(odometry.candidate_surface_reference)

    def test_sparse_feature_acceptance_cannot_seed_geometry(self):
        odometry = RGBDOdometry(32, 24, 80, np.zeros(3), np.eye(3))
        image = np.zeros((24, 32, 3), np.uint8)
        depth = np.ones((24, 32), np.float32)
        odometry.previous = (image[..., 0], depth, np.eye(3), np.zeros(2))
        odometry.previous_time = 1.
        points = np.array([[[8., 5.]], [[24., 5.]], [[16., 19.]]], np.float32)
        current = points - np.array([[[.01 * odometry.fx, 0.]]], np.float32)
        status, error = np.ones((3, 1), np.uint8), np.zeros((3, 1), np.float32)
        empty = [np.empty((0, 2), np.float32)] * 3
        with patch.object(odometry.geometry, 'project', return_value=empty), \
                patch.object(odometry.geometry, 'observable', return_value=True) as observable, \
                patch.object(odometry.floor, 'estimate', return_value=None), \
                patch.object(odometry, 'feature_mask', return_value=np.ones(depth.shape, bool)), \
                patch('navigation.visual.cv2.goodFeaturesToTrack', return_value=points), \
                patch('navigation.visual.cv2.calcOpticalFlowPyrLK', side_effect=[
                    (current, status, error), (points, status, error)]):
            result = odometry.update(CameraFrame(1.1, image, depth), np.eye(3), np.zeros(2))
        self.assertEqual(odometry.method, 'rgbd_features')
        self.assertEqual(odometry.inliers, 3)
        self.assertEqual(observable.call_count, 0)
        self.assertIsNone(odometry.candidate_surface_reference)
        self.assertIsNone(odometry.surface_reference)
        np.testing.assert_allclose(result, [.01, 0.], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
