"""Observed-map recovery regressions; no simulator or robot needed."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from navigation.runtime import CameraFrame, NavigationRuntime
from navigation.visual import Perception, rotation2


class MapRelocalizationTests(unittest.TestCase):
    def runtime(self):
        runtime = NavigationRuntime(32, 24, [30, 30, 15, 11], [0, 0, 0], np.eye(3))
        runtime.perception.obstacle_memory = [dict(center=np.array([.7, 0.]),
                                                  size=np.array([.2, .2]),
                                                  axes=np.eye(2), time=1.)]
        runtime.navigator.corridor_heading = 0.
        runtime.navigator.corridor_point = np.zeros(2)
        return runtime

    def test_trusted_box_corridor_retains_metric_reference_after_long_stop(self):
        runtime = self.runtime()
        runtime.xy[:] = [.1, -.12]
        runtime.lost_since = 2.
        runtime.perception.white_memory = [(1., np.array([[.4, .25]]), np.zeros(2))]
        cloud = runtime._observed_map_white(75.)
        np.testing.assert_allclose(np.unique(cloud[:, 1]), [-.13, .37])
        np.testing.assert_allclose(cloud[:, 0].min(), -.4)
        np.testing.assert_allclose(cloud[:, 0].max(), 2.7)
        # A corridor center stored far behind must not shorten the reference.
        runtime.navigator.corridor_point[0] = -20.
        np.testing.assert_allclose(runtime._observed_map_white(75.), cloud, atol=1e-12)

    def test_reference_follows_measured_world_heading(self):
        runtime = self.runtime()
        angle = .6
        rotation = rotation2(angle)
        runtime.yaw = angle
        runtime.perception.obstacle_memory[0]['center'] = np.array([.7, 0.])@rotation.T
        runtime.navigator.corridor_heading = angle
        runtime.navigator.corridor_point = np.array([.1, -.03])@rotation.T
        local = runtime._observed_map_white(90.)@rotation
        np.testing.assert_allclose(np.unique(np.round(local[:, 1], 8)), [-.28, .22])

    def test_does_not_invent_corridor_without_nearby_observed_box(self):
        runtime = self.runtime()
        runtime.perception.obstacle_memory[0]['center'][:] = [2., 0.]
        self.assertEqual(len(runtime._observed_map_white(75.)), 0)
        runtime.perception.obstacle_memory[0]['center'][:] = [.7, 0.]
        runtime.navigator.corridor_heading = None
        self.assertEqual(len(runtime._observed_map_white(75.)), 0)

    def test_last_measured_white_reference_does_not_expire_while_lost(self):
        runtime = self.runtime()
        runtime.navigator.corridor_heading = None
        points = np.array([[.4, .25], [.7, .25]])
        runtime.perception.white_memory = [(1., points, np.zeros(2))]
        runtime.lost_since = 2.
        np.testing.assert_array_equal(runtime._observed_map_white(75.), points)
        # Normal tracking still expires old local measurements.
        runtime.lost_since = None
        self.assertEqual(len(runtime._observed_map_white(75.)), 0)

    def measured_geometry(self, shift):
        along = np.linspace(1., 1.5, 40)
        white = np.vstack([np.c_[along, np.full_like(along, side)] for side in (-.25, .25)])
        # Only eleven points on the box's far top edge remain visible. Their
        # 7.5 cm initial residual is valid but Huber downweights them strongly.
        top = np.c_[np.full(11, .8), np.linspace(-.04, .04, 11)]
        return [np.asarray(white-shift, np.float32), np.empty((0, 2), np.float32),
                np.asarray(top-shift, np.float32)]

    def test_sparse_top_and_corridor_recover_75mm_then_reseed_tracker(self):
        runtime = self.runtime()
        runtime.lost_since = 2.
        runtime.set_attitude(75., np.eye(3))
        frame = CameraFrame(75., np.zeros((24, 32, 3), np.uint8), np.ones((24, 32)))
        correction = np.array([.075, -.003])
        measured = self.measured_geometry(correction)
        observation = Perception(75., np.zeros(3), np.empty((0, 2)), np.array([[.7, 0., .2, .2]]),
                                 np.empty((0, 2)), 0., frame.rgb)

        def unavailable_tracking(frame, attitude, xy):
            runtime.odometry.accepted = False
            return xy.copy()

        with patch.object(runtime.odometry, 'update', side_effect=unavailable_tracking) as tracker, \
                patch.object(runtime.odometry.geometry, 'project', return_value=measured), \
                patch.object(runtime.perception, 'update', return_value=observation):
            runtime.observe(frame)
        self.assertEqual(tracker.call_count, 2)
        np.testing.assert_allclose(runtime.xy, correction, atol=.001)
        self.assertTrue(runtime.odometry.accepted)
        self.assertEqual(runtime.odometry.method, 'observed_map')
        self.assertIsNone(runtime.lost_since)
        self.assertEqual(len(runtime.perception.obstacle_memory), 1)

    def test_corridor_alone_cannot_invent_forward_position(self):
        runtime = self.runtime()
        measured = self.measured_geometry(np.array([.075, -.003]))
        reference = [runtime.odometry.geometry.sample(runtime._observed_map_white(75.)),
                     np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)]
        self.assertIsNone(runtime.odometry.geometry.register(reference, measured))
        self.assertIsNone(runtime.odometry.geometry.register(reference, [np.empty((0, 2), np.float32)]*3))


if __name__ == '__main__':
    unittest.main()
