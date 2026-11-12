"""Verify every image interpolation source has measured floor depth."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts import test_floor_odometry as fixtures


class FloorProjectionTests(unittest.TestCase):
    def setUp(self):
        scene = fixtures.FloorOdometryTests()
        scene.setUp()
        self.floor = scene.floor
        self.attitude = fixtures.attitude(0., 0.)
        frame = scene.frame([0., 0.], self.attitude, 0.)
        self.depth = frame.depth
        self.gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        self.plane = scene.plane(frame, self.attitude)
        level = np.stack([self.floor.x, self.floor.y,
                          np.full_like(self.floor.x, self.plane.z0)], axis=-1)
        optical = (level @ self.attitude - self.floor.position) @ self.floor.rotation
        fx, fy, cx, cy = self.floor.intrinsics
        self.map_x = (fx * optical[..., 0] / optical[..., 2] + cx).astype(np.float32)
        self.map_y = (fy * optical[..., 1] / optical[..., 2] + cy).astype(np.float32)
        width = self.gray.shape[1]
        nearest = np.rint(self.map_y).astype(int) * width + np.rint(self.map_x).astype(int)
        columns, rows = np.floor(self.map_x).astype(int), np.floor(self.map_y).astype(int)
        contributors = np.concatenate([
            ((rows + dy) * width + columns + dx).ravel()
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ])
        self.additional_sources = np.setdiff1d(contributors, nearest)
        self.assertGreater(len(self.additional_sources), 100)

    def project(self, depth, gray=None):
        return self.floor.project(self.gray if gray is None else gray, depth,
                                  self.attitude, self.plane, 0.)

    def test_valid_perspective_floor_remains_usable(self):
        projected = self.project(self.depth)
        self.assertIsNotNone(projected)
        self.assertTrue(np.isfinite(projected).all())

    def test_window_moves_inside_shallower_view_and_stays_registered(self):
        # Start in the forward-leaning standby pose, then straighten the torso.
        import json
        from navigation.floor_odometry import FloorOdometry
        root = Path(__file__).resolve().parents[1]
        camera = json.loads((root/'deploy/pro_config.json').read_text())
        sensor = json.loads((root/'sim/d435i_camera.json').read_text())
        k = [sensor['fx']/2, sensor['fy']/2, (sensor['cx']+.5)/2-.5, (sensor['cy']+.5)/2-.5]
        floor = FloorOdometry(k, camera['camera_position_body_m'], camera['camera_rotation_optical_to_body'])
        floor.select_window(np.full((240, 320), np.nan), fixtures.attitude(0., np.deg2rad(5.)), self.plane)
        selected = floor.x.copy()
        self.assertGreater(selected.min(), .45)
        floor.select_window(np.full((240, 320), np.nan), np.eye(3), self.plane)
        np.testing.assert_array_equal(selected, floor.x)

    def test_selection_finds_measured_depth_around_a_near_hole(self):
        holes = self.depth.copy()
        # Occlude only the first few centimetres of the nominal patch.
        holes[np.rint(self.map_y[:7]).astype(int), np.rint(self.map_x[:7]).astype(int)] = np.nan
        self.assertIsNone(self.project(holes))
        self.floor.select_window(holes, self.attitude, self.plane)
        self.assertGreater(self.floor.x.min(), .45)
        self.assertIsNotNone(self.project(holes))

    def test_non_nearest_interpolation_sources_require_valid_depth(self):
        original_nearest = cv2.remap(self.depth, self.map_x, self.map_y, cv2.INTER_NEAREST)
        for missing in (np.nan, np.inf, 0.):
            with self.subTest(missing=missing):
                depth = self.depth.copy()
                depth.flat[self.additional_sources] = missing
                gray = self.gray.copy()
                gray.flat[self.additional_sources] = 0
                # The old nearest-depth gate sees the exact same measurements.
                np.testing.assert_array_equal(
                    cv2.remap(depth, self.map_x, self.map_y, cv2.INTER_NEAREST), original_nearest)
                self.assertIsNone(self.project(depth, gray))

    def test_non_nearest_obstacle_pixels_cannot_supply_floor_texture(self):
        depth = self.depth.copy()
        depth.flat[self.additional_sources] -= .10
        gray = self.gray.copy()
        gray.flat[self.additional_sources] = 255
        self.assertIsNone(self.project(depth, gray))

    def test_unused_invalid_pixels_do_not_invalidate_observed_patch(self):
        baseline = self.project(self.depth)
        depth = self.depth.copy()
        gray = self.gray.copy()
        depth[0, 0] = np.nan
        gray[0, 0] = 0
        np.testing.assert_array_equal(self.project(depth, gray), baseline)


if __name__ == '__main__':
    unittest.main()
