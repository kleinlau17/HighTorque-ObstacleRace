"""Dense floor-motion regressions with sensor-only perspective inputs."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from navigation.floor_odometry import FloorOdometry
from navigation.ground import GroundEstimator
from navigation.runtime import CameraFrame
from navigation.visual import RGBDOdometry
from scripts.test_ground_color_integration import (
    CAMERA_POSITION, CAMERA_ROTATION, HEIGHT, INTRINSICS, WIDTH, camera_perception,
)


def attitude(yaw, pitch):
    cz, sz, cy, sy = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]]) @ np.array([
        [cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])


class FloorOdometryTests(unittest.TestCase):
    def setUp(self):
        self.floor = FloorOdometry(INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION)
        rng = np.random.default_rng(97)
        self.texture = cv2.GaussianBlur(rng.normal(0., 1., (1024, 1024)).astype(np.float32), (0, 0), .6)
        self.texture = 100. + self.texture * 1.5
        self.rays = camera_perception().rays

    def frame(self, position, orientation, timestamp):
        direction = self.rays @ CAMERA_ROTATION.T @ orientation.T
        camera = CAMERA_POSITION @ orientation.T
        depth = (-.35 - camera[2]) / direction[..., 2]
        points = camera + direction * depth[..., None]
        world_x = points[..., 0] + position[0]
        world_y = points[..., 1] + position[1]
        gray = cv2.remap(self.texture, ((world_x + .1) / .002).astype(np.float32),
                         ((world_y + 1.) / .002).astype(np.float32), cv2.INTER_LINEAR)
        rgb = np.repeat(np.rint(gray).clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
        return CameraFrame(timestamp, rgb, depth)

    def plane(self, frame, orientation):
        body = (self.rays * frame.depth[..., None]) @ CAMERA_ROTATION.T + CAMERA_POSITION
        valid = np.isfinite(frame.depth) & (body[..., 0] > .3) & (frame.depth < 2.7)
        return GroundEstimator().estimate(body @ orientation.T, valid, timestamp=frame.time)

    def marked_frame(self, position, timestamp):
        orientation = attitude(0., 0.)
        frame = self.frame(position, orientation, timestamp)
        body = (self.rays * frame.depth[..., None]) @ CAMERA_ROTATION.T + CAMERA_POSITION
        markings = ((np.minimum(abs(body[..., 1] - .25), abs(body[..., 1] + .25)) < .007) |
                    (abs(body[..., 0] + position[0] - 1.43) < .007))
        frame.rgb[markings] = 240
        # The transverse line enters measured depth only after 36 cm of
        # motion. Earlier parallel markings cannot constrain forward position.
        frame.depth[frame.depth > 1.1] = np.nan
        return frame

    def test_low_contrast_texture_recovers_speeds_and_stop(self):
        field = self.texture[:300, :200] - 100.
        reference = field[40:240, 40:140]
        for forward_pixels in (0, 2, 7, 13):
            with self.subTest(forward_pixels=forward_pixels):
                current = field[40 + forward_pixels:240 + forward_pixels, 40:140]
                result = self.floor.match(reference, current, .04)
                self.assertIsNotNone(result, self.floor.reason)
                np.testing.assert_allclose(result, [forward_pixels * .002, 0.], atol=.001)

    def test_unobservable_or_unrelated_images_are_rejected(self):
        rng = np.random.default_rng(18)
        texture = rng.normal(0, .5, (200, 100)).astype(np.float32)
        cases = [
            (np.zeros_like(texture), np.zeros_like(texture)),
            (texture * .02, texture * .02),
            (np.repeat(np.sin(np.arange(200) * .3)[:, None], 100, axis=1).astype(np.float32),) * 2,
            (texture, rng.normal(0, .5, texture.shape).astype(np.float32)),
        ]
        for old, new in cases:
            self.assertIsNone(self.floor.match(old, new, .04))

    def test_measured_plane_and_attitude_compensate_rotation(self):
        first_attitude, second_attitude = attitude(0., 0.), attitude(.012, .02)
        first = self.frame([0., 0.], first_attitude, 0.)
        second = self.frame([.016, -.004], second_attitude, .08)
        result = self.floor.estimate(
            cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY), first.depth, first_attitude, self.plane(first, first_attitude),
            cv2.cvtColor(second.rgb, cv2.COLOR_RGB2GRAY), second.depth, second_attitude, self.plane(second, second_attitude), .08)
        self.assertIsNotNone(result, self.floor.reason)
        np.testing.assert_allclose(result, [.016, -.004], atol=.002)

    def test_missing_depth_and_outside_view_cannot_supply_texture(self):
        orientation = attitude(0., 0.)
        frame = self.frame([0., 0.], orientation, 0.)
        plane = self.plane(frame, orientation)
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        self.assertIsNone(self.floor.project(gray, np.full_like(frame.depth, np.nan), orientation, plane, 0.))
        self.assertIsNone(self.floor.project(gray, frame.depth, orientation, plane, np.pi / 2))

    def test_long_camera_gap_recovers_measured_coasting_displacement(self):
        orientation = attitude(0., 0.)
        first = self.frame([0., 0.], orientation, 0.)
        second = self.frame([.062, 0.], orientation, 2.)
        result = self.floor.estimate(
            cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY), first.depth, orientation, self.plane(first, orientation),
            cv2.cvtColor(second.rgb, cv2.COLOR_RGB2GRAY), second.depth, orientation, self.plane(second, orientation), 2.)
        self.assertIsNotNone(result, self.floor.reason)
        np.testing.assert_allclose(result, [.062, 0.], atol=.002)

    def test_long_gap_cannot_infer_missing_texture_or_excess_displacement(self):
        orientation = attitude(0., 0.)
        first = self.frame([0., 0.], orientation, 0.)
        old_gray = cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY)
        for distance in (.081, .10):
            with self.subTest(distance_m=distance):
                second = self.frame([distance, 0.], orientation, 2.)
                self.assertIsNone(self.floor.estimate(
                    old_gray, first.depth, orientation, self.plane(first, orientation),
                    cv2.cvtColor(second.rgb, cv2.COLOR_RGB2GRAY), second.depth, orientation,
                    self.plane(second, orientation), 2.))
        blank = np.full_like(old_gray, 100)
        self.assertIsNone(self.floor.estimate(
            blank, first.depth, orientation, self.plane(first, orientation),
            blank, first.depth, orientation, self.plane(first, orientation), 2.))
        self.assertEqual(self.floor.reason, 'insufficient_floor_texture')

    def test_bright_diagonal_marking_preserves_measured_floor_texture(self):
        from scripts.test_single_boundary import SingleBoundaryScene
        yaw = np.deg2rad(30.)
        orientation = attitude(yaw, 0.)
        for textured in (True, False):
            with self.subTest(textured=textured):
                scene = SingleBoundaryScene()
                if not textured:
                    scene.texture[:] = 0.
                first = scene.frame(0., lateral=0., yaw=yaw)
                second = scene.frame(.08, lateral=.016, yaw=yaw)
                result = self.floor.estimate(
                    cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY), first.depth, orientation,
                    self.plane(first, orientation),
                    cv2.cvtColor(second.rgb, cv2.COLOR_RGB2GRAY), second.depth, orientation,
                    self.plane(second, orientation), .08)
                if textured:
                    self.assertIsNotNone(result, self.floor.reason)
                    np.testing.assert_allclose(result, [0., .016], atol=.002)
                else:
                    self.assertIsNone(result)
                    self.assertEqual(self.floor.reason, 'insufficient_floor_texture')

    def test_rgbd_entry_point_uses_floor_texture_and_tracks_stop(self):
        orientation = attitude(0., 0.)
        odometry = RGBDOdometry(WIDTH, HEIGHT, 80., CAMERA_POSITION, CAMERA_ROTATION, INTRINSICS)
        xy = odometry.update(self.frame([0., 0.], orientation, 0.), orientation, np.zeros(2))
        xy = odometry.update(self.frame([.016, 0.], orientation, .08), orientation, xy)
        self.assertTrue(odometry.accepted)
        self.assertEqual(odometry.method, 'rgbd_floor')
        np.testing.assert_allclose(xy, [.016, 0.], atol=.002)
        stopped = odometry.update(self.frame([.016, 0.], orientation, .16), orientation, xy)
        self.assertEqual(odometry.method, 'rgbd_floor')
        np.testing.assert_allclose(stopped, xy, atol=.0002)

    def test_long_floor_segment_reacquires_new_measured_geometry(self):
        orientation = attitude(0., 0.)
        odometry = RGBDOdometry(WIDTH, HEIGHT, 80., CAMERA_POSITION, CAMERA_ROTATION, INTRINSICS)
        xy = np.zeros(2)
        for index in range(24):
            xy = odometry.update(self.marked_frame([index * .016, 0.], index * .08), orientation, xy)
            if index:
                self.assertEqual(odometry.method, 'rgbd_floor')
        self.assertGreater(xy[0], .35)
        self.assertIsNone(odometry.surface_reference)
        self.assertEqual(odometry.recent_surface_reference[2], 0.)
        candidate = odometry.candidate_surface_reference
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate[2], 1.84)
        xy = odometry.update(self.marked_frame([.384, 0.], 1.92), orientation, xy)
        self.assertEqual(odometry.method, 'rgbd_surfaces')
        self.assertIs(odometry.surface_reference, candidate)
        self.assertIsNone(odometry.candidate_surface_reference)
        np.testing.assert_allclose(xy, [.384, 0.], atol=.004)
        xy = odometry.update(self.marked_frame([.400, 0.], 2.), orientation, xy)
        self.assertEqual(odometry.method, 'rgbd_surfaces')
        np.testing.assert_allclose(xy, [.400, 0.], atol=.004)


if __name__ == '__main__':
    unittest.main()
