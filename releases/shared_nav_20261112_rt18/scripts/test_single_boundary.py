"""Single visible boundary through perspective perception and shared runtime.

Scene geometry creates sensor inputs and supplies assertion targets only. The
runtime receives RGB-D and attitude through its ordinary public entry points.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from navigation.runtime import CameraFrame, NavigationRuntime
from navigation.visual import Perception, VisualNavigator, rotation2
from scripts.test_ground_color_integration import (
    CAMERA_POSITION, CAMERA_ROTATION, HEIGHT, INTRINSICS, WIDTH, camera_perception,
)


class SingleBoundaryScene:
    def __init__(self, curve=0):
        self.curve = curve
        self.radius = 1.25
        self.perception = camera_perception()
        directions = self.perception.rays @ CAMERA_ROTATION.T
        self.depth = (-.35 - CAMERA_POSITION[2]) / directions[..., 2]
        self.points = CAMERA_POSITION + directions * self.depth[..., None]
        random = np.random.default_rng(803)
        self.texture = cv2.GaussianBlur(random.normal(0., 4., (1024, 1024)).astype(np.float32),
                                       (0, 0), .6)

    def frame(self, timestamp, visible=(-1, 1), lateral=0., yaw=0.):
        world = self.points[..., :2] @ rotation2(yaw).T + [0., lateral]
        texture = cv2.remap(self.texture, (world[..., 0] / .002 + 300).astype(np.float32),
                            (world[..., 1] / .002 + 500).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        rgb = np.rint(np.array([19., 125., 78.]) + texture[..., None]).clip(0, 255).astype(np.uint8)
        if self.curve:
            radius = np.hypot(world[..., 0], world[..., 1] - self.curve * self.radius)
            lateral_distance = self.curve * (self.radius - radius)
        else:
            lateral_distance = world[..., 1]
        markings = np.zeros(self.depth.shape, bool)
        for side in visible:
            markings |= abs(lateral_distance - side * .25) < .005
        rgb[markings] = [242, 243, 241]
        return CameraFrame(timestamp, rgb, self.depth.copy())

    def center(self, x, lateral=0., yaw=0.):
        along = np.linspace(-.3, 1.15, 2000)
        y = self.curve * (self.radius - np.sqrt(self.radius ** 2 - along ** 2))
        body = (np.column_stack([along, y]) - [0., lateral]) @ rotation2(yaw)
        return np.interp(x, body[:, 0], body[:, 1])

    @staticmethod
    def attitude(yaw):
        value = np.eye(3)
        value[:2, :2] = rotation2(yaw)
        return value


class SingleBoundaryTests(unittest.TestCase):
    def assert_center(self, lane, scene, lateral=0., yaw=0., tolerance=.03):
        self.assertIsNotNone(lane)
        x = np.array([.35, .55, .75])
        np.testing.assert_allclose(np.polyval(lane, x), scene.center(x, lateral, yaw), atol=tolerance)

    def assert_steering(self, observation, scene, lateral=0., yaw=0.):
        self.assertTrue(observation.lane_observed)
        navigator = VisualNavigator()
        pose = np.array([0., lateral, yaw])
        navigator.observe(observation, pose, 0.)
        command = navigator.update(observation.time, pose, .1)
        self.assertEqual(navigator.mode, 'lane')
        self.assertGreater(command[0], 0.)
        expected_y = scene.center(.18, lateral, yaw)
        expected_slope = (scene.center(.19, lateral, yaw) - scene.center(.17, lateral, yaw)) / .02
        if abs(expected_y) > .025:
            self.assertGreater(command[1] * expected_y, 0., (command, expected_y))
        if abs(expected_slope) > .05:
            self.assertGreater(command[2] * expected_slope, 0., (command, expected_slope))

    def test_cold_start_single_straight_boundary_inside_lane(self):
        for side in (-1, 1):
            for lateral, yaw in ((-.08, -.12), (.08, .12)):
                with self.subTest(side=side, lateral=lateral, yaw=yaw):
                    scene = SingleBoundaryScene()
                    pose = np.array([0., lateral, yaw])
                    result = scene.perception.update(scene.frame(0., [side], lateral, yaw),
                                                     pose, scene.attitude(yaw))
                    self.assert_center(result.lane, scene, lateral, yaw)
                    self.assert_steering(result, scene, lateral, yaw)

    def test_cold_start_single_curved_boundary_inside_lane(self):
        for curve in (-1, 1):
            for side in (-1, 1):
                with self.subTest(curve=curve, side=side):
                    scene = SingleBoundaryScene(curve)
                    result = scene.perception.update(scene.frame(0., [side]), np.zeros(3), np.eye(3))
                    self.assert_center(result.lane, scene)
                    self.assert_steering(result, scene)

    def test_single_boundary_keeps_identity_with_dual_boundary_history(self):
        for curve in (-1, 0, 1):
            for side in (-1, 1):
                for lateral in (-.32, .32):
                    with self.subTest(curve=curve, side=side, lateral=lateral):
                        scene = SingleBoundaryScene(curve)
                        initial = scene.perception.update(scene.frame(0.), np.zeros(3), np.eye(3))
                        self.assert_center(initial.lane, scene)
                        # Old dual-line pixels expire from the fresh fit, while
                        # its transformed centerline still gives side identity.
                        result = scene.perception.update(scene.frame(3., [side], lateral),
                                                         np.array([0., lateral, 0.]), np.eye(3))
                        self.assert_center(result.lane, scene, lateral)
                        if len(result.current_white):
                            self.assert_steering(result, scene, lateral)
                        else:
                            # At two far-side curve poses the entire remaining
                            # stripe falls outside the actual camera view.
                            self.assertFalse(result.lane_observed)

    def test_cold_outside_line_has_same_image_as_opposite_inside_line(self):
        # The implementation initializes side identity assuming the body is
        # inside the lane. With no history, these two physical situations have
        # identical geometry and markings, and their real centers differ 50 cm.
        for side in (-1, 1):
            with self.subTest(side=side):
                outside, inside = SingleBoundaryScene(), SingleBoundaryScene()
                outside_lateral, inside_lateral = side * .32, -side * .18
                first = outside.frame(0., [side], outside_lateral)
                second = inside.frame(0., [-side], inside_lateral)
                for frame in (first, second):
                    frame.rgb[frame.rgb.min(axis=2) < 115] = [19, 125, 78]
                np.testing.assert_array_equal(first.rgb, second.rgb)
                np.testing.assert_array_equal(first.depth, second.depth)
                first_result = outside.perception.update(first, np.zeros(3), np.eye(3))
                second_result = inside.perception.update(second, np.zeros(3), np.eye(3))
                np.testing.assert_array_equal(first_result.lane, second_result.lane)
                self.assert_center(second_result.lane, inside, inside_lateral)
                self.assertAlmostEqual(abs(outside_lateral - inside_lateral), .50)

    def test_disjoint_curve_segments_cannot_form_two_lane_boundaries(self):
        perception = camera_perception()
        near_x = np.linspace(-.10, .98, 180)
        far_x = np.linspace(.963, 1.275, 47)
        near = np.column_stack([near_x, .2409 + .0077 * near_x - .09895 * near_x ** 2])
        far = np.column_stack([far_x, .79035 - .65714 * far_x])
        lane = perception.fit_lane(np.vstack([near, far]), np.zeros(3))
        self.assertIsNotNone(lane)
        self.assertLess(abs(np.polyval(lane, .4)), .08)

    def test_historical_and_legacy_white_support_remain_compatible(self):
        for current in (None, np.empty((0, 2))):
            for white in (np.empty((0, 2)), np.array([[.35, .1], [.7, .2]])):
                with self.subTest(current=current is None, history=len(white)):
                    perception = Perception(0., np.array([0., .2, 0.]), white,
                                            np.empty((0, 4)), np.empty((0, 2)), .35,
                                            np.zeros((1, 1, 3), np.uint8), current)
                    navigator = VisualNavigator()
                    navigator.observe(perception, np.zeros(3), 0.)
                    command = navigator.update(0., np.zeros(3), .1)
                    self.assertEqual(navigator.mode, 'lane')
                    self.assertGreater(command[2], 0.)

    def test_old_lane_memory_cannot_claim_current_boundary_observation(self):
        scene = SingleBoundaryScene()
        first = scene.perception.update(scene.frame(0.), np.zeros(3), np.eye(3))
        self.assertTrue(first.lane_observed)
        hidden = scene.frame(.1, [])
        remembered = scene.perception.update(hidden, np.zeros(3), np.eye(3))
        self.assertIsNotNone(remembered.lane)
        self.assertEqual(len(remembered.current_white), 0)
        self.assertFalse(remembered.lane_observed)
        transverse = scene.frame(.2, [])
        transverse.rgb[abs(scene.points[..., 0] - .6) < .006] = [242, 243, 241]
        stripe = scene.perception.update(transverse, np.zeros(3), np.eye(3))
        self.assertGreater(len(stripe.current_white), 12)
        self.assertFalse(stripe.lane_observed)
        invalid = CameraFrame(.3, first.overlay, np.full_like(hidden.depth, np.nan))
        failed = scene.perception.update(invalid, np.zeros(3), np.eye(3))
        self.assertFalse(failed.lane_observed)

    def test_shared_runtime_single_boundary_has_correct_command(self):
        for curve in (-1, 0, 1):
            for side in (-1, 1):
                for history in (False, True):
                    with self.subTest(curve=curve, side=side, history=history):
                        scene = SingleBoundaryScene(curve)
                        runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS,
                                                    CAMERA_POSITION, CAMERA_ROTATION)
                        lateral = .08 if curve == 0 else 0.
                        for i in range(38):
                            stamp = i * .08
                            visible = (-1, 1) if history and i < 4 else (side,)
                            runtime.set_attitude(stamp, np.eye(3))
                            result = runtime.observe(scene.frame(stamp, visible, lateral))
                        # Observe a stationary sensor sequence, then request a
                        # command once. No physical motion is simulated here.
                        runtime.command(stamp)
                        self.assertTrue(runtime.odometry.accepted)
                        self.assertGreaterEqual(runtime.good_frames, 3)
                        self.assert_center(result.lane, scene, lateral)
                        self.assertEqual(runtime.reason, 'ready')
                        if curve:
                            self.assertGreater(runtime.desired[2] * curve, 0.)
                        else:
                            self.assertLess(runtime.desired[1], 0.)

    def test_shared_runtime_retains_side_when_body_crosses_visible_boundary(self):
        for side in (-1, 1):
            for direction in (-1, 1):
                with self.subTest(side=side, direction=direction):
                    scene = SingleBoundaryScene()
                    runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS,
                                                CAMERA_POSITION, CAMERA_ROTATION)
                    for i in range(58):
                        stamp = i * .08
                        lateral = direction * min(.32, max(0., i - 4) * .016)
                        visible = (-1, 1) if i < 4 else (side,)
                        runtime.set_attitude(stamp, np.eye(3))
                        result = runtime.observe(scene.frame(stamp, visible, lateral))
                    runtime.command(stamp)
                    self.assertTrue(runtime.odometry.accepted)
                    self.assertEqual(runtime.reason, 'ready')
                    self.assert_center(result.lane, scene, direction * .32)
                    self.assertLess(runtime.desired[1] * direction, 0.)


if __name__ == '__main__':
    unittest.main()
