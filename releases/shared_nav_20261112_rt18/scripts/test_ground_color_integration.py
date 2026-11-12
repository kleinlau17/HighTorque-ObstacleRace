"""Check shared RGB-D ground and obstacle processing using perspective depth."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.runtime import CameraFrame, NavigationRuntime
from navigation.surface_odometry import SurfaceOdometry
from navigation.visual import CameraPerception, rotation2


WIDTH, HEIGHT = 320, 240
INTRINSICS = (210., 210., 159.5, 119.5)
CAMERA_POSITION = np.array([.06, 0., .20])
SIN_PITCH = np.sqrt(.5)
CAMERA_ROTATION = np.array([[0., -SIN_PITCH, SIN_PITCH],
                            [-1., 0., 0.],
                            [0., -SIN_PITCH, -SIN_PITCH]])
BOX_CENTERS = np.array([[1., -.45], [1., 0.], [1., .45]])


def camera_perception(camera_position=CAMERA_POSITION):
    return CameraPerception(WIDTH, HEIGHT, 80., camera_position,
                            CAMERA_ROTATION, INTRINSICS)


def perspective_scene(colors, slope=(0., 0.), camera_position=CAMERA_POSITION,
                      box_angle=0., box_dimensions=(.2, .2, .4),
                      box_centers=BOX_CENTERS):
    """Analytic camera depth for a marked floor and three 20 x 20 x 40 cm boxes."""
    rays = camera_perception().rays
    directions = rays @ CAMERA_ROTATION.T
    a, b = slope
    floor_z = -.35
    denominator = directions[..., 2] - a * directions[..., 0] - b * directions[..., 1]
    numerator = a * camera_position[0] + b * camera_position[1] + floor_z - camera_position[2]
    depth = numerator / denominator
    floor_points = camera_position + directions * depth[..., None]
    rgb = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[:] = [19, 125, 78]
    markings = np.minimum(abs(floor_points[..., 1] - .25),
                          abs(floor_points[..., 1] + .25)) < .006
    rgb[markings] = [242, 243, 241]
    visible_box = np.full((HEIGHT, WIDTH), -1)
    # Separate the three boxes in the image so each has enough visible wall
    # height for a cold-start detection, independent of tracking history.
    box_rotation = np.eye(3)
    box_rotation[:2, :2] = rotation2(box_angle)
    for index, (center, color) in enumerate(zip(box_centers, colors)):
        origin = np.r_[center, floor_z]
        local_camera = (camera_position - origin) @ box_rotation
        local_directions = directions @ box_rotation
        low = np.array([-box_dimensions[0] / 2, -box_dimensions[1] / 2, 0.])
        high = low + box_dimensions
        with np.errstate(divide='ignore', invalid='ignore'):
            first = (low - local_camera) / local_directions
            second = (high - local_camera) / local_directions
        near = np.minimum(first, second).max(axis=2)
        far = np.maximum(first, second).min(axis=2)
        hit = (near > 0.) & (far >= near) & (near < depth)
        depth[hit] = near[hit]
        rgb[hit] = color
        visible_box[hit] = index
    return CameraFrame(0., rgb, depth), visible_box


class GroundColorIntegrationTests(unittest.TestCase):
    def project(self, frame, attitude=None, pose=None):
        attitude = np.eye(3) if attitude is None else attitude
        pose = np.zeros(3) if pose is None else pose
        perception = camera_perception()
        surface = SurfaceOdometry(perception.rays, CAMERA_POSITION, CAMERA_ROTATION)
        clouds = surface.project(frame, attitude)
        observation = perception.update(frame, pose, attitude, ground_plane=surface.ground.result)
        return surface, perception, clouds, observation

    def occluded_box_frame(self, frame, box_labels, kind, camera_position):
        body = (camera_perception().rays * frame.depth[..., None]) @ CAMERA_ROTATION.T + camera_position
        height = body[..., 2] + .35
        box = box_labels >= 0
        if kind == 'side':
            hidden = box & ((height < .14) | (height > .23))
        else:
            hidden = box & (height < .34)
            if kind == 'partial_top':
                hidden |= box & (body[..., 0] < 1.)
        depth = frame.depth.copy()
        depth[hidden] = np.nan
        return CameraFrame(.1, frame.rgb, depth)

    def test_known_box_survives_partial_height_and_corrects_visible_surfaces(self):
        camera_position = np.array([.06, 0., .45])
        frame, labels = perspective_scene([[19, 125, 78]] * 3, camera_position=camera_position)
        for kind in ('top', 'partial_top', 'side'):
            with self.subTest(kind=kind):
                perception = camera_perception(camera_position)
                initial = perception.update(frame, np.zeros(3), np.eye(3))
                self.assertEqual(len(initial.obstacles), 3)
                for record in perception.obstacle_memory:
                    record['center'] += [.018, .010]
                occluded = self.occluded_box_frame(frame, labels, kind, camera_position)
                result = perception.update(occluded, np.zeros(3), np.eye(3))
                self.assertEqual(len(result.obstacles), 3)
                boxes = result.obstacles[np.argsort(result.obstacles[:, 1])]
                # A clipped patch's midpoint cannot replace the known center.
                # A single far top edge is quantized by the perspective pixels.
                np.testing.assert_allclose(boxes[:, 0], BOX_CENTERS[:, 0], atol=.012)
                np.testing.assert_allclose(boxes[:, 1], BOX_CENTERS[:, 1], atol=.015)
                self.assertTrue(all(record['time'] == .1 for record in perception.obstacle_memory))

    def test_rotated_square_retains_observed_axes_and_heading_projection(self):
        camera_position = np.array([.06, 0., .45])
        angle = np.pi / 4
        frame, _ = perspective_scene([[35, 175, 70]], camera_position=camera_position,
                                     box_angle=angle, box_centers=np.array([[1., 0.]]))
        perception = camera_perception(camera_position)
        first = perception.update(frame, np.zeros(3), np.eye(3))
        self.assertEqual(len(first.obstacles), 1)
        np.testing.assert_allclose(first.obstacles[0, 2:], .2 * np.sqrt(2), atol=.002)
        axes = perception.obstacle_memory[0]['axes'].copy()
        # A new robot heading changes the projected width of a remembered box.
        empty, _ = perspective_scene([], camera_position=camera_position)
        yaw = .6
        attitude = np.eye(3)
        attitude[:2, :2] = rotation2(yaw)
        second = perception.update(CameraFrame(.1, empty.rgb, empty.depth),
                                   np.array([0., 0., yaw]), attitude)
        expected = .2 * (abs(np.cos(angle - yaw)) + abs(np.sin(angle - yaw)))
        np.testing.assert_allclose(second.obstacles[0, 2:], expected, atol=.002)
        np.testing.assert_array_equal(perception.obstacle_memory[0]['axes'], axes)

    def test_diagonal_internal_fragment_cannot_move_or_resize_known_box(self):
        center = np.array([[1., 0.]])
        frame, _ = perspective_scene([[35, 175, 70]], box_centers=center)
        perception = camera_perception()
        first = perception.update(frame, np.zeros(3), np.eye(3))
        self.assertEqual(len(first.obstacles), 1)
        before = first.obstacles.copy()
        axes = perception.obstacle_memory[0]['axes'].copy()
        # A separated diagonal board lies wholly inside the established box.
        # Its fitted face normal cannot redefine that box's outer geometry.
        fragment, _ = perspective_scene([[35, 175, 70]], box_angle=.95,
                                        box_dimensions=(.006, .11, .4),
                                        box_centers=center + [.025, .025])
        result = perception.update(CameraFrame(.1, fragment.rgb, fragment.depth),
                                   np.zeros(3), np.eye(3))
        self.assertEqual(len(result.obstacles), 1)
        np.testing.assert_allclose(result.obstacles, before, atol=1e-10)
        np.testing.assert_array_equal(perception.obstacle_memory[0]['axes'], axes)
        np.testing.assert_array_equal(perception.map_translation, np.zeros(2))

    def test_partial_height_cannot_bootstrap_unseen_box(self):
        camera_position = np.array([.06, 0., .45])
        frame, labels = perspective_scene([[19, 125, 78]] * 3, camera_position=camera_position)
        for kind in ('top', 'partial_top', 'side'):
            with self.subTest(kind=kind):
                occluded = self.occluded_box_frame(frame, labels, kind, camera_position)
                result = camera_perception(camera_position).update(occluded, np.zeros(3), np.eye(3))
                self.assertEqual(len(result.obstacles), 0)

    def test_pose_correction_moves_visible_and_hidden_boxes_together(self):
        frame, labels = perspective_scene([[19, 125, 78]] * 3)
        perception = camera_perception()
        perception.update(frame, np.zeros(3), np.eye(3))
        before = np.array([record['center'].copy() for record in perception.obstacle_memory])
        depth = frame.depth.copy()
        depth[labels == 1] = np.nan
        shifted = CameraFrame(.1, frame.rgb, depth)
        perception.update(shifted, np.array([.06, 0., 0.]), np.eye(3))
        after = np.array([record['center'] for record in perception.obstacle_memory])
        np.testing.assert_allclose(after - before, np.tile([.06, 0.], (3, 1)), atol=.002)
        np.testing.assert_allclose(perception.map_translation, [.06, 0.], atol=.002)
        hidden = CameraFrame(.2, frame.rgb, np.full_like(frame.depth, np.nan))
        perception.update(hidden, np.array([.06, 0., 0.]), np.eye(3))
        np.testing.assert_array_equal(perception.map_translation, np.zeros(2))

    def test_box_colors_preserve_ground_plane_and_obstacle_geometry(self):
        red, yellow, green, white = [220, 45, 45], [230, 195, 35], [19, 125, 78], [242, 243, 241]
        schemes = [[red, yellow, green], [green] * 3, [red] * 3, [yellow] * 3, [white] * 3]
        baseline = None
        for colors in schemes:
            with self.subTest(colors=colors):
                frame, visible_box = perspective_scene(colors)
                surface, _, clouds, observation = self.project(frame)
                self.assertTrue(surface.ground.result.valid)
                self.assertTrue(all(np.count_nonzero(visible_box == index) > 25 for index in range(3)))
                self.assertEqual(len(observation.obstacles), 3)
                self.assertIsNotNone(observation.lane)
                self.assertAlmostEqual(observation.ground_height, .35, delta=.001)
                ordered = observation.obstacles[np.argsort(observation.obstacles[:, 1])]
                np.testing.assert_allclose(ordered[:, :2], BOX_CENTERS, atol=.012)
                if baseline is None:
                    baseline = surface.ground.result, clouds, observation
                    continue
                plane, previous_clouds, previous_observation = baseline
                np.testing.assert_array_equal(surface.ground.result.normal, plane.normal)
                self.assertEqual(surface.ground.result.offset, plane.offset)
                np.testing.assert_array_equal(observation.obstacles, previous_observation.obstacles)
                np.testing.assert_array_equal(observation.lane, previous_observation.lane)
                np.testing.assert_array_equal(observation.current_white, previous_observation.current_white)
                # Vertical surfaces and top silhouettes use only geometry.
                for current, previous in zip(clouds[1:], previous_clouds[1:]):
                    np.testing.assert_array_equal(current, previous)

    def test_shared_tilted_plane_is_evaluated_before_heading_rotation(self):
        frame, _ = perspective_scene([], slope=(.06, -.04))
        yaw = .9
        attitude = np.eye(3)
        attitude[:2, :2] = rotation2(yaw)
        surface, _, _, shared = self.project(frame, attitude, np.array([0., 0., yaw]))
        independent = camera_perception().update(frame, np.array([0., 0., yaw]), attitude)
        self.assertTrue(surface.ground.result.valid)
        self.assertIsNotNone(shared.lane)
        self.assertAlmostEqual(shared.ground_height, .35, delta=.001)
        np.testing.assert_array_equal(shared.current_white, independent.current_white)
        self.assertGreater(len(shared.current_white), 100)
        self.assertLess(np.max(np.minimum(abs(shared.current_white[:, 1] - .25),
                                         abs(shared.current_white[:, 1] + .25))), .0061)
        self.assertAlmostEqual(np.polyval(shared.lane, .7), 0., delta=.01)

    def test_invalid_shared_plane_prevents_fallback_and_keeps_stored_obstacles(self):
        frame, _ = perspective_scene([[19, 125, 78]] * 3)
        surface, perception, _, observation = self.project(frame)
        self.assertEqual(len(observation.obstacles), 3)
        # Give the private estimator valid history, then share a failed current
        # estimate to exercise the actual runtime integration contract.
        perception.update(frame, np.zeros(3), np.eye(3))
        invalid = CameraFrame(.1, frame.rgb, np.full_like(frame.depth, np.nan))
        clouds = surface.project(invalid, np.eye(3))
        self.assertFalse(surface.ground.result.valid)
        self.assertTrue(all(len(cloud) == 0 for cloud in clouds))
        result = perception.update(frame, np.zeros(3), np.eye(3), ground_plane=surface.ground.result)
        self.assertIsNone(result.lane)
        self.assertTrue(np.isnan(result.ground_height))
        self.assertEqual(len(perception.obstacle_memory), 3)

    def test_runtime_stops_after_depth_loss_and_retains_obstacle_memory(self):
        frame, _ = perspective_scene([[19, 125, 78]] * 3)
        runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION)
        for index in range(5):
            timestamp = index * .1
            runtime.set_attitude(timestamp, np.eye(3))
            runtime.observe(CameraFrame(timestamp, frame.rgb, frame.depth))
        self.assertGreaterEqual(runtime.good_frames, 3)
        self.assertEqual(len(runtime.perception.obstacle_memory), 3)
        runtime.set_attitude(.5, np.eye(3))
        runtime.observe(CameraFrame(.5, frame.rgb, np.full_like(frame.depth, np.nan)))
        self.assertFalse(runtime.odometry.geometry.ground.result.valid)
        self.assertEqual(runtime.good_frames, 0)
        self.assertEqual(len(runtime.perception.obstacle_memory), 3)
        np.testing.assert_array_equal(runtime.command(.5), np.zeros(3))

    def test_feature_tracking_with_hidden_floor_still_inhibits_navigation(self):
        runtime = NavigationRuntime(WIDTH, HEIGHT, INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION)
        directions = runtime.perception.rays @ CAMERA_ROTATION.T
        depth = (.9 - CAMERA_POSITION[0]) / directions[..., 0]
        rows, columns = np.indices((HEIGHT, WIDTH))
        texture = ((rows // 12 + columns // 12) % 2 * 190 + 30).astype(np.uint8)
        rgb = np.repeat(texture[..., None], 3, axis=2)
        for index in range(4):
            timestamp = index * .1
            runtime.set_attitude(timestamp, np.eye(3))
            runtime.observe(CameraFrame(timestamp, rgb, depth))
        self.assertTrue(runtime.odometry.accepted)
        self.assertEqual(runtime.odometry.method, 'rgbd_features')
        self.assertFalse(runtime.odometry.geometry.ground.result.valid)
        self.assertIsNone(runtime.observation.lane)
        self.assertEqual(runtime.good_frames, 0)
        np.testing.assert_array_equal(runtime.command(.3), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
