"""Place recognition regressions with rendered floor pixels and metric depth."""
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from navigation.floor_map import FloorMap, patch_displacement
from scripts import test_floor_odometry as fixtures


class FloorMapTests(unittest.TestCase):
    def setUp(self):
        self.scene = fixtures.FloorOdometryTests()
        self.scene.setUp()
        self.atlas = FloorMap(self.scene.floor)

    def observe(self, time, xy, odom_xy, distance, yaw=0.):
        rotation = fixtures.attitude(yaw, .02)
        frame = self.scene.frame(xy, rotation, time)
        return self.atlas.update(frame, rotation, self.scene.plane(frame, rotation),
                                 np.array(odom_xy), distance)

    def test_export_preserves_the_projection_used_at_capture(self):
        self.observe(0., [0, 0], [0., 0.], 0.)
        floor = self.scene.floor
        position, rotation = floor.position.copy(), floor.rotation.copy()
        # Rebinding the projector after capture must not relabel old keyframes.
        floor.position = np.zeros(3)
        floor.rotation = np.eye(3)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'floor_map.npz'
            self.atlas.save(path)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['camera_position_body_m'], position[None, :])
                np.testing.assert_array_equal(archive['camera_rotation_optical_to_body'], rotation[None, :, :])
                np.testing.assert_array_equal(archive['intrinsics'], floor.intrinsics)

    def test_two_views_correct_global_translation_without_shifting_local_poses(self):
        self.observe(0., [0, 0], [0., 0.], 0)
        self.assertEqual(len(self.atlas.keys), 1)
        odom = np.array([-.39, .04])
        self.observe(30., [.012, .006], odom, 10., .04)
        np.testing.assert_array_equal(self.atlas.offset, [0., 0.])
        self.observe(30.3, [.014, .006], odom+[.002, 0], 10.002, .03)
        self.assertEqual(self.atlas.revisits, 1)
        np.testing.assert_allclose(self.atlas.offset+odom, [.012, .006], atol=.002)
        np.testing.assert_array_equal(self.atlas.keys[0].odom_xy, [0., 0.])

    def test_local_map_tracks_measured_motion_when_odometry_stalls(self):
        self.observe(0., [0, 0], [0., 0.], 0.)
        self.observe(.3, [.04, 0], [0., 0.], 0.)
        self.observe(.6, [.08, 0], [0., 0.], 0.)
        np.testing.assert_allclose(self.atlas.offset, [.08, 0], atol=.002)
        # After a large innovation, use the very next camera cycle.
        self.observe(.67, [.09, 0], [0., 0.], 0.)
        np.testing.assert_allclose(self.atlas.offset, [.09, 0], atol=.002)

    def test_unrelated_texture_cannot_confirm_a_pending_match(self):
        self.observe(0., [0, 0], [0., 0.], 0)
        self.observe(30., [.012, .006], [-.39, .04], 10.)
        self.assertIsNotNone(self.atlas.pending)
        self.scene.texture = np.flipud(self.scene.texture).copy()
        self.observe(30.3, [.012, .006], [-.39, .04], 10.)
        self.assertIsNone(self.atlas.pending)
        np.testing.assert_array_equal(self.atlas.offset, [0., 0.])

    def test_reseed_requires_measured_attachment_before_fusing_more_keys(self):
        self.observe(0., [0, 0], [0., 0.], 0)
        self.atlas.start_segment()
        self.assertFalse(self.atlas.attached)
        self.observe(2., [.70, 0], [.05, 0.], .05)
        self.assertEqual(len(self.atlas.keys), 1)
        self.assertFalse(self.atlas.attached)
        self.observe(2.3, [.02, 0], [.1, 0.], .1)
        self.assertFalse(self.atlas.attached)
        self.observe(2.6, [.024, 0], [.104, 0.], .104)
        self.assertTrue(self.atlas.attached)
        np.testing.assert_allclose(self.atlas.offset, [-.08, 0.], atol=.002)

    def test_periodic_and_one_dimensional_patches_are_rejected(self):
        rng = np.random.default_rng(23)
        repeat = np.tile(rng.normal(size=(20, 10)).astype(np.float32), (10, 10))
        stripe = np.tile(np.sin(np.arange(100)).astype(np.float32), (200, 1))
        for image in (repeat, stripe):
            self.assertIsNone(patch_displacement(image, image.copy(), self.scene.floor))

    def test_inconsistent_tiles_are_rejected(self):
        rng = np.random.default_rng(21)
        ref = cv2.GaussianBlur(rng.normal(size=(200, 100)).astype(np.float32), (0, 0), .6)
        current = ref.copy()
        current[:, :50] = np.roll(ref[:, :50], 9, axis=0)
        self.assertIsNone(patch_displacement(ref, current, self.scene.floor))


if __name__ == '__main__':
    unittest.main()
