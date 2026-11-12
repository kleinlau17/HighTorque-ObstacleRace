"""Finish-line crossing is invariant to a later turn in place."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.visual import Perception, VisualNavigator, rotation2


class FinishTests(unittest.TestCase):
    def observation(self, point, time=1.):
        return Perception(time, np.zeros(3), np.empty((0, 2)), np.empty((0, 4)),
                          np.array([point]), .36, np.zeros((1, 1, 3), np.uint8))

    def test_finish_is_ignored_before_accumulated_lap_turn(self):
        navigator = VisualNavigator()
        navigator.observe(self.observation([1., 0.]), np.zeros(3), 0.)
        self.assertIsNone(navigator.finish_target)

    def test_turning_in_place_does_not_finish(self):
        navigator = VisualNavigator()
        navigator.observe(self.observation([1., 0.]), np.zeros(3), 6.)
        navigator.update(1., np.array([0., 0., np.pi]))
        self.assertFalse(navigator.finished)
        # Reobserving a stripe after turning cannot reverse the stored normal.
        navigator.observe(self.observation([.5, 0.], 1.1), np.array([0., 0., np.pi]), 6.+np.pi)
        np.testing.assert_array_equal(navigator.finish_normal, [1., 0.])
        np.testing.assert_array_equal(navigator.finish_target, [1., 0.])
        navigator.update(1.1, np.array([0., 0., np.pi]))
        self.assertFalse(navigator.finished)

    def test_forward_translation_across_observed_line_finishes(self):
        navigator = VisualNavigator()
        navigator.observe(self.observation([1., 0.]), np.zeros(3), 6.)
        navigator.update(1., np.array([.99, 0., 0.]))
        self.assertFalse(navigator.finished)
        navigator.update(1.02, np.array([1.01, 0., .4]))
        self.assertTrue(navigator.finished)

    def test_crossing_beyond_end_of_half_metre_stripe_does_not_finish(self):
        navigator = VisualNavigator()
        navigator.observe(self.observation([1., 0.]), np.zeros(3), 6.)
        navigator.update(1., np.array([1.01, .4, 0.]))
        self.assertFalse(navigator.finished)

    def test_finish_line_respects_observed_world_heading(self):
        navigator = VisualNavigator()
        pose = np.array([.4, -.2, .7])
        normal = rotation2(pose[2])[:, 0]
        navigator.observe(self.observation([.5, 0.]), pose, 6.)
        np.testing.assert_allclose(navigator.finish_normal, normal)
        before = np.r_[pose[:2]+.49*normal, pose[2]+np.pi]
        navigator.update(1., before)
        self.assertFalse(navigator.finished)
        after = np.r_[pose[:2]+.51*normal, pose[2]+np.pi]
        navigator.update(1.02, after)
        self.assertTrue(navigator.finished)

    def test_visible_stripe_corrects_accumulated_map_drift(self):
        navigator = VisualNavigator()
        navigator.observe(self.observation([1., 0.]), np.zeros(3), 6.)
        # Actual motion is 0.2 m/frame but the estimated map advances 0.1 m.
        # Each new direct stripe observation must correct the saved target.
        for i in range(1, 4):
            navigator.observe(self.observation([1.-.2*i, 0.], 1.+i*.1),
                              np.array([.1*i, 0., 0.]), 6.)
        np.testing.assert_allclose(navigator.finish_target, [.7, 0.])
        np.testing.assert_array_equal(navigator.finish_normal, [1., 0.])
        # The stripe then leaves view; measured motion crosses its last
        # observed position, without time/command extrapolation.
        navigator.update(1.32, np.array([.69, 0., 0.]))
        self.assertFalse(navigator.finished)
        navigator.update(1.34, np.array([.71, 0., 0.]))
        self.assertTrue(navigator.finished)


if __name__ == '__main__':
    unittest.main()
