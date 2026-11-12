"""Straight-corridor feedback rejects curved and rotated white boundaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.visual import Perception, VisualNavigator, rotation2


class CorridorFeedbackTests(unittest.TestCase):
    def feedback(self, white, yaw=0., initial=.0, iterations=50):
        navigator = VisualNavigator()
        navigator.corridor_heading = yaw
        navigator.corridor_point = np.array([0., initial])@rotation2(yaw).T
        pose = np.r_[0., 0., yaw]
        # Retain an observed nearby box while the live lane fit is occluded.
        obstacles = np.array([[.6, .15, .2, .2]])
        for index in range(iterations):
            observation = Perception(index*.06, None, np.empty((0, 2)), obstacles,
                                     np.empty((0, 2)), .36, np.zeros((1, 1, 3), np.uint8), white)
            navigator.observe(observation, pose, 0.)
        return navigator.corridor_point@rotation2(yaw)

    def test_one_visible_parallel_boundary_updates_lateral_position(self):
        x = np.linspace(.35, 1.4, 150)
        white = np.c_[x, .25+.025+.002*np.sin(30*x)]
        result = self.feedback(white)
        self.assertAlmostEqual(result[1], .025, delta=.002)

    def test_curve_inside_expected_band_does_not_drag_corridor(self):
        x = np.linspace(.35, 1.4, 150)
        curve = .04*(x-.35)+.02*(x-.35)**2
        white = np.vstack([np.c_[x, side+curve] for side in (-.25, .25)])
        np.testing.assert_allclose(self.feedback(white), [0., 0.], atol=1e-12)

    def test_symmetric_curve_with_zero_mean_slope_is_rejected(self):
        x = np.linspace(.35, 1.4, 150)
        curve = .13*(x-.875)**2
        white = np.c_[x, .25+curve]
        np.testing.assert_allclose(self.feedback(white), [0., 0.], atol=1e-12)

    def test_straight_side_corrects_drift_while_curved_side_is_rejected(self):
        x = np.linspace(.35, 1.4, 150)
        straight = np.c_[x, np.zeros_like(x)]
        curve = np.c_[x, .5+.04*(x-.35)]
        result = self.feedback(np.vstack([straight, curve]), initial=.21)
        self.assertAlmostEqual(result[1], .25, delta=.001)

    def test_oblique_line_does_not_translate_corridor(self):
        x = np.linspace(.35, 1.4, 150)
        white = np.c_[x, .25+.06*(x-.875)]
        np.testing.assert_allclose(self.feedback(white), [0., 0.], atol=1e-12)

    def test_parallel_feedback_is_invariant_to_world_heading(self):
        x = np.linspace(.35, 1.4, 150)
        white = np.c_[x, -.25-.02+.001*np.sin(35*x)]
        result = self.feedback(white, yaw=.7)
        np.testing.assert_allclose(result, [0., -.02], atol=.002)


if __name__ == '__main__':
    unittest.main()
