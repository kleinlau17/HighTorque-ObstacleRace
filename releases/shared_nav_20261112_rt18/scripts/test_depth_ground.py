"""Ground geometry regressions independent of RGB, robot policy and MuJoCo."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.ground import GroundEstimator


def plane_patch(xlim, ylim, z0, shape=(48, 64), slope=(0., 0.), noise=0.):
    x, y = np.meshgrid(np.linspace(*xlim, shape[1]),
                       np.linspace(*ylim, shape[0]))
    z = z0 + slope[0] * x + slope[1] * y
    if noise:
        z = z + np.random.default_rng(18).normal(0., noise, z.shape)
    return np.stack([x, y, z], axis=-1)


class DepthGroundTests(unittest.TestCase):
    def test_tilted_noisy_floor_recovers_metric_plane(self):
        points = plane_patch((.3, 1.6), (-.5, .5), -.31,
                             slope=(.06, -.04), noise=.001)
        result = GroundEstimator().estimate(points, timestamp=0.)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.z0, -.31, delta=.002)
        expected = np.array([-.06, .04, 1.]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result.normal, expected, atol=.002)
        self.assertLess(result.residual, .002)

    def test_floor_wins_when_raised_box_top_has_more_pixels(self):
        # 90% of samples belong to the close box; pixel majority is misleading.
        floor = plane_patch((.4, 1.3), (-.3, .3), -.62, shape=(24, 32)).reshape(-1, 3)
        top = plane_patch((.6, .8), (-.1, .1), -.22, shape=(100, 100)).reshape(-1, 3)
        result = GroundEstimator().estimate(np.concatenate([floor, top]))
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.z0, -.62, delta=.001)
        self.assertAlmostEqual(float(np.median(result.height(top))), .4, delta=.001)

    def test_top_alone_is_not_bootstrapped_as_ground(self):
        top = plane_patch((.6, .8), (-.1, .1), -.22, shape=(100, 100))
        self.assertFalse(GroundEstimator().estimate(top).valid)

    def test_disconnected_box_tops_do_not_pool_into_floor_support(self):
        tops = [plane_patch((x, x + .2), (-.1, .1), -.22).reshape(-1, 3)
                for x in (.4, .9, 1.4)]
        self.assertFalse(GroundEstimator().estimate(np.concatenate(tops)).valid)

    def test_hidden_ground_is_invalid_and_does_not_return_stale_plane(self):
        estimator = GroundEstimator()
        ground = plane_patch((.3, 1.5), (-.4, .4), -.62)
        self.assertTrue(estimator.estimate(ground, timestamp=0.).valid)
        # Even a much bigger raised platform cannot replace a trusted floor.
        raised = plane_patch((.3, 1.5), (-.4, .4), -.22)
        hidden = estimator.estimate(raised, timestamp=.1)
        self.assertFalse(hidden.valid)
        self.assertTrue(np.isnan(hidden.z0))
        self.assertTrue(estimator.estimate(ground, timestamp=.2).valid)

    def test_partial_ground_uses_recent_evidence_but_not_stale_evidence(self):
        estimator = GroundEstimator()
        ground = plane_patch((.3, 1.5), (-.4, .4), -.31)
        self.assertTrue(estimator.estimate(ground, timestamp=0.).valid)
        strip = plane_patch((.4, .75), (-.055, .055), -.315)
        self.assertTrue(estimator.estimate(strip, timestamp=.2).valid)
        self.assertFalse(estimator.estimate(strip, timestamp=3.).valid)

    def test_vertical_wall_and_invalid_depth_are_not_floor(self):
        y, z = np.meshgrid(np.linspace(-.6, .6, 64), np.linspace(-.6, .4, 48))
        wall = np.stack([np.full_like(y, .8), y, z], axis=-1)
        self.assertFalse(GroundEstimator().estimate(wall).valid)
        invalid = np.full((48, 64, 3), np.nan)
        self.assertFalse(GroundEstimator().estimate(invalid).valid)

    def test_external_mask_excludes_untrusted_depth(self):
        ground = plane_patch((.3, 1.5), (-.4, .4), -.31)
        self.assertFalse(GroundEstimator().estimate(ground, np.zeros((48, 64), bool)).valid)
        with self.assertRaises(ValueError):
            GroundEstimator().estimate(ground, np.ones((48, 63), bool))


if __name__ == '__main__':
    unittest.main()
