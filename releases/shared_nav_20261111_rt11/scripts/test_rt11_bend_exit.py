"""An unconfirmed return near a bend must not freeze the incoming drive command."""
from pathlib import Path
import json
from types import SimpleNamespace
import unittest

import numpy as np

from navigation.runtime import OnboardNavigation


class BendExit(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root/'deploy/pro_model_config.json').read_text())
        self.nav = OnboardNavigation(64, 48, (55,55,31,23), cfg['camera_position_body_m'],
                                     cfg['camera_rotation_optical_to_body'], .22, 1.)
        self.addCleanup(self.nav.worker.shutdown)
        self.nav.set_attitude(1., np.eye(3))
        self.nav.last_frame_time = 1.
        self.nav.tracked.observation = SimpleNamespace(obstacles=[])
        self.nav.tracking_control = True
        self.nav.unconfirmed_sites[(0,0)] = (np.array([.8,.1,.15]), 0, .5)
        self.nav.scene = 'clear'
        self.nav.desired[:] = [.22,0.,.24]
        self.nav.last_drive[:] = [.22,0.]

    def test_new_exit_direction_overrides_carried_bend_turn(self):
        self.nav.measured_lane = (.98, -.26, -.04)
        self.nav.command(1.02)
        self.assertEqual(self.nav.reason, 'follow_recovered_lane')
        self.assertLess(self.nav.desired[2], 0.)
        self.assertLess(self.nav.desired[1], 0.)
        self.assertGreater(self.nav.desired[0], 0.)
        self.assertLess(self.nav.desired[0], .22)

    def test_missing_line_uses_yaw_instead_of_carrying_speed_outside(self):
        self.nav.measured_lane = (.05,-.5,0.)
        self.nav.command(1.02)
        self.assertEqual(self.nav.reason, 'turn_to_recover_lane')
        np.testing.assert_equal(self.nav.desired[:2], [0.,0.])
        self.assertLess(self.nav.desired[2], 0.)
        self.nav.measured_lane = (1.01,.01,0.)
        self.nav.command(1.04)
        self.assertGreater(self.nav.desired[0], 0.)
