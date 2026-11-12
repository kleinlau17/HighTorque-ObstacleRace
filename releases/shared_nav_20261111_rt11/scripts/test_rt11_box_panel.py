"""Exercise the panel-to-controller mailbox without starting ROS or commanding hardware."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from test_rt11_box_steps import BoxServo, view
from deploy.box_panel import BoxPanel, VisionMailbox
from navigation.box_tuning import BoxTuning


class BoxPanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.panel = BoxPanel(root, '127.0.0.1', 0, root / 'missing-key').start()
        self.addCleanup(self.panel.close)
        self.mail = VisionMailbox(self.panel.vision)
        self.nav = BoxServo()

    def request(self, path, data=None, token=None):
        headers = {'Authorization': 'Bearer ' + (self.panel.token if token is None else token),
                   'Content-Type': 'application/json'}
        url = 'http://127.0.0.1:' + str(self.panel.server.server_port) + path
        body = json.dumps(data).encode() if data is not None else None
        try:
            with urlopen(Request(url, data=body, headers=headers), timeout=2) as r:
                return r.status, json.load(r)
        except HTTPError as e:
            with e:
                return e.code, json.load(e)

    def test_forward_adopts_only_after_zero_receipt_and_new_capture(self):
        nav = self.nav
        nav.phase = 'along'
        nav.along_started = 3.
        nav.frame = view(3., (.6, .47))
        first = nav.command(3.02, 0.)
        hook = nav.publication(nav.frame, 4.)
        hook.published(3.03, first)
        values = dict(asdict(BoxTuning()), forward_step=3., forward_speed=24.)
        self.assertEqual(self.request('/api/settings', dict(base=0, values=values))[0], 200)
        self.mail.update(nav, 3.05)
        state = self.request('/api/state')[1]
        self.assertEqual((state['revision'], state['control']['revision']), (1, 0))
        self.assertAlmostEqual(nav.command(3.08, 0.)[0], .22)
        self.assertAlmostEqual(nav.steps.active['distance'], .063)
        np.testing.assert_equal(nav.command(3.60, 0.), np.zeros(3))
        nav.frame = view(3.61, (.65, .47))
        np.testing.assert_equal(nav.command(3.63, 0.), np.zeros(3))
        self.assertEqual(nav.tuning_revision, 0)
        hook.published(3.64, np.zeros(3))
        np.testing.assert_equal(nav.command(3.65, 0.), np.zeros(3))
        self.assertEqual(nav.tuning_revision, 0)
        nav.frame = view(3.70, (.65, .47))
        self.assertAlmostEqual(nav.command(3.72, 0.)[0], .24)
        self.assertAlmostEqual(nav.steps.active['distance'], .03)
        self.mail.update(nav, 3.73)
        self.assertEqual(self.request('/api/state')[1]['control']['applied'], values)

    def test_first_box_gate_and_side_limit_and_distance(self):
        for threshold, expected_axis in [(39, 0), (44, 1)]:
            nav = BoxServo()
            nav.queue_tuning(1, dict(asdict(BoxTuning()), first_gap=threshold, side_speed=10, side_step=2.))
            nav.phase = 'along'
            nav.frame = view(8., (.49, .47))  # body aligned over the opposite boundary
            cmd = nav.command(8.01, 0.)
            self.assertEqual(nav.steps.active['axis'], expected_axis)
            self.assertAlmostEqual(nav.panel_status(8.01)['gap'], 40.)
            if expected_axis == 1:
                self.assertAlmostEqual(cmd[1], .10)
                self.assertAlmostEqual(nav.steps.active['distance'], .02)

    def test_stale_edits_and_invalid_values_preserve_last_request(self):
        initial = asdict(BoxTuning())
        for change in [dict(side_step=0), dict(side_speed=float('nan')), dict(first_gap=40.2)]:
            self.assertEqual(self.request('/api/settings', dict(base=0, values=dict(initial, **change)))[0], 400)
        self.assertEqual(self.request('/api/state', token='expired')[0], 403)
        self.assertEqual(self.request('/api/settings', dict(base=0, values=initial), token='expired')[0], 403)
        values = dict(initial, forward_step=5)
        self.assertEqual(self.request('/api/settings', dict(base=0, values=values))[0], 200)
        self.assertEqual(self.request('/api/settings', dict(base=0, values=initial))[0], 409)
        self.mail.update(self.nav, 0)
        data = self.request('/api/state')[1]
        self.assertEqual(data['desired'], values)
        self.assertEqual(data['control']['applied'], values)
        self.assertEqual(self.request('/api/settings', dict(base=1, values=initial))[0], 200)
        self.mail.update(self.nav, 1)
        self.assertEqual(self.request('/api/state')[1]['control']['applied'], initial)
