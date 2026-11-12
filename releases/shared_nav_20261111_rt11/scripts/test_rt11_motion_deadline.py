"""The 50 Hz motor publisher owns the time limit on a navigation stop."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation.command import AMPCommandAdapter
from navigation.motion_deadline import MotionDeadline


class MotionDeadlineTest(unittest.TestCase):
    def test_missing_lane_or_box_cannot_leave_guard_stopped(self):
        adapter=AMPCommandAdapter()
        deadline=MotionDeadline()
        zero_since=0.
        longest=0.
        starts=[]
        recovering=False
        for tick in range(600):
            t=tick*.02
            raw=adapter.update([0.,0.,0.],enabled=True)
            actual=deadline.output(t,raw,True,-1.)
            deadline.published(t,actual)
            if np.any(actual):
                self.assertEqual(actual[0],0.)
                self.assertEqual(actual[1],0.)
                self.assertGreater(abs(actual[2])*.30,.17)  # factory filter and yaw deadband
                longest=max(longest,t-zero_since)
                zero_since=t
                if not recovering:starts.append(float(np.sign(actual[2])))
            recovering=deadline.recovering
        self.assertGreater(len(starts),4)
        self.assertEqual(starts[:4],[-1.,1.,-1.,1.])
        self.assertLess(longest,2.)

    def test_planning_and_unsent_command_do_not_reset_clock(self):
        d=MotionDeadline()
        d.output(12.,[0.,0.,0.],True)
        for t in (12.8,13.2,13.7):
            d.output(t,[.2,0.,0.],True)
            d.published(t,[0.,0.,0.])
        self.assertNotEqual(d.output(13.90,[0.,0.,0.],True)[2],0.)

    def test_line_motion_preempts_recovery_and_real_nonzero_restarts_clock(self):
        d=MotionDeadline()
        d.output(100.,[0.,0.,0.],True)
        scan=d.output(101.90,[0.,0.,0.],True)
        d.published(101.90,scan)
        forward=[.25,.06,-.1]
        np.testing.assert_array_equal(d.output(101.92,forward,True),forward)
        self.assertFalse(d.recovering)
        d.published(101.93,forward)
        d.published(101.95,[0.,0.,0.])
        np.testing.assert_array_equal(d.output(103.79,[0.,0.,0.],True),np.zeros(3))
        self.assertTrue(d.output(103.85,[0.,0.,0.],True)[2])

    def test_startup_and_shutdown_never_wake_motors(self):
        d=MotionDeadline()
        for t in (0.,20.,60.):
            np.testing.assert_array_equal(d.output(t,np.zeros(3),False),np.zeros(3))
        d.output(62.,np.zeros(3),True)
        self.assertTrue(d.output(63.90,np.zeros(3),True)[2])
        for t in (63.91,64.,80.):
            np.testing.assert_array_equal(d.output(t,np.zeros(3),False),np.zeros(3))
            self.assertFalse(d.recovering)

    def test_panel_shows_actual_guard_turn_during_perception_wait(self):
        from navigation.box_servo import BoxServo
        nav=BoxServo()
        nav.guard_publication(2.,'navigation_recovery')
        self.assertEqual(nav.panel_status(2.01)['gate'],'scan_box_area')
        self.assertTrue(nav.panel_status(2.01)['moving'])
        nav.guard_publication(2.02,'stopped')
        self.assertFalse(nav.panel_status(2.03)['moving'])

    def test_twist_and_guard_receipt_at_actual_proxy_entry_point(self):
        from test_line_supervisor import ProxyHarness, ROOT
        transport=ProxyHarness(ROOT/'deploy/pro_model_config.json')
        token=transport.args.token
        def exercise(guard):
            transport.send_navigation((0.,0.,0.),token=token+'/active_right')
            self.assertEqual(transport.tick(guard)['source'],'stopped')
            transport.now+=1.90
            receipt=transport.tick(guard)
            self.assertEqual(receipt['source'],'navigation_recovery')
            self.assertEqual(receipt['command'],[0.,0.,-.8])
            self.assertEqual(receipt['emitted_s'],transport.now)
            transport.send_navigation((.22,0.,0.),token=token+'/active_right')
            transport.now+=.02
            receipt=transport.tick(guard)
            self.assertGreater(receipt['command'][0],0.)
            self.assertEqual(receipt['command'][2],0.)
            transport.send_navigation((0.,0.,0.),token=token)
            transport.now+=2.
            self.assertEqual(transport.tick(guard)['source'],'stopped')
        transport.run(exercise)
