from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.box_servo import BoxFrame, BoxServo


from navigation.box_corridor import Edge


def edges_at(y):
    return tuple(Edge(line-y,0.,.35,.75) for line in (-.03,.47))


def stationary_limbs(nav,stamp):
    # Current calibrated extremities in a stationary crossing fixture.
    nav.body_sample=(stamp,np.array([[.10,.10,-.23,.01],[-.10,-.10,-.23,.01]]))


def view(stamp, xy):
    return BoxFrame(position=np.array(xy), velocity=np.zeros(2),
                    centers=tuple(np.array(c) for c in [(1.,.05),(1.50,-.22),(2.,.07)]),
                    heading=0., side=1., stamp=stamp, leg_anchor=None, kick_until=0.,
                    identified=(0,1), axis_evidence=(True,True), direct_axes=((0,1),(0,1)),
                    linked=(0,1), visual_xy=tuple(xy), edges=edges_at(xy[1]),
                    previous_edges=edges_at(xy[1]), previous_stamp=stamp-.03,
                    forward_room=.12, depth_status='depth_clear', progress_x=xy[0])


class StepGateTest(unittest.TestCase):
    def test_standoff_target_cannot_finish_gap_station(self):
        nav=BoxServo()
        nav.frame=view(10.,(.64,.47))
        np.testing.assert_equal(nav.command(10.,0.),np.zeros(3))
        self.assertEqual(nav.phase,'along')
        np.testing.assert_equal(nav.command(10.015,0.),np.zeros(3))
        nav.publication(nav.frame,11.).published(10.02,np.zeros(3))
        nav.frame=view(10.22,(.64,.47))
        self.assertGreater(nav.command(10.24,0.)[0],0.)
        self.assertEqual(nav.phase,'along')
        self.assertGreater(nav.error[0],.5)

    def test_short_step_needs_actual_visual_displacement(self):
        nav=BoxServo()
        nav.phase='shift'
        stationary_limbs(nav,2.)
        nav.frame=view(2.,(1.21,.40))
        move=nav.command(2.,0.)
        self.assertLess(move[1],-.1)
        nav.publication(nav.frame,3.).published(2.01,move)
        np.testing.assert_equal(nav.command(2.39,0.),np.zeros(3))
        nav.frame=replace(nav.frame,stamp=2.45,position=np.array([1.21,.33]))
        np.testing.assert_equal(nav.command(2.50,0.),np.zeros(3))
        self.assertEqual(nav.step_count,1)
        nav.publication(nav.frame,3.).published(2.51,np.zeros(3))
        stationary_limbs(nav,2.62)
        nav.frame=view(2.62,(1.21,.38))
        self.assertLess(nav.command(2.65,0.)[1],-.1)
        self.assertEqual(nav.step_count,2)

    def test_forward_requires_zero_receipt_then_newer_capture(self):
        nav=BoxServo()
        nav.phase='along'
        nav.along_started=3.
        nav.frame=view(3.,(.6,.47))
        first=nav.command(3.02,0.)
        self.assertGreater(first[0],.1)
        old=nav.publication(nav.frame,4.)
        old.published(3.03,first)
        np.testing.assert_equal(nav.command(3.60,0.),np.zeros(3))
        nav.frame=view(3.61,(.65,.47))
        np.testing.assert_equal(nav.command(3.63,0.),np.zeros(3))
        old.published(3.64,np.zeros(3))
        np.testing.assert_equal(nav.command(3.65,0.),np.zeros(3))
        nav.frame=view(3.70,(.65,.47))
        self.assertGreater(nav.command(3.72,0.)[0],.1)

    def test_both_side_directions_require_real_zero_and_after_stop_image(self):
        for direction in (-1.,1.):
            with self.subTest(direction=direction):
                nav=BoxServo()
                nav.phase='shift'
                start=.22-.15*direction
                stationary_limbs(nav,5.)
                nav.frame=replace(view(5.,(1.21,start)),side=-direction)
                pulse=nav.command(5.01,0.)
                self.assertGreater(direction*pulse[1],.1)
                hook=nav.publication(nav.frame,6.)
                hook.published(5.02,pulse)
                np.testing.assert_equal(nav.command(5.41,0.),np.zeros(3))
                after=start+.02*direction
                nav.frame=replace(nav.frame,stamp=5.43,position=np.array([1.21,after]),visual_xy=(1.21,after))
                np.testing.assert_equal(nav.command(5.45,0.),np.zeros(3))
                hook.published(5.46,np.zeros(3))
                np.testing.assert_equal(nav.command(5.47,0.),np.zeros(3))
                stationary_limbs(nav,5.51)
                nav.frame=replace(view(5.51,(1.21,after)),side=-direction)
                self.assertGreater(direction*nav.command(5.54,0.)[1],.1)

    def test_forward_loss_stops_before_the_grant_deadline(self):
        nav=BoxServo()
        nav.phase='along'
        nav.along_started=7.
        nav.frame=view(7.,(.6,.47))
        pulse=nav.command(7.01,0.)
        self.assertGreater(pulse[0],.1)
        nav.publication(nav.frame,8.).published(7.02,pulse)
        nav.frame=replace(nav.frame,stamp=7.10,forward_room=0.,depth_status='depth_unobserved',identified=(),axis_evidence=(False,False),
                          direct_axes=((),()),linked=(),visual_xy=(None,None))
        stopped=nav.command(7.11,0.)
        np.testing.assert_equal(stopped,np.zeros(3))
        nav.publication(nav.frame,8.).published(7.12,stopped)
        np.testing.assert_equal(nav.command(7.13,0.),np.zeros(3))

    def test_lock_contention_or_old_plan_must_emit_zero(self):
        nav=BoxServo()
        stationary_limbs(nav,2.)
        nav.frame=view(2.,(.5,0.))
        old=nav.publication(nav.frame,3.)
        with nav.control_lock:
            np.testing.assert_equal(old(2.1,np.ones(3),np.eye(3)),np.zeros(3))
        nav.pending_frame=replace(nav.frame,stamp=2.12)
        nav.commit_frame()
        np.testing.assert_equal(old(2.15,np.ones(3),np.eye(3)),np.zeros(3))

    def test_unmatched_frame_stops_even_with_recent_pose_and_legs(self):
        nav=BoxServo()
        nav.frame=replace(view(4.,(.6,.4)), edges=(None,None),progress_x=None,identified=(),axis_evidence=(False,False),
                          direct_axes=((),()),linked=(),visual_xy=(None,None))
        nav.latest_leg=(4.,np.zeros(2),np.ones(2)*.3)
        np.testing.assert_equal(nav.command(4.,0.),np.zeros(3))


if __name__=='__main__':
    unittest.main(verbosity=2)
