"""Box-step transitions from time-stamped line and depth measurements."""
from dataclasses import replace
import unittest
import numpy as np

from test_rt11_box_steps import BoxServo, view, edges_at, stationary_limbs


class BoundarySteps(unittest.TestCase):
    def stopped_lateral(self):
        nav=BoxServo()
        nav.frame=view(1.,(.6,.14))
        step=nav.command(1.01,0.)
        self.assertGreater(step[1],.1)
        nav.publication(nav.frame,2.).published(1.02,step)
        np.testing.assert_equal(nav.command(1.40,0.),0.)
        return nav

    def test_stationary_robot_can_repeat_lateral_and_forward_steps(self):
        for axis,xy in [(1,(.60,.14)),(0,(.60,.47))]:
            with self.subTest(axis=axis):
                nav=BoxServo()
                if axis==0:
                    nav.phase='along'
                now=1.
                for number in range(1,6):
                    nav.frame=view(now,xy)
                    vector=nav.command(now+.01,0.)
                    self.assertGreater(vector[axis],.1)
                    self.assertEqual(nav.steps.count,number)
                    move=nav.steps.active
                    self.assertEqual(move['retry'],number>1)
                    self.assertLessEqual(move['distance'],.046 if axis==1 else .063)
                    # Include the planner-to-publisher delay in each cycle.
                    sent=now+.10
                    nav.steps.published(sent,vector)
                    duration=move['duration']
                    start=sent if number>1 else now+.01
                    self.assertAlmostEqual(move['until'],start+duration)
                    if number>1:
                        self.assertGreater(duration,.40)
                    during=nav.command(move['until']-.01,0.)
                    self.assertGreater(during[axis],.1)
                    stop=move['until']+.01
                    np.testing.assert_equal(nav.command(stop,0.),0.)
                    nav.steps.published(stop+.01,np.zeros(3))
                    np.testing.assert_equal(nav.command(stop+.02,0.),0.)
                    self.assertEqual(nav.steps.count,number)
                    now=stop+.10

    def test_retry_uses_present_position_after_incoming_gait_creeps_forward(self):
        nav=self.stopped_lateral()
        nav.steps.published(1.41,np.zeros(3))
        nav.frame=view(1.60,(.80,.10))
        vector=nav.command(2.04,0.)
        self.assertGreater(vector[1],.1)
        self.assertAlmostEqual(nav.entry_station,.80)
        self.assertAlmostEqual(vector[0],-.055)
        nav.steps.published(2.05,vector)
        # The capture was acceptable when the step started. The age of the
        # same frame does not reduce its usable publisher time to 0.06 s.
        self.assertGreater(nav.command(2.26,0.)[1],.1)
        self.assertTrue(nav.panel_status(2.26)['moving'])
        nav.steps.published(2.27,np.zeros(3))
        self.assertFalse(nav.panel_status(2.27)['moving'])
        # Fore/aft creep cannot consume a lateral displacement allowance.
        nav.frame=view(2.35,(.88,.10))
        self.assertGreater(nav.command(2.36,0.)[1],.1)
        nav.frame=view(2.40,(.88,.16))
        np.testing.assert_equal(nav.command(2.41,0.),0.)

    def test_borderline_alignment_on_direction_change_confirms_one_extra_frame(self):
        nav=self.stopped_lateral()
        nav.steps.published(1.41,np.zeros(3))
        nav.phase='along'
        nav.boundary_lock=(0,1.,.40-nav.tuning.first_gap/100.)
        nav.frame=view(1.6,(.60,.43))
        np.testing.assert_equal(nav.command(1.61,0.),0.)
        np.testing.assert_equal(nav.command(1.64,0.),0.)
        nav.frame=view(1.80,(.60,.43))
        self.assertGreater(nav.command(1.81,0.)[0],.1)

    def test_replanned_step_needs_two_post_zero_stripe_observations(self):
        for changes in [dict(previous_edges=(None,None)),dict(previous_stamp=1.30),
                        dict(previous_stamp=2.)]:
            with self.subTest(changes=changes):
                nav=self.stopped_lateral()
                nav.steps.published(1.41,np.zeros(3))
                nav.frame=replace(view(2.,(.6,.10)),**changes)
                np.testing.assert_equal(nav.command(2.01,0.),0.)
                self.assertEqual(nav.steps.count,1)

    def test_single_publication_cannot_prove_line_alignment(self):
        nav=BoxServo()
        nav.phase='along'
        nav.frame=replace(view(1.,(.5,.47)),previous_edges=(None,None))
        for now in (1.,1.02,1.04):
            np.testing.assert_equal(nav.command(now,0.),0.)
            self.assertEqual(nav.hold_reason,'confirm_opposite_line')
        nav.frame=view(1.16,(.5,.47))
        self.assertGreater(nav.command(1.17,0.)[0],0.)

    def test_scalar_box_gap_alone_cannot_release_forward(self):
        nav=BoxServo()
        nav.phase='along'
        # Only the stripe beside the box; current box coordinates alone are insufficient.
        nav.frame=replace(view(1.,(.5,.47)),edges=(edges_at(.47)[0],None))
        np.testing.assert_equal(nav.command(1.,0.)[:2],0.)
        self.assertIn(nav.hold_reason,('wait_opposite_line','look_for_opposite_line'))

    def test_unknown_and_blocked_depth_do_not_grant_forward(self):
        for room,reason in [(0.,'depth_unobserved'),(.025,'depth_blocked')]:
            nav=BoxServo()
            nav.phase='along'
            nav.frame=replace(view(2.,(.6,.47)),forward_room=room,depth_status=reason)
            np.testing.assert_equal(nav.command(2.02,0.),0.)
            self.assertEqual(nav.hold_reason,reason)
            self.assertEqual(nav.steps.count,0)

    def test_post_lateral_release_needs_zero_and_two_new_captures(self):
        nav=BoxServo()
        nav.phase='along'
        nav.frame=view(4.,(.6,.41))
        side=nav.command(4.01,0.)
        self.assertGreater(side[1],0.)
        nav.steps.published(4.02,side)
        nav.frame=view(4.21,(.6,.47))
        np.testing.assert_equal(nav.command(4.22,0.),0.)
        self.assertEqual(nav.hold_reason,'await_zero_publication')
        nav.steps.published(4.23,np.zeros(3))
        nav.frame=replace(view(4.26,(.6,.47)),previous_stamp=4.21)
        np.testing.assert_equal(nav.command(4.27,0.),0.)
        nav.frame=replace(view(4.42,(.6,.47)),previous_stamp=4.26)
        self.assertGreater(nav.command(4.43,0.)[0],0.)

    def test_stage_switch_uses_other_measured_boundary(self):
        nav=BoxServo()
        nav.phase='shift'
        stationary_limbs(nav,3.)
        nav.frame=view(3.,(1.23,.44))
        self.assertLess(nav.command(3.01,0.)[1],0.)
        self.assertEqual(nav.index,0)
        nav.steps.published(3.02,nav.last_cmd)
        nav.frame=view(3.40,(1.23,-.02))
        np.testing.assert_equal(nav.command(3.41,0.),0.)
        nav.steps.published(3.42,np.zeros(3))
        nav.frame=view(3.56,(1.23,-.02))
        np.testing.assert_equal(nav.command(3.57,0.),0.)
        self.assertEqual((nav.index,nav.phase),(1,'along'))

    def test_occluded_target_can_only_maintain_confirmed_forward_alignment(self):
        for use_box_face in (False,True):
            nav=BoxServo()
            nav.phase='along'
            sample=view(1.,(.5,.47))
            left=sample.edges[1]
            def occluded(stamp,other,registered):
                return replace(view(stamp,(.5,.47)),edges=(other,None),previous_edges=(other,None),
                               registered_edges=(None,registered),previous_registered=(None,registered))
            nav.frame=occluded(1.,None if use_box_face else sample.edges[0],left if use_box_face else None)
            self.assertFalse(np.any(nav.command(1.01,0.)[:2]))
            nav.frame=sample
            self.assertGreater(nav.command(1.02,0.)[0],0.)
            nav.frame=occluded(1.1,None if use_box_face else sample.edges[0],left if use_box_face else None)
            self.assertGreater(nav.command(1.11,0.)[0],0.)
            drifted=edges_at(.35)
            nav.frame=occluded(1.17,None if use_box_face else drifted[0],drifted[1] if use_box_face else None)
            self.assertFalse(np.any(nav.command(1.18,0.)[:2]))

    def test_proven_partial_clearance_limits_both_distance_and_duration(self):
        nav=BoxServo()
        nav.phase='along'
        nav.frame=replace(view(1.,(.5,.47)),forward_room=.045,depth_status='depth_limited')
        self.assertGreater(nav.command(1.,0.)[0],0.)
        self.assertAlmostEqual(nav.steps.active['distance'],.03)
        self.assertLess(nav.steps.active['until'],1.16)

    def test_target_sighting_during_a_turn_expires_after_one_side_step(self):
        nav=BoxServo()
        nav.phase='shift'
        stationary_limbs(nav,1.)
        sample=view(1.,(1.22,.32))
        wanted=sample.edges[0]
        nav.frame=replace(sample,edges=(None,None),previous_edges=(None,None),
                          registered_edges=(wanted,None),previous_registered=(wanted,None),
                          paint_stamp=(.92,-np.inf))
        motion=nav.command(1.01,0.)
        self.assertLess(motion[1],0.)
        nav.steps.published(1.02,motion)
        nav.steps.stop(1.19)
        nav.steps.published(1.20,np.zeros(3))
        stationary_limbs(nav,1.45)
        nav.frame=replace(nav.frame,stamp=1.45,previous_stamp=1.28)
        self.assertFalse(np.any(nav.command(1.46,0.)[:2]))
        if nav.steps.active is not None:
            self.assertEqual(nav.steps.active['axis'],2)
