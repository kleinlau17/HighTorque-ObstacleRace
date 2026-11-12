"""Exercise in-place observation through the same callbacks as the robot."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from test_rt11_box_steps import BoxServo, view
from deploy.shared_navigation import PublishedPlan, TurnCutoff
from navigation.box_servo import wrap


def missing_line(stamp):
    return replace(view(stamp,(.6,.1)),identified=(),linked=(),edges=(None,None),
                   previous_edges=(None,None))


class ActiveObservationTest(unittest.TestCase):
    def start_search(self,nav,mkframe,yaw=0.):
        for stamp in (1.,1.3,1.6):
            nav.frame=mkframe(stamp)
            np.testing.assert_equal(nav.command(stamp+.02,yaw),np.zeros(3))
        nav.frame=mkframe(1.95)
        turn=nav.command(1.97,yaw)
        np.testing.assert_equal(turn[:2],0.)
        self.assertGreaterEqual(abs(turn[2]),.70)
        self.assertEqual(nav.hold_reason,'scan_box_area')
        nav.steps.published(1.98,turn)
        return float(np.sign(turn[2]))

    def test_all_phases_seek_a_new_view(self):
        for phase in ('enter','along','shift','lane_entry'):
            with self.subTest(phase=phase):
                nav=BoxServo()
                nav.phase=phase
                nav.index=2 if phase=='lane_entry' else 0
                self.start_search(nav,missing_line)

    def test_unknown_side_and_missing_next_crate_seek_a_new_view(self):
        nav=BoxServo()
        self.start_search(nav,lambda t: replace(view(t,(.6,.1)),side=0.))
        nav=BoxServo()
        nav.phase='along'
        self.start_search(nav,lambda t: replace(view(t,(1.36,.47)),centers=(np.array([1.,.05]),)))

    def test_rgb_alone_permits_a_search_but_loss_of_both_streams_stops_it(self):
        nav=BoxServo()
        self.start_search(nav,lambda t: replace(missing_line(t),depth_diagnostics={'reason':'no_ground_plane'}))
        camera=SimpleNamespace(ground=SimpleNamespace(result=SimpleNamespace(valid=False)),
                               rays=np.zeros((12,16,3)),rotation32=np.eye(3),position32=np.zeros(3))
        def capture(stamp,rgb_level):
            raw=SimpleNamespace(depth=np.full((12,16),np.nan),time=stamp,
                                rgb=np.full((12,16,3),rgb_level,dtype=np.uint8))
            nav.observe(raw,np.eye(3),camera,None,0.,None)
            nav.commit_frame()
            return nav.frame
        nav=BoxServo()
        nav.frame=missing_line(.8)
        self.start_search(nav,lambda t:capture(t,110))
        self.assertEqual(nav.frame.depth_diagnostics['reason'],'no_depth_returns')
        for stamp in (2.,3.,4.):
            capture(stamp,0)
            self.assertEqual(nav.frame.depth_diagnostics['reason'],'no_usable_rgbd')
            cmd=nav.command(stamp+.01,0.)
            np.testing.assert_equal(cmd,0.)
            nav.steps.published(stamp+.02,cmd)

    def test_turn_stops_and_new_evidence_can_take_over(self):
        nav=BoxServo()
        direction=self.start_search(nav,missing_line)
        turned=direction*.16
        self.assertFalse(np.any(nav.command(2.,turned)))
        nav.frame=view(2.02,(.6,.1))
        self.assertFalse(np.any(nav.command(2.03,turned)))
        nav.steps.published(2.04,np.zeros(3))
        self.assertFalse(np.any(nav.command(2.06,turned)))
        # Alignment still needs two post-zero observations.
        nav.frame=replace(view(2.09,(.6,.1)),previous_stamp=2.01)
        self.assertFalse(np.any(nav.command(2.10,turned)))
        nav.frame=replace(view(2.22,(.6,.1)),previous_stamp=2.09)
        self.assertGreater(nav.command(2.23,turned)[1],0.)
        self.assertEqual(nav.hold_reason,'lateral_step')

    def test_specific_search_direction_takes_over_from_generic_sweep(self):
        nav=BoxServo()
        self.assertGreater(self.start_search(nav,missing_line),0.)
        def measured_box(t):
            return replace(missing_line(t),identified=(0,),linked=(0,),side=-1.)
        nav.frame=measured_box(2.03)
        self.assertFalse(np.any(nav.command(2.04,0.)))  # stop before reversing
        nav.steps.published(2.05,np.zeros(3))
        nav.frame=measured_box(2.20)
        self.assertLess(nav.command(2.21,0.)[2],0.)
        self.assertEqual(nav.hold_reason,'look_for_opposite_line')
        self.assertIsNone(nav.observation_since)

    def test_one_camera_capture_or_expired_frame_cannot_start_search(self):
        nav=BoxServo()
        nav.frame=missing_line(1.)
        for now in np.arange(1.,3.,.02):
            self.assertFalse(np.any(nav.command(now,0.)))
        self.assertEqual(nav.steps.count,0)

    def test_delayed_capture_can_finish_its_turn_but_cannot_start_another(self):
        nav=BoxServo()
        for stamp in (1.,1.3,1.6):
            nav.frame=missing_line(stamp)
            nav.command(stamp+.44,0.)
        nav.frame=missing_line(2.)
        hook=nav.publication(nav.frame,2.80)
        imu=np.eye(3)
        initial=hook(2.44,np.zeros(3),imu)
        self.assertGreater(abs(initial[2]),.7)
        hook.published(2.45,initial)
        for now in (2.52,2.60,2.70):
            self.assertEqual(hook(now,np.zeros(3),imu)[2],initial[2])
        # The publisher enforces the capture lease even if the chassis has
        # barely turned. Repeated publication cannot rearm the same frame.
        self.assertFalse(np.any(hook(2.80,np.zeros(3),imu)))
        self.assertFalse(np.any(nav.command(3.40,0.)))
        hook.published(3.41,np.zeros(3))
        self.assertFalse(np.any(nav.command(3.42,0.)))
        self.assertEqual(nav.steps.count,1)

    def test_search_sweeps_both_directions_and_does_not_walk_away(self):
        nav=BoxServo()
        start=3.13  # crosses the IMU's +/-pi wrap on its first pulse
        heading=start
        signs=set()
        grants=[]
        turn_samples=0
        max_angle=0.
        for tick in range(901):
            now=1.+tick*.02
            if tick%10==0:
                nav.frame=replace(missing_line(now),heading=start)
            cmd=nav.command(now,wrap(heading))
            nav.steps.published(now,cmd)
            np.testing.assert_equal(cmd[:2],0.)
            if cmd[2]:
                signs.add(np.sign(cmd[2]))
                turn_samples+=1
            elif turn_samples:
                grants.append(turn_samples*.02)
                turn_samples=0
            heading+=cmd[2]*.02
            max_angle=max(max_angle,abs(wrap(heading-start)))
        self.assertEqual(signs,{-1.,1.})
        self.assertGreater(len(grants),6)
        self.assertLess(max_angle,.27)
        self.assertGreater(max(grants),.17)

    def test_search_reverses_if_real_gait_keeps_unwinding_its_turn(self):
        nav=BoxServo()
        signs=set()
        for tick in range(350):
            now=1.+tick*.02
            if tick%10==0:
                nav.frame=missing_line(now)
            cmd=nav.command(now,0.)
            nav.steps.published(now,cmd)
            self.assertFalse(np.any(cmd[:2]))
            if cmd[2]:
                signs.add(np.sign(cmd[2]))
        self.assertEqual(signs,{-1.,1.})

    def test_real_publication_rejects_expired_frames_and_missing_or_stale_imu(self):
        nav=BoxServo()
        self.start_search(nav,missing_line)
        hook=nav.publication(nav.frame,2.5)
        plan=PublishedPlan(box_servo=hook,deadline=2.5)
        def evaluate(t,imu):
            return plan.evaluate(t,imu,np.eye(3),TurnCutoff())[0]
        unit=[0.,0.,0.,1.]
        self.assertFalse(np.any(evaluate(2.,None)))
        self.assertFalse(np.any(evaluate(2.,(1.7,unit))))
        self.assertGreater(abs(evaluate(2.,(1.99,unit))[2]),.7)
        self.assertFalse(np.any(evaluate(2.5,(2.49,unit))))
        nav.frame=missing_line(2.12)
        self.assertFalse(np.any(evaluate(2.13,(2.12,unit))))

    def test_lane_entry_advances_on_confirmed_depth_then_returns_control(self):
        nav=BoxServo()
        nav.index=2
        nav.phase='lane_entry'
        nav.frame=replace(view(1.,(2.20,.22)),depth_status='depth_unobserved',forward_room=0.)
        self.assertFalse(np.any(nav.command(1.,0.)))
        nav.frame=view(1.2,(2.20,.22))
        forward=nav.command(1.21,0.)
        self.assertGreater(forward[0],.1)
        self.assertFalse(nav.done)
        nav.steps.published(1.22,forward)
        nav.frame=view(1.5,(2.31,.22))
        self.assertFalse(np.any(nav.command(1.51,0.)))
        nav.steps.published(1.52,np.zeros(3))
        nav.frame=view(1.75,(2.31,.22))
        self.assertIsNone(nav.command(1.76,0.))
        self.assertTrue(nav.done)


if __name__=='__main__':
    unittest.main(verbosity=2)
