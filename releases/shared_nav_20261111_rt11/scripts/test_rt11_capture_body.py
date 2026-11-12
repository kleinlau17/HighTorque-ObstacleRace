"""The RGB-D worker must see encoder geometry at exposure despite concurrent ROS updates."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation.box_servo import BoxServo
from navigation.leg_odometry import LEG_JOINTS
from test_rt11_box_steps import view


class CaptureBodyTests(unittest.TestCase):
    def setUp(self):
        self.servo=BoxServo()
        self.joints=dict.fromkeys(LEG_JOINTS,0.)
        for side in ('l','r'):
            for link in ('shoulder_pitch','shoulder_roll','upper_arm','elbow'):
                self.joints[f'{side}_{link}_joint']=0.

    def update(self,t,arm_angle):
        self.joints['r_shoulder_pitch_joint']=arm_angle
        self.assertTrue(self.servo.update_legs(t,self.joints,np.eye(3)))

    def test_short_delay_uses_same_published_swept_geometry(self):
        for i in range(6):
            self.update(1+i*.055,i*.06)
        points,info=self.servo._capture_body(1.18)
        self.assertIs(points,self.servo.body_sample[1])
        self.assertEqual(info['body_evidence'],'capture_in_live_sweep')
        self.assertAlmostEqual(info['body_capture_age_s'],.015)

    def test_delayed_capture_retains_exposure_pose_and_current_arm(self):
        self.update(1.,1.2)
        exposure=self.servo.body_sample[1].copy()
        for i in range(1,9):
            self.update(1+i*.055,0.)
        live=self.servo.body_sample[1]
        points,info=self.servo._capture_body(1.01)
        self.assertEqual(info['body_evidence'],'capture_and_live_sweep')
        self.assertGreater(info['body_latest_offset_s'],.30)
        np.testing.assert_allclose(points[:len(exposure)],exposure)
        np.testing.assert_allclose(points[-len(live):],live)
        self.assertFalse(np.allclose(exposure[-1],live[-1]))

    def test_no_substitute_for_missing_expired_or_gapped_feedback(self):
        self.update(1.,0.)
        self.update(1.4,.2)
        for stamp,why in [(.95,'missing_capture_body'),(1.25,'stale_capture_body'),
                          (1.02,'interrupted_body_history')]:
            with self.subTest(stamp=stamp):
                points,info=self.servo._capture_body(stamp)
                self.assertIsNone(points)
                self.assertEqual(info['body_evidence'],why)
        self.update(3.,.2)
        self.assertIsNone(self.servo._capture_body(1.40)[0])

    def test_gap_shorter_than_the_live_sweep_also_rejects_capture(self):
        self.update(1.,0.)
        self.update(1.17,.2)
        points,info=self.servo._capture_body(1.02)
        self.assertIsNone(points)
        self.assertEqual(info['body_evidence'],'interrupted_body_history')

    def test_forward_requires_live_encoders_and_new_image_after_gap(self):
        self.update(1.,0.)
        self.servo.phase='along'
        self.servo.frame=replace(view(1.01,(.5,.47)),depth_diagnostics={'reason':'clear'})
        np.testing.assert_equal(self.servo.command(1.18,0.),0.)
        self.assertEqual(self.servo.hold_reason,'stale_live_body')
        self.update(1.20,0.)
        np.testing.assert_equal(self.servo.command(1.21,0.),0.)
        self.assertEqual(self.servo.hold_reason,'stale_live_body')
        self.servo.frame=replace(view(1.25,(.5,.47)),depth_diagnostics={'reason':'clear'})
        self.assertGreater(self.servo.command(1.26,0.)[0],0.)

    def test_invalid_projection_does_not_reuse_depth_details(self):
        frame=replace(view(1.,(0.,0.)),depth_diagnostics={'reason':'test_clear'})
        self.servo.frame=frame
        self.assertEqual(self.servo.panel_status(1.1)['depth_detail'],'test_clear')
        self.assertEqual(self.servo.diagnostic()['crate_depth_detail'],'test_clear')
        self.assertIsNone(self.servo.panel_status(1.8)['depth_detail'])
        self.servo.invalidate_projection(1.)
        self.assertIsNone(self.servo.panel_status(1.1)['depth_detail'])
        self.assertIsNone(self.servo.diagnostic()['crate_depth_diagnostics'])


if __name__=='__main__':
    unittest.main()
