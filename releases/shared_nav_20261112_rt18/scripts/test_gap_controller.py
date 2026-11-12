"""Observed-gap control regressions; no simulator pose or AMP modification."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation.gap_controller import GapController, rotation


class GapTests(unittest.TestCase):
    @staticmethod
    def observation():
        return SimpleNamespace(lane=np.array([0.,0.,.24]),
            obstacles=np.array([[-.15,.39,.2,.2],[.35,.09,.2,.2]]))

    def test_alternating_gap_brakes_before_crossing_and_waits_for_motion(self):
        control=GapController();pose=np.zeros(3);p=self.observation()
        np.testing.assert_array_equal(control.update(0,p,pose,pose,np.array([.22,0.])),0)
        self.assertEqual(control.phase,'brake')
        np.testing.assert_array_equal(control.update(.6,p,pose,pose,np.array([.15,0.])),0)
        self.assertEqual(control.phase,'brake')
        command=control.update(.7,p,pose,pose,np.zeros(2))
        self.assertEqual(control.phase,'shift')
        self.assertGreater(command[1],0)

    def test_same_side_and_closed_gaps_do_not_start_a_crossing(self):
        for front in ([.35,.39,.2,.2],[.30,.09,.2,.2]):
            p=self.observation();p.obstacles[1]=front
            self.assertIsNone(GapController().update(0,p,np.zeros(3),np.zeros(3),np.zeros(2)))

    def test_completed_gap_does_not_retrigger_on_the_same_observation(self):
        c=GapController();p=self.observation();pose=np.zeros(3)
        c.update(0,p,pose,pose,np.zeros(2))
        c.update(.6,p,pose,pose,np.zeros(2))
        arrived=np.r_[c.target,0.]
        np.testing.assert_array_equal(c.update(1.2,p,pose,arrived,np.zeros(2)),0)
        self.assertEqual(c.phase,'idle')
        self.assertIsNone(c.update(1.3,p,pose,arrived,np.zeros(2)))
        # A fresh camera observation after crossing changes all body-relative
        # lateral coordinates, but must still identify the same static gap.
        shifted=self.observation();shifted.obstacles[:,:2]-=arrived[:2]
        shifted.lane[-1]-=arrived[1]
        self.assertIsNone(c.update(1.4,shifted,arrived,arrived,np.zeros(2)))

    def test_crossing_control_rotates_with_observed_heading(self):
        p=self.observation();local=GapController();rotated=GapController()
        origin=np.array([2.,3.]);yaw=1.1;pose=np.r_[origin,yaw]
        local.update(0,p,np.zeros(3),np.zeros(3),np.zeros(2))
        rotated.update(0,p,pose,pose,np.zeros(2))
        np.testing.assert_allclose(rotated.target,origin+local.target@rotation(yaw).T)
        a=local.update(.6,p,np.zeros(3),np.zeros(3),np.zeros(2))
        b=rotated.update(.6,p,pose,pose,np.zeros(2))
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_measured_forward_velocity_brakes_before_position_overshoot(self):
        c=GapController();c.phase='shift';c.target=np.array([.02,.3]);c.started=0.
        command=c.update(.6,self.observation(),np.zeros(3),np.zeros(3),np.array([.12,0.]))
        self.assertLess(command[0],0)

    def test_stalled_shift_returns_to_standing_then_resumes_visual_servo(self):
        c=GapController();p=self.observation();pose=np.zeros(3)
        c.update(0,p,pose,pose,np.zeros(2))
        c.update(.6,p,pose,pose,np.zeros(2))
        np.testing.assert_array_equal(c.update(1.3,p,pose,pose,np.zeros(2)),0)
        self.assertEqual(c.phase,'settle')
        np.testing.assert_array_equal(c.update(1.7,p,pose,pose,np.zeros(2)),0)
        self.assertGreater(c.update(1.81,p,pose,pose,np.zeros(2))[1],0)
        self.assertEqual(c.phase,'shift')
        self.assertEqual(c.recoveries,1)

    def test_unresolved_stall_has_bounded_retries(self):
        c=GapController();p=self.observation();pose=np.zeros(3)
        c.update(0,p,pose,pose,np.zeros(2))
        for time in np.arange(.6,6.,.1):c.update(time,p,pose,pose,np.zeros(2))
        self.assertEqual(c.phase,'failed')
        np.testing.assert_array_equal(c.update(8,p,pose,pose,np.zeros(2)),0)
        self.assertEqual(c.recoveries,3)

    def test_map_correction_keeps_gap_identity_and_target_with_boxes(self):
        c=GapController();p=self.observation();pose=np.zeros(3)
        c.update(0,p,pose,pose,np.zeros(2))
        offset=c.target-c.identity
        old=c.target.copy();translation=np.array([-.05,.02])
        c.translate_map(translation)
        np.testing.assert_allclose(c.target-c.identity,offset)
        np.testing.assert_allclose(c.target,old+translation)

    def test_crossing_uses_corridor_axis_when_body_yaw_is_offset(self):
        yaw=.06;pose=np.array([0.,0.,yaw])
        world=np.array([[-.15,.15,.2,.2],[.35,-.15,.2,.2]])
        boxes=world.copy();boxes[:,:2]=world[:,:2]@rotation(yaw)
        p=SimpleNamespace(lane=np.array([0.,-np.tan(yaw),0.]),obstacles=boxes)
        c=GapController();c.update(0,p,pose,pose,np.zeros(2),corridor_heading=0.)
        self.assertEqual(c.phase,'brake')
        self.assertAlmostEqual(c.heading,0.)
        self.assertAlmostEqual(c.target[0],.10,delta=1e-12)


if __name__=='__main__':unittest.main()
