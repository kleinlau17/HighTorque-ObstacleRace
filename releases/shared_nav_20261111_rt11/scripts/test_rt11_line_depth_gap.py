"""Line-follow regression for RT11's 043150 run and intermittent visibility."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from test_rt11_box_steps import BoxServo, view
from deploy.shared_navigation import PublishedPlan, TurnCutoff
from navigation.box_corridor import Edge, forward_clearance


class LineDepthGapTest(unittest.TestCase):
    def setUp(self):
        # 67.41 s of 043150. The recorded error and signed offset determine
        # slope magnitude. Use a consistent tangent for the two captures.
        self.stamp=2800.165286198
        self.heading=.687989376
        self.edge=Edge(-.027093243,.203,.35,.75)
        self.prior=Edge(-.027283585,.203,.35,.75)
        xy=(.400412309,-.407194781)
        self.frame=replace(view(self.stamp,xy),
            centers=(np.array([.863419578,.076194843]),np.array([1.640850362,-.597111484])),
            side=-1.,heading=self.heading,progress_x=.400440753,
            edges=(self.edge,None),previous_edges=(self.prior,None),previous_stamp=2800.007030179,
            forward_room=0.,depth_status='depth_unobserved',
            depth_diagnostics={'reason':'unseen_swept_path','body_evidence':'capture_and_live_sweep'})
        self.nav=BoxServo()
        self.nav.phase='along'
        self.nav.boundary_lock=(0,-1.,.40-self.nav.tuning.first_gap/100.)
        self.nav.aligned_box=0
        self.nav.steps.zero_stamp=2799.802286446

    def publish(self,frame,now):
        self.nav.frame=frame
        hook=self.nav.publication(frame,frame.stamp+.85)
        plan=PublishedPlan(box_servo=hook,deadline=frame.stamp+.85)
        imu=(now,[0.,0.,np.sin(self.heading/2),np.cos(self.heading/2)])
        vector,_,_=plan.evaluate(now,imu,np.eye(3),TurnCutoff())
        hook.published(now,vector)
        return np.array(vector)

    def test_first_real_frame_grants_and_publishes_forward_immediately(self):
        cmd=self.publish(self.frame,self.stamp+.34)  # measured capture and processing delay
        self.assertGreater(cmd[0],.20)
        self.assertAlmostEqual(cmd[1]/cmd[0],self.edge.slope)
        self.assertGreater(cmd[2],0.)
        self.assertEqual(self.nav.hold_reason,'line_follow_unseen_path')
        self.assertEqual(self.nav.steps.active['axis'],0)
        self.assertTrue(self.nav.steps.active['sent'])

    def test_midstep_line_dropout_continues_but_cannot_start_a_blind_step(self):
        self.assertGreater(self.publish(self.frame,self.stamp+.02)[0],.20)
        missing=replace(self.frame,stamp=self.stamp+.10,previous_stamp=self.stamp,
                        edges=(None,None))
        self.assertGreater(self.publish(missing,missing.stamp+.02)[0],.20)
        np.testing.assert_equal(self.publish(missing,self.stamp+.45),0.)
        second=replace(missing,stamp=self.stamp+.51,previous_stamp=missing.stamp)
        self.assertLessEqual(self.publish(second,second.stamp+.01)[0],0.)
        # The very next observed line can resume a previously confirmed pass.
        found=replace(self.frame,stamp=self.stamp+.65,previous_stamp=second.stamp,
                      previous_edges=(None,None))
        # If the missing frame had initiated an in-place look, acknowledge the
        # direction change and give the publisher a new capture after zero.
        resumed=self.publish(found,found.stamp+.01)
        if resumed[0]==0.:
            found=replace(found,stamp=found.stamp+.16)
            resumed=self.publish(found,found.stamp+.01)
        self.assertGreater(resumed[0],.2)

    def test_new_obstacle_stops_existing_forward_pulse(self):
        self.assertGreater(self.publish(self.frame,self.stamp+.02)[0],0.)
        blocked=replace(self.frame,stamp=self.stamp+.12,depth_status='depth_blocked',
                        depth_diagnostics={'reason':'obstacle_in_step'})
        np.testing.assert_equal(self.publish(blocked,blocked.stamp+.01),0.)
        self.assertIsNone(self.nav.steps.active)

    def test_measured_obstacle_with_incomplete_sweep_keeps_obstacle_status(self):
        # Far ground is measured in all five bands. No rays cover the near
        # sweep. Adding a partly seen object shortens a 6.3 cm step to 4 cm.
        shape=(16,32)
        points=np.zeros((*shape,3))
        points[...,0]=.60
        points[...,1]=np.linspace(-.19,.19,shape[1])
        heights=np.zeros(shape)
        depth=np.full(shape,np.nan)
        camera=SimpleNamespace(position32=np.zeros(3),rotation32=np.eye(3),
                               intrinsics=(50.,50.,16.,8.))
        plane=SimpleNamespace(valid=True,normal=np.array([0.,0.,1.]),offset=0.)
        body=np.array([[.10,.10,0.,.01]])
        def reason():
            details={}
            room,status=forward_clearance(points,heights,np.ones(shape,bool),body,depth,
                camera,np.eye(3),np.eye(2),plane,diagnostics=details)
            return room,status,details['reason']
        self.assertEqual(reason(),(0.,'depth_unobserved','unseen_swept_path'))
        points[5:10,12:16,0]=.405
        heights[5:10,12:16]=.12
        self.assertEqual(reason(),(0.,'depth_blocked','obstacle_in_step'))

    def test_depth_cannot_substitute_for_lateral_alignment(self):
        bad=Edge(-.105,.203,.35,.75)
        displaced=replace(self.frame,edges=(bad,None),previous_edges=(bad,None))
        self.publish(displaced,displaced.stamp+.02)
        self.assertEqual(self.nav.hold_reason,'lateral_step')
        self.assertEqual(self.nav.steps.active['axis'],1)

    def test_short_blind_step_retries_without_expanding_its_distance(self):
        xy=np.array([.92,-.41])  # near the next box-to-box crossing
        base=replace(self.frame,position=xy,progress_x=xy[0])
        self.assertGreater(self.publish(base,self.stamp+.02)[0],.15)
        first=self.nav.steps.active
        self.assertAlmostEqual(first['distance'],.03)
        self.assertLess(first['duration'],.18)
        np.testing.assert_equal(self.publish(base,self.stamp+.22),0.)
        fresh=replace(base,stamp=self.stamp+.46,previous_stamp=self.stamp+.33)
        self.assertGreater(self.publish(fresh,fresh.stamp+.01)[0],.15)
        repeat=self.nav.steps.active
        self.assertTrue(repeat['retry'])
        self.assertAlmostEqual(repeat['distance'],first['distance'])
        self.assertGreater(repeat['duration'],.5)
        self.assertGreater(self.publish(fresh,fresh.stamp+.28)[0],.15)
        # A measured range cap takes effect even when it equals the existing
        # soft 3 cm target. It cannot inherit the longer blind-retry window.
        clear=replace(fresh,stamp=fresh.stamp+.30,previous_stamp=fresh.stamp,
                      depth_status='depth_limited',forward_room=.045,
                      depth_diagnostics={'reason':'measured_clearance'})
        self.nav.body_sample=(clear.stamp,np.array([[.10,.10,0.,.01]]))
        np.testing.assert_equal(self.publish(clear,clear.stamp+.01),0.)
        self.assertIsNone(self.nav.steps.active)


if __name__=='__main__':
    unittest.main(verbosity=2)
