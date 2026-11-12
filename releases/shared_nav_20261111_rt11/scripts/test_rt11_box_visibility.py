"""Exercise sensor-loss and lane-handoff behavior of the frozen candidate."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from dataclasses import replace

import numpy as np


SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from navigation.box_servo import BoxFrame, BoxServo
from test_rt11_box_steps import edges_at
from navigation.leg_odometry import LEG_JOINTS
from navigation.runtime import CameraFrame
from navigation.visual import CameraPerception


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        config=json.loads((SOURCE/'deploy/pro_model_config.json').read_text())
        self.cam=CameraPerception(160,120,60.,config['camera_position_body_m'],
                                  config['camera_rotation_optical_to_body'])
        self.rotation=np.eye(3)
        self.rgb=np.zeros((120,160,3),np.uint8)
        self.tracker=SimpleNamespace(xy=np.zeros(2),odometry=SimpleNamespace(accepted=False))
        self.nav=BoxServo()
        # Marks in a box-relative test map, previously measured by the tracker.
        self.nav.heading=0.
        self.nav.boxes=[np.array(v) for v in [(1.,-.12),(1.55,.12),(2.10,-.12)]]
        self.nav.side=1.

    def observe(self,stamp,depth):
        frame=CameraFrame(stamp,self.rgb,depth)
        observation,_=self.cam.inspect_lane(frame,self.rotation)
        self.nav.observe(frame,self.rotation,self.cam,self.tracker,0.,observation.lane)
        self.nav.commit_frame()

    def test_floor_only_then_lost_depth(self):
        directions=self.cam.rays@self.cam.rotation32.T
        depth=(-.45-self.cam.position32[2])/directions[...,2]
        depth[(depth<.15)|(depth>4.)]=np.nan
        self.observe(1.,depth)
        self.assertTrue(self.cam.ground.result.valid)
        self.assertEqual(self.nav.visible,[])
        self.assertEqual(self.nav.frame.stamp,1.)
        first=self.nav.frame
        self.observe(1.2,np.full(depth.shape,np.nan))
        self.assertFalse(self.cam.ground.result.valid)
        self.assertEqual(self.nav.frame.stamp,1.2)
        self.assertEqual(self.nav.frame.visual_xy,(None,None))
        np.testing.assert_equal(self.nav.command(1.21,0.),np.zeros(3))
        hook=self.nav.publication(first,2.)
        np.testing.assert_equal(hook(1.5,np.ones(3),self.rotation),np.zeros(3))
        np.testing.assert_equal(hook(2.,np.ones(3),self.rotation),np.zeros(3))

    def test_encoder_outage_uses_frame_velocity(self):
        frame=BoxFrame(np.zeros(2),np.array([.10,0.]),tuple(self.nav.boxes),0.,1.,1.,np.zeros(2),0.)
        self.nav.phase='along'
        self.nav.along_started=1.
        self.nav.frame=replace(frame,position=np.array([0.,.30]),edges=edges_at(.47),previous_edges=edges_at(.47),
                               previous_stamp=.90,progress_x=.30,forward_room=.12,depth_status='depth_clear',identified=(0,),axis_evidence=(True,True),visual_xy=(.30,.30))
        self.nav.latest_leg=(1.,np.array([.30,0.]),np.zeros(2))
        self.nav.command(1.05,0.)
        self.assertAlmostEqual(self.nav.steps.active['origin'][0],.30)
        # Inspect the captured-pose fallback in a new plan. The elapsed first
        # step separately requires an acknowledged stop before it can move again.
        self.nav.steps.stop(1.10)
        np.testing.assert_equal(self.nav.command(1.4,0.), np.zeros(3))
        self.nav.steps.published(1.41,np.zeros(3))
        self.nav.frame=replace(self.nav.frame,stamp=1.44,previous_stamp=1.42,progress_x=.32)
        self.nav.command(1.45,0.)
        self.assertAlmostEqual(self.nav.steps.active['origin'][0],.10*.01)

    def test_encoder_restart_does_not_integrate_missing_time(self):
        joints=dict.fromkeys(LEG_JOINTS,0.)
        self.assertTrue(self.nav.update_legs(1.,joints,self.rotation))
        previous=self.nav.latest_leg[1]
        joints[LEG_JOINTS[0]]=.4
        self.assertTrue(self.nav.update_legs(1.5,joints,self.rotation))
        np.testing.assert_equal(self.nav.latest_leg[1],previous)
        np.testing.assert_equal(self.nav.latest_leg[2],np.zeros(2))
        self.assertIsNone(self.nav.legs.at(1.2))

    def test_lane_handoff_is_one_shot(self):
        self.nav.index=2
        self.nav.phase='lane_entry'
        self.nav.frame=BoxFrame(np.array([2.53,0.]),np.zeros(2),tuple(self.nav.boxes),
                                0.,1.,1.,None,0.)
        self.nav.frame=replace(self.nav.frame,stamp=1.,tape_fix=True,tape_longitudinal=True,visual_xy=(2.53,0.),progress_x=2.53,
                               edges=edges_at(.22),previous_edges=edges_at(.22),previous_stamp=.9)
        self.assertIsNone(self.nav.command(1.,0.))
        self.assertTrue(self.nav.done)
        self.assertIsNone(self.nav.command(20.,0.))
        new_process=BoxServo()
        self.assertFalse(new_process.done)
        self.assertEqual(new_process.boxes,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
