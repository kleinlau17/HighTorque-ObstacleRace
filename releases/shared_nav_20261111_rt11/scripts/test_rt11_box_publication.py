"""Exercise a candidate's frozen ROS publication plan without a ROS master."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.shared_navigation import PublishedPlan, TurnCutoff
from navigation.box_servo import BoxFrame, BoxServo


class BoxPublicationTest(unittest.TestCase):
    def setUp(self):
        self.nav = BoxServo()
        self.rotation = np.eye(3)
        self.vector = (.10, .02, .01)
        self.imu = (.2, [0., 0., 0., 1.])
        self.frame = BoxFrame(np.zeros(2), np.zeros(2), (np.array([1., .13]),),
                              0., 1., 0., None, 0., identified=(0,), axis_evidence=(True,True), visual_xy=(0.,0.))
        from test_rt11_box_steps import edges_at
        self.frame=replace(self.frame,edges=edges_at(.22),previous_edges=edges_at(.22),previous_stamp=-.1,
                           forward_room=.12,depth_status='depth_clear',progress_x=0.)
        self.nav.pending_frame = self.frame
        self.nav.commit_frame()
        self.plan = PublishedPlan(vector=self.vector, deadline=1., reason='box_enter',
                                  box_servo=self.nav.publication(self.frame, 1.))

    def evaluate(self, plan, t, imu):
        return plan.evaluate(t, imu, self.rotation, TurnCutoff())

    def test_frame_visibility_until_commit(self):
        next_frame = replace(self.frame, side=-1.)
        self.nav.pending_frame = next_frame
        first, *_ = self.evaluate(self.plan, .2, self.imu)
        self.assertGreater(first[1], 0.)
        self.nav.commit_frame()
        self.nav.target = None
        stale, *_ = self.evaluate(self.plan, .2, self.imu)
        np.testing.assert_equal(stale, np.zeros(3))
        self.assertIsNone(self.nav.target)
        new_plan = replace(self.plan, box_servo=self.nav.publication(next_frame, 1.))
        new, *_ = self.evaluate(new_plan, .2, self.imu)
        np.testing.assert_equal(new,np.zeros(3))  # reversing sides first emits zero
        self.nav.steps.published(.22,np.zeros(3))
        self.nav.frame=replace(next_frame,stamp=.32,previous_stamp=.27)
        self.assertLess(self.nav.command(.34,0.)[1],0.)

    def test_box_expiry_and_missing_imu_stop(self):
        for t, imu, reason in [(1., (.99, self.imu[1]), 'box_observation_expired'),
                               (.2, None, 'no_fresh_box_imu'),
                               (.36, self.imu, 'no_fresh_box_imu')]:
            vector, actual, fresh = self.evaluate(self.plan, t, imu)
            np.testing.assert_equal(vector, (0., 0., 0.))
            self.assertEqual(actual, reason)
            self.assertFalse(fresh)

    def test_lane_preserves_rt10_command(self):
        lane = replace(self.plan, reason='ready', box_servo=None)
        vector, reason, _ = self.evaluate(lane, 8., None)
        np.testing.assert_equal(vector, self.vector)
        self.assertEqual(reason, 'ready')

    def test_publisher_uses_completed_sensor_data_without_waiting(self):
        with self.nav.leg_lock:
            vector, *_ = self.evaluate(self.plan, .2, self.imu)
            self.assertGreater(vector[1], 0.)
        with self.nav.control_lock:
            vector, *_ = self.evaluate(self.plan, .2, self.imu)
            np.testing.assert_equal(vector, np.zeros(3))

    def test_weak_entrance_look_uses_measured_angle_and_original_lease(self):
        for ending in ('angle','lease','stripe'):
            with self.subTest(ending=ending):
                nav=BoxServo()
                for capture in (1.,2.,3.):
                    frame=replace(self.frame, side=-1., stamp=capture, edges=(None,None))
                    nav.frame=frame
                    self.assertLess(nav.command(capture+.47,0.)[2],0.)
                    self.assertEqual(bool(nav.steps.active.get('measured_look')),capture==3.)
                    hook=nav.publication(frame,capture+.75)
                    tick=capture+(.51 if capture==3. else .48)
                    vector=hook(tick,None,self.rotation)
                    self.assertLess(vector[2],0.)
                    hook.published(tick,vector)
                    if capture<3.:
                        zero=hook(capture+.52,None,self.rotation)
                        np.testing.assert_equal(zero,np.zeros(3))
                        hook.published(capture+.52,zero)
                # Even a newer frame offered with a longer lease cannot prolong this grant.
                seen=replace(frame,stamp=3.55)
                if ending=='stripe':
                    seen=replace(seen,edges=(self.frame.edges[1],None))
                nav.frame=seen
                new_hook=nav.publication(seen,4.55)
                if ending=='angle':
                    yaw=-.17
                    attitude=np.array([[np.cos(yaw),-np.sin(yaw),0.],
                                       [np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
                    command=new_hook(3.60,None,attitude)
                else:
                    command=new_hook(3.76 if ending=='lease' else 3.60,None,self.rotation)
                np.testing.assert_equal(command,np.zeros(3))

    def test_box_step_time_limit_at_large_monotonic_origin(self):
        for origin in (17.5213, 56231.271, 3285599.063801):
            nav=BoxServo()
            nav.frame=replace(self.frame,stamp=origin,previous_stamp=origin-.10)
            self.assertGreater(nav.command(origin,0.)[1],0.)
            np.testing.assert_equal(nav.command(origin+.41,0.),np.zeros(3))
            hook=nav.publication(nav.frame,origin+1.)
            hook.published(origin+.42,np.zeros(3))
            np.testing.assert_equal(nav.command(origin+.44,0.),np.zeros(3))

if __name__ == '__main__':
    unittest.main(verbosity=2)
