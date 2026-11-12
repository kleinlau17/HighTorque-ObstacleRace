from pathlib import Path
import copy
import json
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.head_profile import bounded_absolute_target,smooth_setpoint,validate_body_only_amp,saved_head_targets


class HeadProfileTests(unittest.TestCase):
    def test_continuous_restore_stays_bounded_and_has_limited_speed(self):
        for start,target in [(0.,.61046),(.8,.61046),(-.8,.61046)]:
            bounded_absolute_target(start,target,-1.05,.994)
            duration=max(1.,1.5*abs(target-start)/.1)
            t=np.linspace(0,duration,1001)
            q=np.array([smooth_setpoint(start,target,x,duration) for x in t])
            self.assertAlmostEqual(q[0],start)
            self.assertAlmostEqual(q[-1],target)
            self.assertGreaterEqual(q.min(),min(start,target)-1e-12)
            self.assertLessEqual(q.max(),max(start,target)+1e-12)
            self.assertLessEqual(np.max(np.abs(np.diff(q)/np.diff(t))),.100001)
            self.assertLess(abs((q[1]-q[0])/(t[1]-t[0])),.001)
            self.assertLess(abs((q[-1]-q[-2])/(t[-1]-t[-2])),.001)

    def test_refuses_out_of_range_start_and_target(self):
        for start,target in [(1.096,.61),(0.,.994),(float('nan'),.61)]:
            with self.assertRaises(ValueError):bounded_absolute_target(start,target,-1.05,.994)

    def test_invalid_profile_cannot_produce_a_command(self):
        for duration in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):smooth_setpoint(0.,.61,.5,duration)

    def test_saved_restore_targets_both_axes_and_preserves_pitch_feedback_offset(self):
        root=Path(__file__).resolve().parents[1]
        pose=json.loads((root/'deploy/task_head_pose.json').read_text())
        current={'head_pitch_joint':.1,'head_yaw_joint':-.5}
        limits={'head_pitch_joint':(-1.05,.994),'head_yaw_joint':(-1.57,1.57)}
        for old_record in (False,True):
            with self.subTest(old_record=old_record):
                record=copy.deepcopy(pose)
                if old_record:del record['joint_command_reference_rad']['head_yaw_joint']
                before=copy.deepcopy(record)
                targets,expected=saved_head_targets(record,current,limits)
                self.assertEqual(targets['head_yaw_joint'],0.)
                self.assertEqual(expected['head_yaw_joint'],0.)
                self.assertAlmostEqual(targets['head_pitch_joint'],.6104601502418519)
                self.assertAlmostEqual(expected['head_pitch_joint'],.6383716464042664)
                self.assertEqual(record,before)

    def test_restore_rejects_body_targets_and_invalid_yaw(self):
        root=Path(__file__).resolve().parents[1]
        pose=json.loads((root/'deploy/task_head_pose.json').read_text())
        current={'head_pitch_joint':.1,'head_yaw_joint':.5}
        limits={'head_pitch_joint':(-1.05,.994),'head_yaw_joint':(-1.57,1.57)}
        for value in (float('nan'),1.5):
            invalid=copy.deepcopy(pose)
            invalid['joint_command_reference_rad']['head_yaw_joint']=value
            with self.assertRaises(ValueError):saved_head_targets(invalid,current,limits)
        invalid=copy.deepcopy(pose)
        invalid['joint_command_reference_rad']['l_hip_pitch_joint']=0.
        with self.assertRaises(ValueError):saved_head_targets(invalid,current,limits)

    def test_running_policy_with_head_or_wrong_body_mapping_is_rejected(self):
        names=['body_joint_'+str(i) for i in range(20)]
        cfg=dict(algorithm='amp',motor_group='all-amp',dofs=20,joint_names=names,policy_name='walk/policy.rknn')
        self.assertEqual(validate_body_only_amp(cfg,names),'walk/policy.rknn')
        bad=dict(cfg,joint_names=names[:-1]+['head_pitch_joint'])
        with self.assertRaises(ValueError):validate_body_only_amp(bad,names)
        with self.assertRaises(ValueError):validate_body_only_amp(dict(cfg,dofs=22),names)
        with self.assertRaises(ValueError):validate_body_only_amp(cfg,names[:-1]+['different_body_joint'])


if __name__=='__main__':unittest.main()
