"""Run the actual head command entry with in-memory ROS feedback and actuation."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.navigation_buttons import URDF


class JointState:
    def __init__(self):
        self.header=SimpleNamespace(stamp=None)
        self.name=[]
        self.position=[]


class HeadHarness:
    def __init__(self,pitch,yaw,pitch_bias=None,yaw_bias=0.,stuck_yaw=False,legacy_record=False):
        self.now=0.
        self.head={'head_pitch_joint':pitch,'head_yaw_joint':yaw}
        self.pose=json.loads((ROOT/'deploy/task_head_pose.json').read_text())
        if legacy_record:self.pose['joint_command_reference_rad'].pop('head_yaw_joint',None)
        reference=self.pose['measured_joint_reference_rad']
        command=self.pose['joint_command_reference_rad']
        self.pitch_bias=(reference['head_pitch_joint']-command['head_pitch_joint']
                         if pitch_bias is None else pitch_bias)
        self.yaw_bias=yaw_bias
        self.stuck_yaw=stuck_yaw
        self.commands=[]
        self.subscribers={}
        self.body_names=['body_'+str(index) for index in range(20)]
        self.policy=dict(algorithm='amp',motor_group='all-amp',dofs=20,
                         joint_names=self.body_names,policy_name='walk/test.rknn')
        self.pd=dict(joint_names=self.body_names+['head_yaw_joint','head_pitch_joint'],
                     lower=[-3.]*20+[-1.58,-1.58],upper=[3.]*20+[1.58,1.58])

    def snapshot(self,timestamp):
        body={name:(self.now,0.,0.) for name in self.body_names}
        head={name:(self.now,value,0.) for name,value in self.head.items()}
        return (self.now,np.array([0.,0.,0.,1.]),self.now),body,head

    def feedback(self):
        messages={
            '/fsm_state':SimpleNamespace(data=5),
            '/sim2real_state_info':SimpleNamespace(state='running',policy_name='amp',
                                                   policy_file_name=self.policy['policy_name'])}
        for topic,callback in self.subscribers.items():
            message=messages[topic]
            message._connection_header={'callerid':'/sim2real_master_node'}
            callback(message)

    def subscribe(self,topic,message_type,callback,**kwargs):
        self.subscribers[topic]=callback
        self.feedback()
        return SimpleNamespace(unregister=lambda:None)

    def sleep(self,duration):
        self.now+=duration
        self.feedback()

    def publisher(self,topic,message_type,**kwargs):
        if topic!='/pi_plus_absolute':raise AssertionError('Unexpected motor publisher')

        def publish(message):
            self.commands.append((self.now,list(message.name),list(message.position)))
            for name,value in zip(message.name,message.position):
                if name=='head_pitch_joint':self.head[name]=value+self.pitch_bias
                elif name=='head_yaw_joint' and not self.stuck_yaw:self.head[name]=value+self.yaw_bias
                elif name!='head_yaw_joint':raise AssertionError('Unexpected body joint command')

        return SimpleNamespace(publish=publish,get_num_connections=lambda:1,unregister=lambda:None)

    def run(self,execute=True,delta=None):
        rospy=ModuleType('rospy')
        rospy.init_node=lambda *args,**kwargs:None
        rospy.get_name=lambda:'/competition_head_pitch_step'
        rospy.Subscriber=self.subscribe
        rospy.Publisher=self.publisher
        rospy.Time=SimpleNamespace(now=lambda:self.now)
        rospy.signal_shutdown=lambda reason:None
        rosgraph=ModuleType('rosgraph')
        rosgraph.Master=lambda name:SimpleNamespace(getSystemState=lambda:([],[['/pi_plus_absolute',['/sim2real_master_node']]],[]))
        shadow=ModuleType('deploy.shadow')
        shadow.Sensors=lambda:SimpleNamespace(snapshot=self.snapshot)
        modules={'rospy':rospy,'rosgraph':rosgraph,'deploy.shadow':shadow}
        for package,classes in {
            'std_msgs':{'Int32':SimpleNamespace},
            'sensor_msgs':{'JointState':JointState},
            'sim2real_msg':{'StateInfo':SimpleNamespace},
        }.items():
            parent=ModuleType(package)
            child=ModuleType(package+'.msg')
            child.__dict__.update(classes)
            parent.msg=child
            modules[package]=parent
            modules[package+'.msg']=child
        original_text=Path.read_text
        original_bytes=Path.read_bytes

        def read_text(path,*args,**kwargs):
            if str(path).startswith('/home/nvidia/sim2real/') and path.name=='pd.yaml':
                return yaml.safe_dump(self.pd)
            if str(path)=='/home/nvidia/.ros/log/latest/rosout.log':
                return '1 INFO /sim2real_master_node [ModeFsmDef::print] from Init to ExecDefault\n'
            if path==ROOT/'deploy/task_head_pose.json':return json.dumps(self.pose)
            return original_text(path,*args,**kwargs)

        def read_bytes(path):
            if str(path)=='/home/nvidia/action_library/policy/amp/amp.yaml':
                return yaml.safe_dump(self.policy).encode()
            return original_bytes(path)

        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'head.json'
            argv=['head_step.py','--output',str(report),'--urdf',str(ROOT/URDF)]
            if delta is None:argv+=['--restore-saved-pose','--allow-running']
            else:argv+=['--delta',str(delta)]
            if execute:argv+=['--execute']
            with patch.dict(sys.modules,modules), \
                    patch.object(sys,'argv',argv), \
                    patch.object(Path,'read_text',read_text), \
                    patch.object(Path,'read_bytes',read_bytes), \
                    patch('time.monotonic',lambda:self.now), \
                    patch('time.sleep',self.sleep), \
                    contextlib.redirect_stdout(io.StringIO()):
                entry=runpy.run_path(str(ROOT/'deploy/head_step.py'))
                if delta is not None:
                    # The legacy pitch probe requires standby.
                    original_feedback=self.feedback
                    def standby_feedback():
                        original_feedback()
                        message=SimpleNamespace(state='standby',policy_name='amp',
                                                policy_file_name=self.policy['policy_name'],
                                                _connection_header={'callerid':'/sim2real_master_node'})
                        if '/sim2real_state_info' in self.subscribers:
                            self.subscribers['/sim2real_state_info'](message)
                    self.feedback=standby_feedback
                code=entry['main']()
            return code,json.loads(report.read_text())


class HeadRestoreTests(unittest.TestCase):
    def test_both_axes_reach_saved_pose_with_synchronized_bounded_profiles(self):
        for pitch,yaw in ((.1,.45),(.8,-.55)):
            with self.subTest(pitch=pitch,yaw=yaw):
                harness=HeadHarness(pitch,yaw)
                code,report=harness.run()
                self.assertEqual(code,0,report)
                self.assertEqual(report['status'],'saved_pose_restored')
                self.assertAlmostEqual(report['final_pitch'],.6383716464042664)
                self.assertAlmostEqual(report['final_yaw'],0.)
                self.assertEqual(report['named_joints'],['head_pitch_joint','head_yaw_joint'])
                time=np.array([row[0] for row in harness.commands])
                commands=np.array([row[2] for row in harness.commands])
                np.testing.assert_allclose(commands[0],[pitch,yaw])
                np.testing.assert_allclose(commands[-1],[.6104601502418519,0.])
                self.assertLessEqual(np.max(abs(np.diff(commands,axis=0)/np.diff(time)[:,None])),.100001)
                self.assertTrue(all(row[1]==report['named_joints'] for row in harness.commands))

    def test_yaw_alone_triggers_restore_with_pitch_already_at_reference(self):
        for legacy in (False,True):
            with self.subTest(legacy_record=legacy):
                harness=HeadHarness(.6383716464042664,.4,legacy_record=legacy)
                code,report=harness.run()
                self.assertEqual(code,0,report)
                self.assertGreater(report['messages_sent'],0)
                self.assertAlmostEqual(report['final_yaw'],0.)
                self.assertAlmostEqual(report['final_pitch'],.6383716464042664)
                np.testing.assert_allclose([row[2][0] for row in harness.commands],.6104601502418519)

    def test_pitch_alone_keeps_yaw_at_its_saved_zero_command(self):
        harness=HeadHarness(.1,0.)
        code,report=harness.run()
        self.assertEqual(code,0,report)
        np.testing.assert_array_equal([row[2][1] for row in harness.commands],0.)

    def test_already_at_both_targets_needs_no_new_motion(self):
        harness=HeadHarness(.6383716464042664,0.)
        code,report=harness.run()
        self.assertEqual(code,0,report)
        self.assertEqual(report['messages_sent'],0)
        self.assertEqual(report['final_yaw'],0.)

    def test_yaw_tracking_failure_and_final_residual_fail_preparation(self):
        for kwargs in (dict(stuck_yaw=True),dict(yaw_bias=.06)):
            with self.subTest(kwargs=kwargs):
                harness=HeadHarness(.1,.4,**kwargs)
                code,report=harness.run()
                self.assertEqual(code,2,report)
                self.assertEqual(report['status'],'aborted')
                self.assertIn('yaw',report['error'])

    def test_out_of_range_yaw_is_rejected_before_any_publisher(self):
        harness=HeadHarness(.1,1.5)
        code,report=harness.run()
        self.assertEqual(code,2,report)
        self.assertEqual(report['messages_sent'],0)

    def test_dry_run_reports_both_targets_without_publishing(self):
        harness=HeadHarness(.1,-.4)
        code,report=harness.run(execute=False)
        self.assertEqual(code,0,report)
        self.assertEqual(report['status'],'preflight_passed_no_motion')
        self.assertEqual(report['target_yaw'],0.)
        self.assertEqual(report['messages_sent'],0)

    def test_legacy_delta_probe_still_commands_pitch_only(self):
        harness=HeadHarness(.1,.3,pitch_bias=0.)
        code,report=harness.run(delta=.04)
        self.assertEqual(code,0,report)
        self.assertEqual(report['status'],'step_completed')
        self.assertEqual(report['named_joints'],['head_pitch_joint'])
        self.assertAlmostEqual(report['final_pitch'],.14)
        self.assertAlmostEqual(report['final_yaw'],.3)


if __name__=='__main__':
    unittest.main()
