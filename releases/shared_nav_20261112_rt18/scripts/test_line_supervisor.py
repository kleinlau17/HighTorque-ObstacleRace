"""Exercise the real velocity proxy with in-memory ROS transport."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import line_trial


class Twist:
    def __init__(self):
        self.linear = SimpleNamespace(x=0., y=0., z=0.)
        self.angular = SimpleNamespace(x=0., y=0., z=0.)


class TwistStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id='supervisor-test')
        self._connection_header = {'callerid': '/competition_lane_trial'}
        self.twist = Twist()


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ProxyHarness:
    def __init__(self, config_path):
        self.now = 100.
        self.publishers = {}
        self.callbacks = {}
        self.config_path = config_path
        self.policy_text = (ROOT/'vendor/pi_plus_pro/amp.yaml').read_text()
        self.policy = yaml.safe_load(self.policy_text)
        self.args = SimpleNamespace(
            shared_navigation=True, calibrated_turns=False, turn_test=False,
            duration=30., speed=.22, yaw_limit=1., acceleration=.4,
            token='supervisor-test', config=config_path, execute=True)

    def publisher(self, topic, *args, **kwargs):
        publisher = Publisher()
        self.publishers[topic] = publisher
        return publisher

    def subscriber(self, topic, message_type, callback, **kwargs):
        self.callbacks[topic] = callback
        return SimpleNamespace()

    def send_navigation(self, values=(.22, .1, .3), caller='/competition_lane_trial',
                        token='supervisor-test'):
        message = TwistStamped()
        message._connection_header['callerid'] = caller
        message.header.frame_id = token
        message.twist.linear.x, message.twist.linear.y, message.twist.angular.z = values
        self.callbacks['/competition/lane_request'](message)

    def seed(self, guard):
        self.callbacks['/joy'](SimpleNamespace(axes=[0.]*8, buttons=[0]*11))
        self.callbacks['/fsm_state'](SimpleNamespace(
            data=5, _connection_header={'callerid': '/sim2real_master_node'}))
        self.callbacks['/sim2real_state_info'](SimpleNamespace(
            state='running', policy_name='amp', policy_file_name=self.policy['policy_name']))
        names = self.policy['joint_names']+['head_pitch_joint', 'head_yaw_joint']
        self.callbacks['/sim2real_master_node/rbt_state'](SimpleNamespace(
            name=names, position=[0.]*20+[.6383716464, 0.]))
        self.callbacks['/imu/data'](SimpleNamespace(
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.),
            angular_velocity=SimpleNamespace(x=0., y=0., z=0.)))
        self.callbacks['/battery_level'](SimpleNamespace(data=80))
        self.callbacks['/sim2real_debug_data_obs'](SimpleNamespace(data=[0.]*69))
        guard.publish(Twist())
        self.send_navigation()

    def tick(self, guard):
        guard.tick(None)
        status = json.loads(self.publishers['/competition/line_guard'].messages[-1].data)
        message = self.publishers['/cmd_vel'].messages[-1]
        output = [message.linear.x, message.linear.y, message.angular.z]
        if output != status['command']:
            raise AssertionError('Status differs from published velocity')
        return status

    def run(self, exercise):
        rospy = ModuleType('rospy')
        rospy.Publisher = self.publisher
        rospy.Subscriber = self.subscriber
        rospy.Timer = lambda *args: None
        rospy.Duration = float
        rospy.resolve_name = lambda name: name
        rospy.on_shutdown = lambda callback: None
        rospy.signal_shutdown = lambda reason: None
        modules = {'rospy': rospy, 'rosgraph': ModuleType('rosgraph')}
        for package, classes in {
            'geometry_msgs': {'Twist': Twist, 'TwistStamped': TwistStamped},
            'sensor_msgs': dict.fromkeys(('Joy', 'Imu', 'JointState'), SimpleNamespace),
            'std_msgs': dict.fromkeys(('Int32', 'String', 'UInt8'), SimpleNamespace),
            'sim2real_msg': dict.fromkeys(('StateInfo', 'DebugData'), SimpleNamespace),
        }.items():
            parent = ModuleType(package)
            child = ModuleType(package+'.msg')
            child.__dict__.update(classes)
            parent.msg = child
            modules[package] = parent
            modules[package+'.msg'] = child

        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path) == '/home/nvidia/action_library/policy/amp/amp.yaml':
                return self.policy_text
            return original_read(path, *args, **kwargs)

        def factory_script(path, run_name):
            if path != '/opt/ros/noetic/lib/joy_teleop/joy_teleop.py':
                raise AssertionError('Unexpected factory entry point')
            guard = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            self.seed(guard)
            exercise(guard)

        with patch.dict(sys.modules, modules), \
                patch.object(Path, 'read_text', read_text), \
                patch.object(line_trial.runpy, 'run_path', factory_script), \
                patch.object(line_trial.signal, 'signal'), \
                patch.object(line_trial.time, 'monotonic', lambda: self.now), \
                patch.object(line_trial.time, 'sleep'), \
                patch.object(sys, 'argv', list(sys.argv)):
            line_trial.proxy_role(self.args)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = json.loads((ROOT/'deploy/pro_config.json').read_text())
        config.update(extrinsics_verified=True, imu_alignment_verified=True,
                      calibration_evidence='in-memory test fixture',
                      head_reference_rad={'head_pitch_joint': .6383716464, 'head_yaw_joint': 0.})
        self.config = Path(directory.name)/'config.json'
        self.config.write_text(json.dumps(config))

    def assert_navigating(self, status):
        self.assertTrue(status['allowed'], status)
        self.assertEqual(status['source'], 'navigation', status)
        self.assertGreater(status['command'][0], 0.)

    def assert_inhibited(self, status, reason):
        self.assertFalse(status['allowed'], status)
        self.assertEqual(status['reason'], reason)
        self.assertEqual(status['command'], [0., 0., 0.])

    def test_low_missing_and_stale_battery_are_telemetry_only(self):
        for shared in (False, True):
            for level, age in ((0, 0.), (14, 0.), (80, 20.), (None, None)):
                with self.subTest(shared=shared, level=level, age=age):
                    harness = ProxyHarness(self.config)
                    harness.args.shared_navigation = shared

                    def exercise(guard):
                        if level is None:
                            del guard.cache['battery']
                        else:
                            guard.cache['battery'] = (harness.now-age, SimpleNamespace(data=level))
                        status = harness.tick(guard)
                        self.assert_navigating(status)
                        self.assertEqual(status['battery'], dict(percent=level, age_s=age))

                    harness.run(exercise)

    def test_tilt_and_angular_rate_are_telemetry_only(self):
        for shared in (False, True):
            for tilt, rate in ((20., 0.), (0., 2.), (45., -3.)):
                with self.subTest(shared=shared, tilt=tilt, rate=rate):
                    harness = ProxyHarness(self.config)
                    harness.args.shared_navigation = shared

                    def exercise(guard):
                        imu = guard.cache['imu'][1]
                        imu.orientation.x = math.sin(math.radians(tilt)/2)
                        imu.orientation.w = math.cos(math.radians(tilt)/2)
                        imu.angular_velocity.z = rate
                        status = harness.tick(guard)
                        self.assert_navigating(status)
                        self.assertAlmostEqual(status['attitude']['absolute_tilt_deg'], tilt)
                        self.assertEqual(status['attitude']['yaw_rate_rad_s'], rate)

                    harness.run(exercise)

    def test_large_command_integrals_do_not_stop_navigation(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                harness = ProxyHarness(self.config)
                harness.args.shared_navigation = shared

                def exercise(guard):
                    guard.budget = 10000.
                    guard.turn_budget = 10000.
                    status = harness.tick(guard)
                    self.assert_navigating(status)
                    self.assertGreater(status['distance_command_budget'], 10000.)
                    self.assertGreater(status['turn_command_budget'], 10000.)

                harness.run(exercise)

    def test_manual_takeover_with_removed_limits_and_stale_navigation(self):
        harness = ProxyHarness(self.config)

        def exercise(guard):
            del guard.cache['battery']
            guard.budget = guard.turn_budget = 10000.
            guard.cache['imu'][1].angular_velocity.z = 3.
            guard.cache['joy'][1].axes[0] = .5
            guard.cache['nav'] = (harness.now-1., guard.cache['nav'][1])
            message = Twist()
            message.linear.x, message.linear.y, message.angular.z = -.2, .1, -.4
            guard.publish(message)
            status = harness.tick(guard)
            self.assertTrue(status['allowed'], status)
            self.assertEqual(status['source'], 'manual')
            self.assertEqual(status['command'], [-.2, .1, -.4])

        harness.run(exercise)

    def test_navigation_timeout_stops_and_fresh_request_resumes(self):
        harness = ProxyHarness(self.config)

        def exercise(guard):
            self.assert_navigating(harness.tick(guard))
            guard.cache['nav'] = (harness.now-.201, guard.cache['nav'][1])
            status = harness.tick(guard)
            self.assertEqual(status['source'], 'stopped')
            self.assertEqual(status['command'], [0., 0., 0.])
            harness.send_navigation()
            self.assert_navigating(harness.tick(guard))

        harness.run(exercise)

    def test_stale_control_feedback_stops(self):
        for key in ('joy', 'fsm', 'joints', 'imu', 'manual', 'policy_obs'):
            with self.subTest(key=key):
                harness = ProxyHarness(self.config)

                def exercise(guard):
                    guard.cache[key] = (harness.now-.201, guard.cache[key][1])
                    reason = 'policy_observation_stale' if key == 'policy_obs' else 'source_stale'
                    self.assert_inhibited(harness.tick(guard), reason)

                harness.run(exercise)

    def test_request_source_and_task_token_are_checked(self):
        for caller, token in (('/another_node', 'supervisor-test'), ('/competition_lane_trial', 'old-task')):
            with self.subTest(caller=caller, token=token):
                harness = ProxyHarness(self.config)

                def exercise(guard):
                    del guard.cache['nav']
                    harness.send_navigation(caller=caller, token=token)
                    self.assertNotIn('nav', guard.cache)
                    self.assertEqual(harness.tick(guard)['command'], [0., 0., 0.])

                harness.run(exercise)

    def test_control_state_policy_and_joint_contract_are_checked(self):
        cases = (
            ('fsm', 'data', 0, 'not_running_amp'),
            ('fsm', '_connection_header', {'callerid': '/other'}, 'wrong_fsm_source'),
            ('state', 'state', 'standby', 'not_running_amp'),
            ('state', 'policy_name', 'other', 'not_running_amp'),
            ('state', 'policy_file_name', 'other.rknn', 'policy_changed'),
            ('joints', 'position', [0.]*21, 'incomplete_joint_feedback'),
            ('joints', 'position', [float('nan')]*22, 'incomplete_joint_feedback'),
        )
        for key, field, value, reason in cases:
            with self.subTest(key=key, field=field, value=value):
                harness = ProxyHarness(self.config)

                def exercise(guard):
                    setattr(guard.cache[key][1], field, value)
                    self.assert_inhibited(harness.tick(guard), reason)

                harness.run(exercise)

    def test_nonfinite_imu_values_are_rejected(self):
        for field in ('orientation', 'angular_velocity'):
            with self.subTest(field=field):
                harness = ProxyHarness(self.config)

                def exercise(guard):
                    setattr(getattr(guard.cache['imu'][1], field), 'z', float('nan'))
                    self.assert_inhibited(harness.tick(guard), 'imu_invalid')

                harness.run(exercise)

    def test_invalid_navigation_and_policy_feedback_are_rejected(self):
        for values, reason in (((float('nan'), 0., 0.), 'invalid_navigation_request'),
                               ((.401, 0., 0.), 'navigation_target_out_of_bounds')):
            with self.subTest(values=values):
                harness = ProxyHarness(self.config)

                def exercise(guard):
                    harness.send_navigation(values)
                    self.assert_inhibited(harness.tick(guard), reason)

                harness.run(exercise)
        harness = ProxyHarness(self.config)

        def exercise(guard):
            guard.cache['policy_obs'][1].data[6] = float('nan')
            self.assert_inhibited(harness.tick(guard), 'Invalid policy command feedback')

        harness.run(exercise)

    def test_head_calibration_is_checked(self):
        harness = ProxyHarness(self.config)

        def exercise(guard):
            guard.cache['joints'][1].position[-1] = .1
            self.assert_inhibited(harness.tick(guard), 'calibrated_head_pose_changed')

        harness.run(exercise)

    def test_explicit_duration_and_shadow_mode_stop_navigation(self):
        for execute in (False, True):
            with self.subTest(execute=execute):
                harness = ProxyHarness(self.config)
                harness.args.execute = execute

                def exercise(guard):
                    if execute:
                        guard.begun = harness.now-harness.args.duration
                    status = harness.tick(guard)
                    self.assertEqual(status['source'], 'stopped')
                    self.assertEqual(status['command'], [0., 0., 0.])

                harness.run(exercise)

    def test_shared_navigation_accepts_large_heading_change(self):
        harness = ProxyHarness(self.config)

        def exercise(guard):
            self.assert_navigating(harness.tick(guard))
            q = guard.cache['imu'][1].orientation
            q.z, q.w = math.sin(math.radians(120)/2), math.cos(math.radians(120)/2)
            self.assert_navigating(harness.tick(guard))

        harness.run(exercise)


if __name__ == '__main__':
    unittest.main()
