"""Gamepad gesture, head preparation and full-lap process lifecycle tests."""
import argparse
import contextlib
import fcntl
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.navigation_buttons import FullLapStarter, HoldRelease, JoystickConnection, parse_buttons


def buttons(*pressed):
    return [int(index in pressed) for index in range(11)]


class GestureTests(unittest.TestCase):
    def setUp(self):
        self.gesture = HoldRelease((9, 10))
        self.now = 0.

    def feed(self, pressed=(), axes=None, dt=.1):
        self.now += dt
        return self.gesture.update(self.now, [0.]*8 if axes is None else axes, buttons(*pressed))

    def hold(self):
        for _ in range(13):
            self.assertFalse(self.feed((9, 10)))

    def test_one_event_on_full_release_and_another_hold_required(self):
        self.assertFalse(self.feed())
        self.hold()
        self.assertFalse(self.feed((9,)))
        self.assertTrue(self.feed())
        self.assertFalse(self.feed())
        self.hold()
        self.assertTrue(self.feed())

    def test_starting_with_held_buttons_requires_release_and_new_press(self):
        self.hold()
        self.assertFalse(self.feed())
        self.hold()
        self.assertTrue(self.feed())

    def test_short_tap_and_partial_chord_do_not_start(self):
        self.feed()
        for _ in range(3):self.feed((9, 10))
        self.assertFalse(self.feed())
        for _ in range(15):self.assertFalse(self.feed((9,)))
        self.assertFalse(self.feed())

    def test_other_button_or_manual_axis_cancels_gesture(self):
        for pressed, axes in (((9, 10, 0), [0.]*8), ((9, 10), [.5]+[0.]*7)):
            with self.subTest(pressed=pressed, axes=axes):
                self.feed()
                self.hold()
                self.assertFalse(self.feed(pressed, axes))
                self.assertFalse(self.feed())

    def test_disconnection_and_repeated_timestamp_cancel_gesture(self):
        for dt in (.3, 0.):
            with self.subTest(dt=dt):
                self.feed()
                self.hold()
                self.assertFalse(self.feed(dt=dt))

    def test_missing_buttons_or_nonfinite_axes_cancel_gesture(self):
        self.feed()
        self.hold()
        self.assertFalse(self.gesture.update(self.now+.1, [0.]*8, [0]*3))
        self.assertFalse(self.feed())
        self.hold()
        self.assertFalse(self.feed((9, 10), [float('nan')]+[0.]*7))
        self.assertFalse(self.feed())

    def test_binding_requires_explicit_distinct_indices(self):
        self.assertEqual(parse_buttons('9,10'), (9, 10))
        for value in ('', '9', '9,9', '-1,2', 'a,b'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_buttons(value)


class ConnectionTests(unittest.TestCase):
    def test_disconnection_is_detected_even_if_ros_keeps_replaying(self):
        poller = Mock()
        poller.poll.side_effect = [[], [(123, 16)], []]
        connection = JoystickConnection('/dev/input/js0')
        with patch('deploy.navigation_buttons.os.open', return_value=123), \
                patch('deploy.navigation_buttons.os.close') as close, \
                patch('deploy.navigation_buttons.os.fstat', return_value=SimpleNamespace(st_mode=stat.S_IFCHR)), \
                patch('deploy.navigation_buttons.select.poll', return_value=poller, create=True), \
                patch('deploy.navigation_buttons.select.POLLHUP', 16, create=True), \
                patch('deploy.navigation_buttons.select.POLLERR', 8, create=True), \
                patch('deploy.navigation_buttons.select.POLLNVAL', 32, create=True), \
                patch.object(Path, 'exists', return_value=True):
            self.assertTrue(connection.connected())
            self.assertFalse(connection.connected())
            close.assert_called_once_with(123)
            self.assertTrue(connection.connected())
            connection.close()

    def test_missing_device_is_disconnected(self):
        with patch('deploy.navigation_buttons.os.open', side_effect=FileNotFoundError):
            self.assertFalse(JoystickConnection('/dev/input/js0').connected())


class Process:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.terminations = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminations += 1


class StarterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root/'deploy').mkdir()
        (self.root/'deploy/task_head_pose.json').write_bytes((ROOT/'deploy/task_head_pose.json').read_bytes())
        config = json.loads((ROOT/'deploy/pro_config.json').read_text())
        self.reference = json.loads((ROOT/'deploy/task_head_pose.json').read_text())['measured_joint_reference_rad']
        config.update(extrinsics_verified=True, imu_alignment_verified=True,
                      calibration_evidence='offline test fixture', head_reference_rad=self.reference)
        self.config = self.root/'config.json'
        self.config.write_text(json.dumps(config))
        self.calls = []

        def create(command, **kwargs):
            process = Process(100+len(self.calls))
            self.calls.append((command, kwargs, process))
            return process

        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.starter = FullLapStarter(self.root, self.config, self.root/'head.urdf',
                                      self.root/'runs/buttons', process_factory=create,
                                      control_lock=self.root/'control.lock')
        self.starter.controller = dict(fsm=5, state='running', policy='amp')
        self.addCleanup(self.starter.release_control)

    def head_ready(self):
        self.starter.press()
        self.finish_head()

    def finish_head(self, **extra):
        report = dict(status='saved_pose_restored',
                      final_pitch=self.reference['head_pitch_joint'], final_yaw=0.)
        report.update(extra)
        (self.starter.lap/'head.json').write_text(json.dumps(report))
        self.starter.process.returncode = 0
        self.starter.poll()

    def finish_navigation(self, status, code=0):
        path = self.starter.lap/'navigation/vision'
        path.mkdir(parents=True)
        (path/'vision_summary.json').write_text(json.dumps(dict(status=status)))
        self.starter.process.returncode = code
        self.starter.poll()

    def test_first_press_only_restores_head_second_press_starts_full_lap(self):
        self.starter.press()
        self.assertEqual(self.starter.phase, 'preparing_head')
        command, kwargs, _ = self.calls[0]
        self.assertTrue(command[1].endswith('/deploy/head_step.py'))
        self.assertIn('--restore-saved-pose', command)
        self.assertIn('--allow-running', command)
        self.assertEqual(len(kwargs['pass_fds']), 1)
        self.starter.press()
        self.assertEqual(len(self.calls), 1)
        self.finish_head()
        self.assertEqual(self.starter.phase, 'head_ready')
        self.starter.poll()
        self.assertEqual(len(self.calls), 1)
        self.starter.press()
        command = self.calls[1][0]
        self.assertTrue(command[1].endswith('/deploy/line_trial.py'))
        self.assertIn('--shared-navigation', command)
        self.assertIn('--execute', command)
        self.assertEqual(command[command.index('--duration')+1], '300')
        self.assertEqual(self.starter.phase, 'running')
        self.starter.press()
        self.assertEqual(len(self.calls), 2)

    def test_failed_or_unverified_head_never_becomes_ready(self):
        for extra in (dict(status='aborted'), dict(final_pitch=0.),
                      dict(final_yaw=.1), dict(final_pitch=float('nan'))):
            with self.subTest(extra=extra):
                self.starter.press()
                self.finish_head(**extra)
                self.assertEqual(self.starter.phase, 'failed')
                self.assertTrue(all(call[0][1].endswith('head_step.py') for call in self.calls))

    def test_nonzero_head_exit_code_is_a_failure_even_with_success_report(self):
        self.starter.press()
        (self.starter.lap/'head.json').write_text(json.dumps(dict(status='saved_pose_restored')))
        self.starter.process.returncode = 2
        self.starter.poll()
        self.assertEqual(self.starter.phase, 'failed')

    def test_combination_during_head_motion_is_discarded(self):
        gesture = HoldRelease((9, 10))
        gesture.update(-.1, [0.]*8, buttons())
        self.starter.press()
        for i in range(15):
            self.starter.handle_joy(gesture, i*.1, [0.]*8, buttons(9, 10), i*.1)
        self.finish_head()
        self.starter.handle_joy(gesture, 1.5, [0.]*8, buttons(), 1.5)
        self.assertEqual(len(self.calls), 1)
        for i in range(16, 30):
            self.starter.handle_joy(gesture, i*.1, [0.]*8, buttons(9, 10), i*.1)
        self.starter.handle_joy(gesture, 3., [0.]*8, buttons(), 3.)
        self.assertEqual(self.starter.phase, 'running')

    def test_standby_keeps_head_ready_and_requires_a_new_press_after_running(self):
        self.starter.controller = dict(fsm=5, state='standby', policy='amp')
        self.head_ready()
        self.starter.press()
        self.assertEqual(self.starter.phase, 'head_ready')
        self.assertEqual(len(self.calls), 1)
        self.starter.controller = dict(fsm=5, state='running', policy='amp')
        self.starter.poll()
        self.assertEqual(len(self.calls), 1)
        self.starter.press()
        self.assertEqual(self.starter.phase, 'running')

    def test_manual_input_cancels_head_preparation(self):
        gesture = HoldRelease((9, 10))
        gesture.update(0., [0.]*8, buttons())
        self.starter.press()
        self.starter.handle_joy(gesture, .1, [.5]+[0.]*7, buttons(), .1)
        self.assertEqual(self.starter.phase, 'stopping')
        self.assertEqual(self.starter.process.terminations, 1)

    def test_stale_release_cannot_start_a_task(self):
        gesture = HoldRelease((9, 10))
        self.starter.handle_joy(gesture, 0., [0.]*8, buttons(), 0.)
        for i in range(1, 15):
            self.starter.handle_joy(gesture, i*.1, [0.]*8, buttons(9, 10), i*.1)
        self.starter.handle_joy(gesture, 1.5, [0.]*8, buttons(), 1.8)
        self.assertEqual(len(self.calls), 0)

    def test_control_lock_prevents_head_restore_during_another_task(self):
        with (self.root/'control.lock').open('a+') as other:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.starter.press()
            self.assertEqual(len(self.calls), 0)
            self.assertEqual(self.starter.phase, 'failed')
        self.head_ready()
        with (self.root/'control.lock').open('a+') as other:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_placeholder_or_different_head_calibration_blocks_second_press(self):
        for change in (dict(extrinsics_verified=False),
                       dict(head_reference_rad={'head_pitch_joint': .2, 'head_yaw_joint': 0.})):
            with self.subTest(change=change):
                self.head_ready()
                config = json.loads(self.config.read_text())
                config.update(change)
                self.config.write_text(json.dumps(config))
                count = len(self.calls)
                self.starter.press()
                self.assertEqual(self.starter.phase, 'failed')
                self.assertEqual(len(self.calls), count)

    def test_only_visual_finish_is_reported_as_a_completed_lap(self):
        self.head_ready()
        self.starter.press()
        self.finish_navigation('visual_finish_detected')
        self.assertEqual(self.starter.phase, 'finished')
        old_lap = self.starter.lap
        self.starter.press()
        self.assertEqual(self.starter.phase, 'preparing_head')
        self.assertNotEqual(self.starter.lap, old_lap)

    def test_timeout_and_old_duration_completion_are_not_lap_completion(self):
        for status, code in (('navigation_time_limit', 2), ('bounded_navigation_completed', 0),
                             ('error: No camera frames for 2 seconds', 2)):
            with self.subTest(status=status):
                self.head_ready()
                self.starter.press()
                self.finish_navigation(status, code)
                self.assertEqual(self.starter.phase, 'failed')

    def test_stop_waits_for_child_and_next_press_requires_head_preparation(self):
        self.head_ready()
        self.starter.press()
        process = self.starter.process
        self.starter.stop()
        self.starter.stop()
        self.assertEqual(process.terminations, 1)
        self.assertEqual(self.starter.phase, 'stopping')
        self.starter.press()
        self.assertEqual(len(self.calls), 2)
        process.returncode = 2
        self.starter.poll()
        self.assertEqual(self.starter.phase, 'idle')
        self.starter.press()
        self.assertEqual(self.starter.phase, 'preparing_head')

    def test_shared_supervisor_default_is_full_lap_duration(self):
        from deploy import line_trial
        with patch.object(sys, 'argv', ['line_trial.py', 'proxy', '--shared-navigation',
                                       '--output', str(self.root/'run')]), \
                patch.object(line_trial, 'proxy_role', return_value=0) as proxy:
            self.assertEqual(line_trial.main(), 0)
            self.assertEqual(proxy.call_args[0][0].duration, 300.)


if __name__ == '__main__':
    unittest.main()
