"""Terminal entry regression tests with fake processes, without ROS or motors."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import navigation_cli as cli


class TerminalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root/'runs/terminal/lap'
        self.folder.mkdir(parents=True)
        (self.root/'deploy').mkdir()
        self.reference = dict(head_pitch_joint=.6383716464, head_yaw_joint=0.)
        (self.root/'deploy/task_head_pose.json').write_text(json.dumps(dict(measured_joint_reference_rad=self.reference)))
        self.state_path = self.folder.parent/'latest.json'
        self.config = self.root/'config.json'
        self.config.write_text('{}')
        self.lock = self.root/'control.lock'
        self.addCleanup(patch.stopall)
        patch.object(cli, 'ROOT', self.root).start()
        patch.object(cli, 'CONTROL_LOCK', self.lock).start()
        patch.object(cli, 'CLI_LOCK', self.root/'cli.lock').start()
        patch('navigation.release.verify_release').start()

    def head_report(self, **changes):
        report = dict(status='saved_pose_restored', final_pitch=self.reference['head_pitch_joint'], final_yaw=0.)
        report.update(changes)
        (self.folder/'head.json').write_text(json.dumps(report))

    def test_head_restores_both_axes_and_waits_for_feedback(self):
        self.head_report()
        process = Mock(pid=123)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(cli.subprocess, 'Popen', return_value=process) as spawn:
            result = cli.restore_head(self.root, self.folder, self.state_path, self.reference)
        command = spawn.call_args[0][0]
        self.assertIn('--restore-saved-pose', command)
        self.assertIn('--allow-running', command)
        self.assertIn('--execute', command)
        self.assertEqual(len(spawn.call_args[1]['pass_fds']), 1)
        process.wait.assert_called_once_with()
        self.assertEqual(result['status'], 'head_ready')

    def test_head_rejects_failed_missing_and_incomplete_feedback(self):
        process = Mock(pid=123)
        process.wait.return_value = 0
        process.poll.return_value = 0
        for changes in (dict(status='aborted'), dict(final_yaw=.06), dict(final_pitch=.70), dict(final_yaw=float('nan'))):
            with self.subTest(changes=changes):
                self.head_report(**changes)
                with patch.object(cli.subprocess, 'Popen', return_value=process), self.assertRaises(RuntimeError):
                    cli.restore_head(self.root, self.folder, self.state_path, self.reference)

    def test_head_does_not_spawn_when_control_is_owned(self):
        with self.lock.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(cli.subprocess, 'Popen') as spawn, self.assertRaises(BlockingIOError):
                cli.restore_head(self.root, self.folder, self.state_path, self.reference)
            spawn.assert_not_called()

    def test_ctrl_c_stops_head_child(self):
        process = Mock(pid=123)
        process.wait.side_effect = [KeyboardInterrupt, 0]
        process.poll.return_value = None
        with patch.object(cli.subprocess, 'Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            cli.restore_head(self.root, self.folder, self.state_path, self.reference)
        process.send_signal.assert_called_once_with(signal.SIGINT)

    def test_full_lap_uses_shared_detached_launcher_and_300_seconds(self):
        output = self.folder/'navigation'
        output.mkdir()
        (output/'launch.json').write_text(json.dumps(dict(status='launched', pid=123)))
        completed = SimpleNamespace(returncode=0, stdout='', stderr='')
        with patch.object(cli.subprocess, 'run', return_value=completed) as run:
            result = cli.start_lap(self.root, self.folder, self.state_path, self.config, .22)
        command = run.call_args[0][0]
        self.assertEqual(command[1], str(self.root/'deploy/launch_line_trial.py'))
        self.assertIn('--shared-navigation', command)
        self.assertIn('--execute', command)
        self.assertEqual(command[command.index('--duration')+1], '300')
        self.assertEqual(result['status'], 'lap_started')
        self.assertEqual(cli.read_json(self.state_path)['pid'], 123)

    def test_failed_launcher_does_not_write_task_record(self):
        with patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=2, stdout='', stderr='control busy')):
            with self.assertRaisesRegex(RuntimeError, 'control busy'):
                cli.start_lap(self.root, self.folder, self.state_path, self.config, .22)
        self.assertFalse(self.state_path.exists())

    def test_cli_start_checks_calibration_before_launch(self):
        with patch.object(cli, 'validate_start', side_effect=ValueError('calibration missing')), \
                patch.object(cli, 'start_lap') as start, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['start', '--config', str(self.config)]), 2)
        start.assert_not_called()

    def test_cli_start_accepts_current_ready_head_without_previous_head_command(self):
        with patch.object(cli, 'validate_start') as validate, \
                patch.object(cli, 'start_lap', return_value=dict(status='lap_started', detail='后台启动')) as start, \
                patch.object(cli, 'follow_lap', return_value=0) as follow, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['start', '--config', str(self.config)]), 0)
        validate.assert_called_once()
        start.assert_called_once()
        follow.assert_called_once()

    def test_duplicate_active_task_does_not_launch(self):
        with patch.object(cli, 'owned_process', return_value=True), patch.object(cli, 'start_lap') as start, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['start']), 2)
        start.assert_not_called()

    def test_status_and_stop_are_available_without_release_preflight(self):
        with patch('navigation.release.verify_release', side_effect=ValueError('broken release')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['status']), 0)
            self.assertEqual(cli.main(['stop']), 0)

    def test_follow_lap_forwards_stops_and_real_terminal_reason(self):
        vision = self.folder/'vision'
        vision.mkdir()
        (vision/'telemetry.jsonl').write_text('{"reason":"checking_raised_object","pending_raised_cells":2}\n{"t":')
        (vision/'stops.jsonl').write_text('{"event":"stop","reason":"checking_raised_object","image":"stops/001.jpg"}\n')
        (self.folder/'guard.jsonl').write_text('{"source":"stopped","command":[0,0,0]}\n')
        record = dict(kind='lap',pid=123,output=str(self.folder),maximum_duration_s=300)
        states = [dict(active=True,status='running',report={}),
                  dict(active=False,status='failed: navigation_time_limit',report={'manual_translator_restored':True})]
        output = io.StringIO()
        with patch.object(cli,'task_status',side_effect=states), patch.object(cli.time,'sleep'), contextlib.redirect_stdout(output):
            self.assertEqual(2,cli.follow_lap(record))
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(['stop','progress','terminal'],[entry['event'] for entry in messages])
        self.assertEqual(['vision/stops/001.jpg']*2,[messages[0]['image'],messages[-1]['last_stop_image']])
        self.assertEqual('failed: navigation_time_limit',messages[-1]['status'])
        self.assertEqual('checking_raised_object',messages[-1]['last_stop_reason'])
        self.assertEqual(2,messages[-1]['pending_raised_cells'])
        self.assertTrue(messages[-1]['manual_translator_restored'])

    def test_follow_reports_updated_reason_and_picture_identity(self):
        vision = self.folder/'vision'
        vision.mkdir()
        events = [dict(event='stop',stop_id=7,image='stops/7.jpg',reason='inhibited'),
                  dict(event='stop_reason',stop_id=7,image=None,reason='turn_stalled'),
                  dict(event='stop',stop_id=8,image=None,reason='final_guard_stop')]
        (vision/'stops.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
        record = dict(kind='lap',pid=123,output=str(self.folder),maximum_duration_s=300)
        output = io.StringIO()
        with patch.object(cli,'task_status',return_value=dict(active=False,status='full_lap_completed',report={})), contextlib.redirect_stdout(output):
            self.assertEqual(0,cli.follow_lap(record))
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual('vision/stops/7.jpg',rows[1]['image'])
        self.assertEqual(7,rows[-1]['last_stop_image_id'])
        self.assertIsNone(rows[-1]['last_stop_event']['image'])
        self.assertEqual('final_guard_stop',rows[-1]['last_stop_reason'])

    def test_follow_reads_incomplete_stop_only_after_line_is_finished(self):
        path = self.folder/'stops.jsonl'
        path.write_bytes(b'{"event":"stop","stop_id":1}\n{"event":"stop"')
        first, where = cli.stop_events(path,0)
        with path.open('ab') as stream:
            stream.write(b',"stop_id":2}\n')
        second, where = cli.stop_events(path,where)
        self.assertEqual([1,2],[event['stop_id'] for event in first+second])
        self.assertEqual([],cli.stop_events(path,where)[0])

    def test_empty_exit_report_and_ctrl_c_do_not_hide_the_supervisor_state(self):
        record = dict(kind='lap',pid=123,output=str(self.folder),maximum_duration_s=300)
        (self.folder/'summary.json').write_bytes(b'')
        with patch.object(cli,'owned_process',return_value=False):
            self.assertEqual('exited_with_unreadable_report',cli.task_status(record)['status'])
        with patch.object(cli,'owned_process',side_effect=KeyboardInterrupt), patch.object(cli,'stop_task') as stop, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(130,cli.follow_lap(record))
        stop.assert_not_called()

    def test_pid_reuse_is_not_owned(self):
        record = dict(pid=123, script='/release/deploy/line_trial.py', output='/release/runs/unique', kind='lap')
        for command, expected in ((b'python\0/release/deploy/line_trial.py\0--output\0/release/runs/unique\0', True),
                                  (b'python\0/release/deploy/line_trial.py\0--output\0/release/runs/other\0', False),
                                  (b'python\0/other.py\0--output\0/release/runs/unique\0', False)):
            with patch.object(Path, 'read_bytes', return_value=command):
                self.assertEqual(cli.owned_process(record), expected)

    def test_stop_waits_for_supervisor_cleanup_and_uses_term(self):
        record = dict(pid=123, kind='lap', output=str(self.folder), script='line_trial.py')
        with patch.object(cli, 'owned_process', side_effect=[True, False, False]), \
                patch.object(cli.os, 'kill') as kill:
            result = cli.stop_task(record)
        kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertFalse(result['active'])

    def test_visual_completion_and_timeout_remain_distinct_in_status(self):
        record = dict(pid=123, kind='lap', output=str(self.folder), script='line_trial.py')
        for status in ('full_lap_completed', 'failed: navigation_time_limit'):
            (self.folder/'summary.json').write_text(json.dumps(dict(status=status)))
            with patch.object(cli, 'owned_process', return_value=False):
                self.assertEqual(cli.task_status(record)['status'], status)


if __name__ == '__main__':
    unittest.main()
