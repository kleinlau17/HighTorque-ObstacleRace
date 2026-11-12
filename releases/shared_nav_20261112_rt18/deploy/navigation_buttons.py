"""Two separate gamepad gestures: restore the saved view, then run one lap."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URDF = Path('reference/robot/PiPlusPro_SF_12L8A0G2H0W_Z_261010/urdf/PiPlusPro_SF_12L8A0G2H0W_Z_261010.urdf')
CONTROL_LOCK = Path('/tmp/competition_navigation_control.lock')
BUTTON_LOCK = Path('/tmp/competition_navigation_buttons.lock')


class JoystickConnection:
    """Observe Linux joystick disconnects independently of ROS replayed input."""
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None
        self.poller = None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.poller = None

    def connected(self):
        try:
            if self.fd is None:
                self.fd = os.open(str(self.path), os.O_RDONLY | os.O_NONBLOCK)
                if not stat.S_ISCHR(os.fstat(self.fd).st_mode):
                    self.close()
                    return False
                self.poller = select.poll()
                self.poller.register(self.fd, select.POLLHUP | select.POLLERR | select.POLLNVAL)
            if not self.path.exists() or self.poller.poll(0):
                self.close()
                return False
            return True
        except OSError:
            self.close()
            return False


def parse_buttons(text):
    try:
        indices = tuple(int(value) for value in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('Use comma-separated button indices')
    if len(indices) < 2 or min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError('Choose at least two different nonnegative button indices')
    return indices


class HoldRelease:
    """Require fresh neutral feedback and a complete hold/release per action."""
    def __init__(self, buttons, hold_seconds=1.):
        self.buttons = set(buttons)
        self.hold_seconds = hold_seconds
        self.last_receipt = None
        self.trigger_reference = None
        self.reset()

    def reset(self):
        self.require_release = True
        self.started = None
        self.qualified = False

    def neutral(self, axes):
        if len(axes) < 8 or not all(math.isfinite(value) for value in axes):
            return False
        return (all(abs(axes[index]) <= .08 for index in (0, 1, 3, 4, 6, 7))
                and self.trigger_reference is not None
                and all(abs(axes[index]-value) <= .1
                        for index, value in zip((2, 5), self.trigger_reference)))

    def update(self, timestamp, axes, buttons):
        if self.last_receipt is not None and not 0 < timestamp-self.last_receipt <= .2:
            self.reset()
        self.last_receipt = timestamp
        if (len(axes) < 8 or not self.buttons or max(self.buttons) >= len(buttons)
                or not all(math.isfinite(value) for value in axes)
                or any(value not in (0, 1) for value in buttons)):
            self.reset()
            return False
        pressed = {index for index, value in enumerate(buttons) if value}
        neutral = all(abs(axes[index]) <= .08 for index in (0, 1, 3, 4, 6, 7))
        if self.trigger_reference is None and not pressed and neutral:
            self.trigger_reference = (axes[2], axes[5])
        if not self.neutral(axes) or pressed-self.buttons:
            self.reset()
            return False
        if not pressed:
            activated = not self.require_release and self.qualified
            self.reset()
            self.require_release = False
            return activated
        if self.require_release:
            return False
        if pressed == self.buttons:
            if self.started is None:
                self.started = timestamp
            self.qualified |= timestamp-self.started >= self.hold_seconds
        else:
            self.started = None
        return False


def read_report(path):
    return json.loads(path.read_text()) if path.exists() else {}


class FullLapStarter:
    """Own head preparation and the existing supervisor as separate processes."""
    def __init__(self, root, config, urdf, output, speed=.22,
                 process_factory=subprocess.Popen, control_lock=CONTROL_LOCK):
        self.root = Path(root).resolve()
        self.config = Path(config).resolve()
        self.urdf = Path(urdf).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.speed = speed
        self.process_factory = process_factory
        self.control_lock_path = control_lock
        self.control_lock = None
        self.process = None
        self.lap = None
        self.controller = {}
        pose = json.loads((self.root/'deploy/task_head_pose.json').read_text())
        if pose.get('schema') != 'competition.task_head_pose.v1':
            raise ValueError('Unexpected saved head pose schema')
        self.reference = pose['measured_joint_reference_rad']
        if (set(self.reference) != {'head_pitch_joint', 'head_yaw_joint'}
                or not all(math.isfinite(value) for value in self.reference.values())):
            raise ValueError('Invalid saved head feedback reference')
        self.set_state('idle', '等待第一次组合键，恢复已保存的头部视角')

    def set_state(self, phase, detail):
        self.phase = phase
        status = dict(phase=phase, detail=detail, launcher_pid=os.getpid(),
                      child_pid=None if self.process is None else self.process.pid,
                      lap=None if self.lap is None else str(self.lap),
                      maximum_duration_s=300, head_reference_rad=self.reference,
                      controller=self.controller,
                      time_utc=datetime.now(timezone.utc).isoformat())
        temporary = self.output/'status.new'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        temporary.replace(self.output/'status.json')
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(status, ensure_ascii=False)+'\n')
        print(json.dumps(status, ensure_ascii=False), flush=True)

    def release_control(self):
        if self.control_lock is not None:
            self.control_lock.close()
            self.control_lock = None

    def spawn(self, command, log, pass_fds=()):
        with log.open('ab') as stream:
            self.process = self.process_factory(
                command, cwd=str(self.root), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=stream, start_new_session=True, pass_fds=pass_fds)

    def handle_joy(self, gesture, receipt, axes, buttons, now):
        if self.phase == 'preparing_head':
            other_button = any(value and index not in gesture.buttons for index, value in enumerate(buttons))
            if not gesture.neutral(axes) or other_button:
                self.stop()
        if now-receipt > .2 or self.phase in ('preparing_head', 'running', 'stopping'):
            gesture.reset()
        elif gesture.update(receipt, axes, buttons):
            self.press()
            gesture.reset()

    def press(self):
        if self.process is not None:
            return
        try:
            if self.phase == 'head_ready':
                if (self.controller.get('fsm') != 5 or self.controller.get('state') != 'running'
                        or self.controller.get('policy') != 'amp'):
                    self.set_state('head_ready', '头部已到位，请先通过原厂手柄进入 AMP running，再按组合键')
                    return
                from navigation.calibration import calibration_reasons
                reasons = calibration_reasons(json.loads(self.config.read_text()), self.reference)
                if reasons:
                    raise ValueError('Navigation calibration: '+', '.join(reasons))
                command = [sys.executable, str(self.root/'deploy/line_trial.py'), 'supervise',
                           '--shared-navigation', '--execute', '--duration', '300',
                           '--speed', str(self.speed), '--config', str(self.config),
                           '--output', str(self.lap/'navigation')]
                self.spawn(command, self.lap/'navigation.log')
                self.set_state('running', '整圈导航已启动，识别终点后结束，最长 300 秒')
            else:
                self.control_lock = self.control_lock_path.open('a+')
                fcntl.flock(self.control_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                name = datetime.now(timezone.utc).strftime('lap_%Y%m%d_%H%M%SZ_')+uuid.uuid4().hex[:8]
                self.lap = self.output/name
                self.lap.mkdir()
                command = [sys.executable, str(self.root/'deploy/head_step.py'),
                           '--restore-saved-pose', '--allow-running', '--execute',
                           '--urdf', str(self.urdf), '--output', str(self.lap/'head.json')]
                self.spawn(command, self.lap/'head.log', (self.control_lock.fileno(),))
                self.set_state('preparing_head', '正在恢复头部视角，完成后等待第二次组合键')
        except Exception as exc:
            self.release_control()
            self.set_state('failed', str(exc))

    def poll(self):
        if self.process is None or self.process.poll() is None:
            return
        returncode = self.process.returncode
        self.process = None
        self.release_control()
        if self.phase == 'stopping':
            self.set_state('idle', '任务已停止，下一次组合键重新准备头部')
            return
        try:
            if self.phase == 'preparing_head':
                report = read_report(self.lap/'head.json')
                if returncode or report.get('status') != 'saved_pose_restored':
                    raise RuntimeError(report.get('error', 'Head restore failed; see head.log'))
                for field, joint, tolerance in (('final_pitch', 'head_pitch_joint', .025),
                                                 ('final_yaw', 'head_yaw_joint', .04)):
                    value = report.get(field, float('nan'))
                    if not math.isfinite(value) or abs(value-self.reference[joint]) > tolerance:
                        raise RuntimeError('Saved head feedback was not reached: '+joint)
                detail = '头部已到位，请再次按下并松开组合键开始整圈导航'
                if self.controller.get('state') != 'running' or self.controller.get('policy') != 'amp':
                    detail = '头部已到位，请先通过原厂手柄进入 AMP running，再按组合键'
                self.set_state('head_ready', detail)
            else:
                supervisor = read_report(self.lap/'navigation/summary.json')
                vision = read_report(self.lap/'navigation/vision/vision_summary.json')
                if returncode == 0 and vision.get('status') == 'visual_finish_detected':
                    self.set_state('finished', '已识别并通过终点，本圈完成')
                else:
                    detail = dict(supervisor=supervisor.get('status'), vision=vision.get('status'),
                                  returncode=returncode, log=str(self.lap/'navigation.log'))
                    self.set_state('failed', detail)
        except Exception as exc:
            self.set_state('failed', str(exc))

    def stop(self):
        if self.process is not None and self.phase != 'stopping':
            self.process.terminate()
            self.set_state('stopping', '正在停止任务并等待控制程序退出')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-buttons', action='store_true')
    parser.add_argument('--start-buttons', type=parse_buttons,
                        help='Verified gamepad indices, e.g. i,j; no default binding')
    parser.add_argument('--hold-seconds', type=float, default=1.)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/pro_config.json')
    parser.add_argument('--urdf', type=Path, default=ROOT/URDF)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/navigation_buttons')
    parser.add_argument('--speed', type=float, default=.22)
    parser.add_argument('--joy-device', type=Path,
                        help='Linux joystick device; defaults to the original joy_node device')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if not args.inspect_buttons and args.start_buttons is None:
        parser.error('Inspect the original controller actions, then set --start-buttons to the verified combination')
    if not math.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error('--hold-seconds must be finite and positive')
    if not math.isfinite(args.speed) or not 0 < args.speed <= .4:
        parser.error('--speed must be in (0, 0.4]')
    if (ROOT/'runs').resolve() not in args.output.resolve().parents:
        parser.error('--output must be below this release runs directory')
    if args.background:
        if args.inspect_buttons:
            parser.error('Use button inspection in the foreground')
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output/'launcher.log').open('ab') as stream:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 *[value for value in sys.argv[1:] if value != '--background']],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                start_new_session=True)
        print(json.dumps(dict(status='starting', pid=child.pid, log=str(args.output/'launcher.log'))))
        return 0

    import rospy
    from sensor_msgs.msg import Joy
    from sim2real_msg.msg import StateInfo
    from std_msgs.msg import Int32
    guard = BUTTON_LOCK.open('a+')
    try:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        guard.close()
        raise SystemExit('A gamepad launcher is already active')
    rospy.init_node('competition_navigation_buttons', disable_signals=True)
    latest = {}
    control = {}
    lock = threading.Lock()
    stopping = threading.Event()

    def on_joy(message):
        with lock:
            latest.update(receipt=time.monotonic(), axes=list(message.axes), buttons=list(message.buttons))

    def on_state(message):
        if message._connection_header.get('callerid') == '/sim2real_master_node':
            with lock:
                control.update(state=message.state, policy=message.policy_name)

    def on_fsm(message):
        if message._connection_header.get('callerid') == '/sim2real_master_node':
            with lock:
                control.update(fsm=message.data)

    subscriptions = [rospy.Subscriber('/joy', Joy, on_joy, queue_size=1),
                     rospy.Subscriber('/sim2real_state_info', StateInfo, on_state, queue_size=1),
                     rospy.Subscriber('/fsm_state', Int32, on_fsm, queue_size=1)]
    signal.signal(signal.SIGINT, lambda *unused: stopping.set())
    signal.signal(signal.SIGTERM, lambda *unused: stopping.set())
    mapping = rospy.get_param('/teleop', {}).get('mode_change', {}).get('axis_mappings', [])
    print(json.dumps(dict(button_mapping={str(row['button']): row['target']
                                         for row in mapping if 'button' in row})), flush=True)
    starter = None
    connection = JoystickConnection(args.joy_device or rospy.get_param('/joy_node/dev', '/dev/input/js0'))
    gesture = HoldRelease(args.start_buttons or (), args.hold_seconds)
    previous_receipt = None
    previous_buttons = None
    was_connected = None
    try:
        if not args.inspect_buttons:
            starter = FullLapStarter(ROOT, args.config, args.urdf, args.output, args.speed)
        while not stopping.is_set() and not rospy.is_shutdown():
            with lock:
                sample = dict(latest)
                state = dict(control)
            connected = connection.connected()
            if connected != was_connected:
                print(json.dumps(dict(joystick_connected=connected, device=str(connection.path))), flush=True)
                gesture.reset()
                was_connected = connected
            if starter is not None:
                starter.controller = state
                if not connected:
                    starter.stop()
            # Consume the sample before completing a preparation job. A press
            # made during head movement must never become the second gesture.
            if sample and sample['receipt'] != previous_receipt:
                previous_receipt = sample['receipt']
                if args.inspect_buttons:
                    if sample['buttons'] != previous_buttons:
                        print(json.dumps(dict(pressed=[i for i, value in enumerate(sample['buttons']) if value],
                                              axes=sample['axes'], controller=state)), flush=True)
                        previous_buttons = sample['buttons']
                elif connected:
                    starter.handle_joy(gesture, sample['receipt'], sample['axes'],
                                       sample['buttons'], time.monotonic())
            if starter is not None:
                starter.poll()
            time.sleep(.02)
    finally:
        if starter is not None:
            starter.stop()
            while starter.process is not None:
                starter.poll()
                time.sleep(.02)
            starter.release_control()
        for subscription in subscriptions:
            subscription.unregister()
        rospy.signal_shutdown('gamepad launcher stopped')
        connection.close()
        guard.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
