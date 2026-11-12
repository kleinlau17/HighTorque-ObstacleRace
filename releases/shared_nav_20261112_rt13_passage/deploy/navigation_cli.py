"""On-robot terminal commands for saved head pose and one full navigation lap."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROL_LOCK = Path('/tmp/competition_navigation_control.lock')
CLI_LOCK = Path('/tmp/competition_navigation_cli.lock')
REASON_TEXT = {
    'checking_raised_object': '正在确认前方凸起',
    'unknown_depth': '深度净空不足以判定',
    'lost_vision': '前方巡线观测丢失',
    'turn_reobserving': '等待转向后的新画面',
    'turn_observation_expired': '转向观测超时',
    'turn_direction_mismatch': '测得的偏航方向与指令相反',
    'turn_stalled': '已发转向指令，未测得足够转动',
    'motion_stalled': '已发移动指令，未测得足够位移',
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def owned_process(record):
    """Match the unique output directory as well as the executable before signalling."""
    try:
        command = Path('/proc', str(int(record['pid'])), 'cmdline').read_bytes().split(b'\0')
        arguments = [value.decode() for value in command if value]
        return (record['script'] in arguments and '--output' in arguments
                and arguments[arguments.index('--output')+1] == record['output'])
    except (OSError, ValueError, KeyError, IndexError, UnicodeError):
        return False


def task_status(record):
    if not record:
        return dict(status='idle')
    output = Path(record['output'])
    report_path = output if record['kind'] == 'head' else output/'summary.json'
    active = owned_process(record)
    try:
        report = read_json(report_path)
        if not isinstance(report, dict):
            raise ValueError('expected a JSON object')
        missing = 'exited_without_report'
    except (ValueError, OSError) as exc:
        # A supervisor being torn down may leave an empty or partial summary.
        report = dict(report_error=str(exc))
        missing = 'exited_with_unreadable_report'
    status = 'running' if active else report.get('status', missing)
    return dict(record, status=status, active=active, report=report)


def stop_task(record):
    if not owned_process(record):
        return task_status(record)
    os.kill(record['pid'], signal.SIGINT if record['kind'] == 'head' else signal.SIGTERM)
    deadline = time.monotonic()+15
    while owned_process(record) and time.monotonic() < deadline:
        time.sleep(.1)
    result = task_status(record)
    if result['active']:
        result['status'] = 'stop_requested_waiting_for_cleanup'
    return result


def latest_row(path):
    """Read the latest complete row, ignoring a concurrent partial append."""
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size-32768))
            lines = stream.read().split(b'\n')[:-1]
            age = max(0., time.time()-os.fstat(stream.fileno()).st_mtime)
    except FileNotFoundError:
        return {}, None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict):
            return row, round(age, 1)
    return {}, None


def stop_events(path, offset):
    events = []
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(offset if offset<=size else 0)
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line.endswith(b'\n'):
                    break
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(event, dict) and event.get('event') in ('stop', 'stop_reason'):
                    events.append(event)
    except FileNotFoundError:
        pass
    return events, offset


def follow_lap(record):
    """Watch the exact detached run. Reading logs never takes ownership of motion."""
    output = Path(record['output'])
    vision = output/'vision'
    start = time.monotonic()
    # Allow the existing supervisor's startup, watchdog and teleop restoration.
    monitor_deadline = start+record['maximum_duration_s']+75
    last_print, signature = None, None
    offset, last_image, last_stop, last_cells = 0, None, None, None
    last_event = None
    last_image_stop_id = None
    try:
        while True:
            state = task_status(record)
            row, vision_age = latest_row(vision/'telemetry.jsonl')
            guard, guard_age = latest_row(output/'guard.jsonl')
            events, offset = stop_events(vision/'stops.jsonl', offset)
            for event in events:
                event = dict(event)
                if event.get('image'):
                    last_image = str(Path('vision')/event['image'])
                    last_image_stop_id = event.get('stop_id')
                    event['image'] = last_image
                elif event['event']=='stop_reason' and last_event and last_event.get('stop_id')==event.get('stop_id'):
                    event['image'] = last_event.get('image')
                if event.get('reason') in REASON_TEXT:
                    event['detail'] = REASON_TEXT[event['reason']]
                last_event = event
                print(json.dumps(event, ensure_ascii=False), flush=True)
            reason = row.get('reason')
            if reason and reason not in ('ready', 'finished'):
                last_stop = reason
            if 'pending_raised_cells' in row:
                last_cells = row['pending_raised_cells']
            now = time.monotonic()
            if not state['active'] or now>=monitor_deadline:
                report = state['report']
                status = state['status']
                if state['active']:
                    # A hung process may still own the robot. Do not call it stopped.
                    status = 'monitor_deadline_supervisor_still_active'
                visual = {}
                try:
                    visual = read_json(vision/'vision_summary.json')
                except (ValueError, OSError):
                    pass
                visual = visual if isinstance(visual, dict) else {}
                final_nav = visual.get('last_navigation')
                final_nav = final_nav if isinstance(final_nav, dict) else {}
                last_guard = {k:guard.get(k) for k in ('t','allowed','source','reason','command')}
                last_guard['age_s'] = guard_age
                print(json.dumps(dict(event='terminal', release=ROOT.name, pid=record['pid'],
                    status=status, active=state['active'], visual_status=visual.get('status'),
                    last_navigation_reason=final_nav.get('reason') or reason,
                    last_navigation_stop_reason=last_stop,
                    last_stop_reason=(last_event or {}).get('reason') or last_stop,
                    pending_raised_cells=final_nav.get('pending_raised_cells',last_cells),
                    last_guard=last_guard if guard else None, last_stop_event=last_event,
                    last_stop_image=last_image, last_stop_image_id=last_image_stop_id,
                    manual_translator_restored=report.get('manual_translator_restored'),
                    output=str(output)), ensure_ascii=False), flush=True)
                return 0 if status=='full_lap_completed' else 2
            fields = dict(vision=reason, scene=row.get('scene'), lane=row.get('lane_basis'),
                          mode=row.get('mode'), gap=row.get('gap_phase'),
                          guard=guard.get('source'), guard_reason=guard.get('reason'),
                          vision_stale=vision_age is not None and vision_age>3.,
                          guard_stale=guard_age is not None and guard_age>3.)
            changed = tuple(fields.values())
            if changed!=signature or last_print is None or now-last_print>=5.:
                print(json.dumps(dict(event='progress', elapsed_s=round(now-start, 1),
                    requested=row.get('desired_velocity'), command=guard.get('command'),
                    vision_age_s=vision_age, guard_age_s=guard_age, detail=REASON_TEXT.get(reason),
                    pending_raised_cells=last_cells, **fields), ensure_ascii=False), flush=True)
                last_print, signature = now, changed
            time.sleep(.5)
    except KeyboardInterrupt:
        print('已退出终端跟踪。后台任务继续运行。停止本次导航请执行 '
              +str(ROOT/'deploy/nav')+' stop', flush=True)
        return 130


def validate_start(config, reference):
    from navigation.calibration import calibration_reasons
    from deploy.shared_navigation import preflight
    report = preflight(config)
    # RT10: start preconditions are reported, never enforced. Once the walk
    # begins, only the 300 s mission limit stops it.
    warnings = list(calibration_reasons(config, reference))
    import rospy
    from sensor_msgs.msg import JointState
    from sim2real_msg.msg import StateInfo
    from std_msgs.msg import Int32
    rospy.init_node('competition_navigation_cli', anonymous=True, disable_signals=True)
    try:
        state = rospy.wait_for_message('/sim2real_state_info', StateInfo, timeout=3)
        fsm = rospy.wait_for_message('/fsm_state', Int32, timeout=3)
        if fsm.data != 5 or state.state != 'running' or state.policy_name != 'amp':
            warnings.append('amp_not_running:'+str(state.state))
        joints = rospy.wait_for_message('/sim2real_master_node/rbt_state', JointState, timeout=3)
        warnings.extend(calibration_reasons(config, dict(zip(joints.name, joints.position))))
    except Exception as exc:
        warnings.append('start_check_incomplete:'+str(exc))
    report['start_warnings'] = warnings
    return report


def restore_head(root, folder, state_path, reference):
    output = folder/'head.json'
    script = str(root/'deploy/head_step.py')
    command = [sys.executable, script, '--restore-saved-pose', '--allow-running',
               '--execute', '--output', str(output)]
    with CONTROL_LOCK.open('a+') as control:
        fcntl.flock(control.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder/'head.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True,
                                       pass_fds=(control.fileno(),))
        record = dict(kind='head', pid=process.pid, script=script, output=str(output),
                      log=str(folder/'head.log'))
        try:
            write_json(state_path, record)
            print('正在恢复俯仰和偏航，等待两个关节反馈到位……', flush=True)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
        report = read_json(output)
        if returncode or report.get('status') != 'saved_pose_restored':
            raise RuntimeError('头部恢复失败，请查看 '+str(folder/'head.log')+' '+str(report.get('error', '')))
        for key, joint, tolerance in (('final_pitch', 'head_pitch_joint', .025),
                                      ('final_yaw', 'head_yaw_joint', .04)):
            value = report.get(key, float('nan'))
            if not math.isfinite(value) or abs(value-reference[joint]) > tolerance:
                raise RuntimeError('头部反馈未到位 '+joint)
        return dict(record, status='head_ready', report=report)


def start_lap(root, folder, state_path, config, speed):
    output = folder/'navigation'
    command = [sys.executable, str(root/'deploy/launch_line_trial.py'),
               '--shared-navigation', '--execute', '--duration', '300',
               '--speed', str(speed), '--config', str(config), '--output', str(output)]
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    launch = read_json(output/'launch.json')
    if launch.get('status') != 'launched':
        raise RuntimeError('未取得整圈任务启动记录 '+result.stdout)
    record = dict(kind='lap', pid=launch['pid'], script=str(root/'deploy/line_trial.py'),
                  output=str(output), log=str(output/'mission.log'), maximum_duration_s=300)
    try:
        write_json(state_path, record)
    except OSError:
        stop_task(record)
        raise
    return dict(record, status='lap_started', detail='整圈任务已在后台启动，识别终点后结束，最长 300 秒')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('head', 'start', 'status', 'stop', 'check'))
    parser.add_argument('--config', type=Path,
                        default=Path(os.environ.get('NAV_CALIBRATION_FILE', str(ROOT/'deploy/pro_model_config.json'))))
    parser.add_argument('--speed', type=float, default=.22)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or not 0 < args.speed <= .4:
        parser.error('speed must be within (0, 0.4] m/s')
    base = ROOT/'runs/terminal'
    base.mkdir(parents=True, exist_ok=True)
    state_path = base/'latest.json'
    try:
        if args.action in ('status', 'stop'):
            record = read_json(state_path)
            result = stop_task(record) if args.action == 'stop' else task_status(record)
        else:
            from navigation.release import verify_release
            verify_release(ROOT)
            reference = read_json(ROOT/'deploy/task_head_pose.json')['measured_joint_reference_rad']
            if args.action == 'check':
                from deploy.shared_navigation import preflight
                result = preflight(read_json(args.config))
            else:
                with CLI_LOCK.open('a+') as operation:
                    fcntl.flock(operation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    current = read_json(state_path)
                    if owned_process(current):
                        raise RuntimeError('已有终端任务运行，请执行 nav status 查看或 nav stop 停止')
                    if args.action == 'start':
                        validate_start(read_json(args.config), reference)
                    name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid.uuid4().hex[:8]
                    folder = base/name
                    folder.mkdir()
                    if args.action == 'head':
                        result = restore_head(ROOT, folder, state_path, reference)
                    else:
                        result = start_lap(ROOT, folder, state_path, args.config.resolve(), args.speed)
        if args.action=='start':
            result['release'] = ROOT.name
            result['stop_command'] = str(ROOT/'deploy/nav')+' stop'
            result['detail'] += '。Ctrl-C 仅退出日志跟踪，可另开终端执行 stop_command。'
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return follow_lap(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BlockingIOError:
        print('已有头部或导航任务占用控制，请等待完成或执行 nav stop。', file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
