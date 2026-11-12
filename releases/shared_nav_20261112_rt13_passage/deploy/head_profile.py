"""Continuous head motion profile inside the verified mechanical work range."""
import math


def bounded_absolute_target(current, target, lower, upper):
    if not all(math.isfinite(v) for v in (current, target, lower, upper)):
        raise ValueError('Non-finite head command')
    lo, hi = max(-1.4, lower+.1), min(1.4, upper-.1)
    if not lo <= current <= hi or not lo <= target <= hi:
        raise ValueError('Head target outside bounded work range')
    return float(target)


def smooth_setpoint(start, target, elapsed, duration):
    if not all(math.isfinite(v) for v in (start, target, elapsed, duration)) or duration <= 0:
        raise ValueError('Invalid head motion profile')
    u = min(1., max(0., elapsed/duration))
    return start+(target-start)*(u*u*(3.-2.*u))


def saved_head_targets(pose, current, limits):
    """Validate both head axes, including previously saved pitch-only records."""
    if pose.get('schema') != 'competition.task_head_pose.v1':
        raise ValueError('Unexpected saved head pose schema')
    names = {'head_pitch_joint', 'head_yaw_joint'}
    command = dict(pose['joint_command_reference_rad'])
    measured = dict(pose['measured_joint_reference_rad'])
    if set(measured) != names or set(current) != names or set(limits) != names:
        raise ValueError('Saved head restore requires both head joints')
    # Earlier records saved only the pitch command with measured yaw at zero.
    if set(command) == {'head_pitch_joint'} and measured['head_yaw_joint'] == 0.:
        command['head_yaw_joint'] = 0.
    if set(command) != names:
        raise ValueError('Saved commands must contain exactly the two head joints')
    for name in ('head_pitch_joint', 'head_yaw_joint'):
        lower, upper = limits[name]
        command[name] = bounded_absolute_target(current[name], float(command[name]), lower, upper)
        measured[name] = bounded_absolute_target(current[name], float(measured[name]), lower, upper)
    return command, measured


def validate_body_only_amp(config, body_names):
    names=config.get('joint_names',[])
    if (config.get('algorithm')!='amp' or config.get('motor_group')!='all-amp'
            or config.get('dofs')!=20 or len(names)!=20 or len(set(names))!=20
            or set(names)!=set(body_names)
            or any('head' in name for name in names)):
        raise ValueError('Running policy must control exactly the 20 body joints and exclude head')
    policy_file=config.get('policy_name')
    if not isinstance(policy_file,str) or not policy_file:
        raise ValueError('Running policy filename is unavailable')
    return policy_file
