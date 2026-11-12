"""Restore both head joints or probe pitch through the manufacturer's interface.

Never starts controllers, changes modes, publishes body joints or clears faults.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
import rospy
import rosgraph
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
from sim2real_msg.msg import StateInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.shadow import Sensors
from deploy.geometry import quaternion_matrix_xyzw
from deploy.pro_head_model import ProHeadModel
from deploy.controller_state import factory_state_evidence
from deploy.head_profile import saved_head_targets,smooth_setpoint,validate_body_only_amp


def bounded_target(current,delta,lower,upper):
    values=np.asarray([current,delta,lower,upper],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Non-finite head command')
    if abs(delta)>.050001:raise ValueError('Each invocation is limited to 0.05 rad')
    target=current+delta
    # Stay inside the manufacturer's range with an additional 0.1 rad margin.
    lo,hi=max(-1.4,lower+.1),min(1.4,upper-.1)
    if not lo<=current<=hi or not lo<=target<=hi:raise ValueError('Head target outside bounded work range')
    return float(target)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    motion=ap.add_mutually_exclusive_group(required=True)
    motion.add_argument('--delta',type=float)
    motion.add_argument('--restore-saved-pose',action='store_true')
    ap.add_argument('--allow-running',action='store_true',help='Saved-pose restore only, with validated body-only AMP policy')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--urdf',type=Path,default=ROOT/'reference/robot/PiPlusPro_SF_12L8A0G2H0W_Z_261010/urdf/PiPlusPro_SF_12L8A0G2H0W_Z_261010.urdf')
    args=ap.parse_args()
    if args.allow_running and not args.restore_saved_pose:ap.error('--allow-running requires --restore-saved-pose')
    if args.delta is not None and (not np.isfinite(args.delta) or abs(args.delta)>.050001):ap.error('delta must be finite and within +/-0.05 rad')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rospy.init_node('competition_head_pitch_step',anonymous=True,disable_signals=True)
    sensors=Sensors();fsm={};control={};running_contract={};lock=threading.Lock();sent=0;publisher=None;report={}
    def on_fsm(msg):
        if msg._connection_header.get('callerid')!='/sim2real_master_node':return
        with lock:
            if msg.data!=5:control['child_state']=None
            fsm.update(time=time.monotonic(),state=msg.data)
    def on_control_state(msg):
        if msg._connection_header.get('callerid')!='/sim2real_master_node':return
        with lock:control.update(child_state=msg.state,source='live_factory_StateInfo',policy_name=msg.policy_name,policy_file_name=msg.policy_file_name)
    sub=rospy.Subscriber('/fsm_state',Int32,on_fsm,queue_size=5)
    state_sub=rospy.Subscriber('/sim2real_state_info',StateInfo,on_control_state,queue_size=5)
    def snapshot(require_standby=False):
        now=time.monotonic()
        with lock:state=dict(fsm)
        if state.get('state')!=5 or now-state.get('time',0)>.3:raise RuntimeError('Factory Default mode is not fresh/active')
        with lock:current_control=dict(control)
        child=current_control.get('child_state')
        if require_standby:
            if child=='running' and args.allow_running:
                if (not running_contract or current_control.get('policy_name')!='amp'
                        or current_control.get('policy_file_name')!=running_contract.get('policy_file')):
                    raise RuntimeError('Running head restore policy contract is not verified')
            elif child!='standby':raise RuntimeError('Factory child state must be confirmed standby; Default selection alone is insufficient')
        imu,joints,head=sensors.snapshot(now)
        if len(joints)!=20 or any(now-v[0]>.3 or not np.isfinite(v[1:]).all() for v in joints.values()):raise RuntimeError('Incomplete/stale body feedback')
        if len(head)!=2 or any(now-v[0]>.3 or not np.isfinite(v[1:]).all() for v in head.values()):raise RuntimeError('Incomplete/stale head feedback')
        return imu,joints,head
    try:
        time.sleep(1.)
        imu,body,head=snapshot()
        pubs,subs,_=rosgraph.Master(rospy.get_name()).getSystemState()
        subscribers=dict(subs).get('/pi_plus_absolute',[])
        if '/sim2real_master_node' not in subscribers:raise RuntimeError('Factory named-joint receiver unavailable')
        others=dict(pubs).get('/pi_plus_absolute',[])
        if others:raise RuntimeError('Another joint command publisher is present: '+str(others))
        import yaml
        cfg=yaml.safe_load(Path('/home/nvidia/sim2real/install/share/sim2real/robot_config/pi_plus-S-12L8A0G2H0W/pd.yaml').read_text())
        i=cfg['joint_names'].index('head_pitch_joint')
        start=head['head_pitch_joint'][1]
        model=ProHeadModel(args.urdf)
        model_limits=model.limits['head_pitch_joint']
        report.update(initial_pitch=start,initial_yaw=head['head_yaw_joint'][1],
                      factory_limits=[cfg['lower'][i],cfg['upper'][i]],
                      supplied_urdf_pitch_limits=list(model_limits),
                      urdf_path=str(args.urdf))
        log=Path('/home/nvidia/.ros/log/latest/rosout.log')
        evidence=factory_state_evidence(log.read_text())
        with lock:
            if 'child_state' not in control:control.update(child_state=evidence['child_state'],source='factory_rosout_log')
            report['controller_state_evidence']=dict(evidence,**{'observed_child_state':control.get('child_state'),'source':control.get('source')})
        if evidence['mode']!='ExecDefault':raise RuntimeError('Factory log mode does not match active Default mode')
        if args.allow_running:
            import hashlib
            policy_path=Path('/home/nvidia/action_library/policy/amp/amp.yaml')
            policy_data=policy_path.read_bytes()
            policy_file=validate_body_only_amp(yaml.safe_load(policy_data),body.keys())
            running_contract.update(policy_file=policy_file,config_sha256=hashlib.sha256(policy_data).hexdigest())
            report['running_policy_contract']=dict(running_contract)
            report['running_body_motion_expected']=True
        snapshot(require_standby=True)
        # The supplied mechanical model is narrower than the factory PD YAML.
        # Do not continue the old, wider-range probe if feedback is inconsistent.
        model.check_angles({n:v[1] for n,v in head.items()},margin=.03)
        lower=max(cfg['lower'][i],model_limits[0])
        upper=min(cfg['upper'][i],model_limits[1])
        if args.restore_saved_pose:
            pose=json.loads((ROOT/'deploy/task_head_pose.json').read_text())
            names=['head_pitch_joint','head_yaw_joint']
            starts={name:head[name][1] for name in names}
            limits={}
            for name in names:
                index=cfg['joint_names'].index(name)
                limits[name]=(max(cfg['lower'][index],model.limits[name][0]),
                              min(cfg['upper'][index],model.limits[name][1]))
            targets,expected=saved_head_targets(pose,starts,limits)
            profile_starts=dict(starts)
            for name in names:
                if abs(starts[name]-expected[name])<=.015:
                    # An axis already at its feedback reference keeps the
                    # saved holding command, including the measured PD bias.
                    profile_starts[name]=targets[name]
            # One shared duration synchronizes both cubic trajectories.
            # Each joint starts/stops at zero speed and peaks at <= 0.1 rad/s.
            ramp_duration=max(1.,1.5*max(abs(targets[name]-profile_starts[name]) for name in names)/.1)
            need_motion=any(abs(starts[name]-expected[name])>.015 for name in names)
            report.update(action='restore_saved_pose_continuous',
                          expected_feedback_pitch=expected['head_pitch_joint'],
                          expected_feedback_yaw=expected['head_yaw_joint'],
                          target_yaw=targets['head_yaw_joint'],head_joint_limits=limits,
                          profile_start_rad=profile_starts,
                          ramp_duration_s=ramp_duration,maximum_profile_speed_rad_s=.1)
        else:
            target=bounded_target(start,args.delta,lower,upper)
            names=['head_pitch_joint']
            starts={'head_pitch_joint':start}
            targets={'head_pitch_joint':target}
            expected=dict(targets);ramp_duration=1.;need_motion=True
        initial_body={n:v[1] for n,v in body.items()}
        gravity=quaternion_matrix_xyzw(imu[1]).T@np.array([0.,0.,-1.])
        report.update(initial_pitch=start,target_pitch=targets['head_pitch_joint'],
                      initial_yaw=head['head_yaw_joint'][1],named_joints=names)
        if args.execute and need_motion:
            publisher=rospy.Publisher('/pi_plus_absolute',JointState,queue_size=1)
            deadline=time.monotonic()+2
            while publisher.get_num_connections()==0:
                if time.monotonic()>deadline:raise RuntimeError('Joint subscriber connection timeout')
                time.sleep(.02)
            begun=time.monotonic()
            while time.monotonic()-begun<ramp_duration+1.:
                now=time.monotonic();current_imu,current_body,current_head=snapshot(require_standby=True)
                with lock:running=control.get('child_state')=='running'
                if not running and max(abs(v[1]-initial_body[n]) for n,v in current_body.items())>.08:raise RuntimeError('Body moved; head step stopped')
                g=quaternion_matrix_xyzw(current_imu[1]).T@np.array([0.,0.,-1.])
                if np.arccos(np.clip(g@gravity,-1,1))>np.deg2rad(4):raise RuntimeError('Body tilt changed; head step stopped')
                if args.restore_saved_pose:
                    positions=[smooth_setpoint(profile_starts[name],targets[name],now-begun,ramp_duration) for name in names]
                else:
                    if abs(current_head['head_yaw_joint'][1]-report['initial_yaw'])>.04:raise RuntimeError('Head yaw changed unexpectedly')
                    positions=[start+(target-start)*min(1.,(now-begun)/ramp_duration)]
                for name,q in zip(names,positions):
                    if abs(current_head[name][1]-q)>.10:raise RuntimeError('Head joint tracking error: '+name)
                msg=JointState();msg.header.stamp=rospy.Time.now();msg.name=names;msg.position=positions
                publisher.publish(msg);sent+=1;time.sleep(.05)
            time.sleep(.2)
        _,_,head=snapshot(require_standby=True);report['final_pitch']=head['head_pitch_joint'][1]
        report['final_yaw']=head['head_yaw_joint'][1]
        if args.execute and abs(report['final_pitch']-expected['head_pitch_joint'])>.025:raise RuntimeError('Head did not reach the requested target')
        if args.execute and args.restore_saved_pose and abs(report['final_yaw']-expected['head_yaw_joint'])>.04:raise RuntimeError('Head yaw did not reach the requested target')
        report['status']=('saved_pose_restored' if args.restore_saved_pose else 'step_completed') if args.execute else 'preflight_passed_no_motion'
    except Exception as exc:report.update(status='aborted',error=str(exc))
    finally:
        report['messages_sent']=sent
        report['body_joint_commands_sent']=False
        report['controller_mode_changed']=False
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        if publisher is not None:publisher.unregister()
        sub.unregister();state_sub.unregister();rospy.signal_shutdown('head step finished')
    return 0 if report['status']!='aborted' else 2


if __name__=='__main__':raise SystemExit(main())
