"""Re-sense MuJoCo using the candidate release and a recorded rt10 onboard schedule."""
from pathlib import Path
import argparse
from dataclasses import replace
from collections import Counter
import csv
import hashlib
import json
import sys
import time

import os
import shutil
from rgbd_view import (FrameWriter, millimeters, sensor_panel, LABEL, SMALL,
                       PANEL_WIDTH, PANEL_HEIGHT)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('case')
parser.add_argument('--seed',type=int,default=1,help='reproducible noise in simulated IMU, joint and depth measurements')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--limit',type=float)
parser.add_argument('--timing',type=Path,help='alternate rt10 navigation run directory with guard and camera logs')
parser.add_argument('--gui',action='store_true')
parser.add_argument('--hold',action='store_true')
parser.add_argument('--images',action='store_true')
parser.add_argument('--set',action='append',default=[],metavar='section.key=value',help='override tuning for this run')
parser.add_argument('--baseline',type=Path,default=Path(__file__).with_name('baseline_tuning.json'),help='tuning to diff the ablation against')
parser.add_argument('--no-target',action='store_true',help='do not draw the lane target in the 3D view')
parser.add_argument('--no-launch',action='store_true',help='disable the RT10 launch kick (RT16 has none of its own)')
parser.add_argument('--floor-only-for',type=float,default=0.,
                    help='Seconds of featureless floor frames after the first lateral box request')
args=parser.parse_args()
RESTART=False
TEST_DIR=args.out.resolve()
TEST_DIR.mkdir(parents=True,exist_ok=False)
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reference/rt11_sim'
OUT=TEST_DIR/'source'
for folder in ('navigation','deploy'):
    chosen=(args.source.resolve()/folder)
    shutil.copytree(chosen,OUT/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','realsense_stream'))

sys.path[:0] = [str(OUT), str(ROOT)]
import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from deploy.shared_navigation import TurnCutoff, frame_head_reasons
try:
    from deploy.shared_navigation import PublishedPlan
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class PublishedPlan:
        """Stand-in for packages that dropped the RT13 plan publisher."""
        vector: tuple = (0., 0., 0.)
        deadline: float = 0.
        pulse: object = None
        reason: str = 'initializing'
        scene: object = None
        frame_time: object = None
        fresh: bool = False
        box_servo: object = None

        def evaluate(self, stamp, imu, body_from_imu, cutoff):
            return self.vector, self.reason, self.fresh
from deploy.geometry import quaternion_matrix_xyzw
from navigation.runtime import OnboardNavigation, CameraFrame, core_hashes
import navigation.runtime as _nav_runtime
if not hasattr(_nav_runtime,'NavigationRuntime'):
    # run_visual imports the RT13-only runtime at module level; only its world
    # builder is reused here.
    _nav_runtime.NavigationRuntime=object
from navigation.command import AMPCommandAdapter
from navigation.environment import verify_dependencies
from navigation import params as nav_params
from sim.navigation import Track
from sim.robot import JOINTS, Robot
from sim.scoring import Referee
from sim.sensors import SensorRig
from sim.rt1_control import RecordedWorker
from scripts.run_visual import visual_world
# run_visual and the device entry point prepend their own import roots.
sys.path[:0] = [str(OUT), str(ROOT)]
cv2.setNumThreads(1)
RUN = args.timing.resolve() if args.timing else BASE
HW = RUN / 'vision'
LOAD = lambda p: json.loads(p.read_text())
HASH = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
recorded_core = LOAD(HW/'vision_summary.json')['navigation_core_sha256']
parents = [BASE/'rt10_manifest.json', BASE/'rt11_manifest.json']
release = next(manifest for manifest in map(LOAD, parents) if manifest['navigation_core_sha256']==recorded_core)
config = LOAD(HW/'config.json')
intrinsics_path = HW/'camera_intrinsics.json'
if not intrinsics_path.exists():
    intrinsics_path = BASE/'vision/camera_intrinsics.json'
camera = dict(LOAD(intrinsics_path), processing_scale=config['perception_scale'], rectify=True)
assert config == LOAD(OUT/'deploy/pro_model_config.json')
angle = np.deg2rad(110.)
cases = {'full_01':[.55,-1.25,0.], 'boxes_center':[2.80,1.25,np.pi],
         'boxes_oblique':[2.62,1.40,np.pi-.20], 'small_entry':[1.,1.25,np.pi],
         'small_inside':[1.25*np.cos(angle),1.25*np.sin(angle),angle+np.pi/2],
         'small_runup':[2.8,1.25,np.pi+.14], 'sharp_entry':[1.60,-1.25,.14],
         'small_entry_offset':[1.,1.25,np.pi+.14], 'full_01_offset':[.55,-1.25,.14]}
# Enter the large bend near the upper join, then measure the exit onto the straight.
angle_exit=np.deg2rad(116.)
for name in ('large_exit','large_exit_pending'):
    cases[name]=[5.+2.25*np.cos(angle_exit),2.25*np.sin(angle_exit),angle_exit+np.pi/2]
def source_check():
    assert release['navigation_core_sha256']==recorded_core
    loaded = {}
    for name,mod in list(sys.modules.items()):
        if name.split('.')[0] not in ('navigation','deploy') or not getattr(mod,'__file__',None):
            continue
        p = Path(mod.__file__).resolve()
        k = str(p.relative_to(OUT))
        assert p.is_file(), k
        loaded[name] = dict(path=str(p),sha256=HASH(p))
    return loaded

def timeline(duration):
    rows=[json.loads(l) for l in (HW/'telemetry.jsonl').open()]
    cmds=LOAD(HW/'commands.json')
    guards=[json.loads(l) for l in (RUN/'guard.jsonl').open()]
    stops=[json.loads(l) for l in (HW/'stops.jsonl').open()]
    # local_lane_time is absolute monotonic, t-age is the same capture relative to vision start.
    captures=np.array([r['t']-r['camera_host_age_s'] for r in rows])
    offsets=[r['local_lane_time']-r['t']+r['camera_host_age_s'] for r in rows if r['local_lane_time']]
    if offsets:
        offset=float(np.median(offsets))
    else:
        startup=next(v for v in stops if v['frame_relation']=='subsequent_startup_reference')
        offset=float(startup['frame_timestamp_s']-captures[0])
    image_stamps=[r['tracking_frame_time'] for r in rows if r['tracking_frame_time'] is not None]
    assert np.isfinite(offset) and max(np.min(abs(captures+offset-stamp)) for stamp in image_stamps)<.0001
    utc_offset=float(np.median([s['time']-s['monotonic_s'] for s in stops if s['event']=='stop']))
    events=[]
    previous=None
    for i,r in enumerate(rows):
        capture=r['t']-r['camera_host_age_s']
        events.extend([(capture,0,i,r['imu_host_delta_s']),
                       (r['t'],1,i,r['tracking_frame_time']!=previous)])
        previous=r['tracking_frame_time']
    i=0
    for c in cmds:
        idx=i if c['has_camera_frame'] else -1
        head_rejected=bool(set(c.get('inhibit_reasons',())) &
                           {'fixed_head_reference_required','head_pose_not_at_calibrated_reference'})
        events.append((c['t'],2,idx,head_rejected))
        if idx>=0:
            events.append((max(c['t'],rows[i]['t']+c['computation_ms']/1000),3,i,None))
            i+=1
    assert i==len(rows)
    end=cmds[-1]['t']
    pub_times=set()
    for g in guards:
        t=g['t']-offset
        if not 0<=t<=end or g['navigation_stamp_ns'] is None:continue
        pub=g['navigation_stamp_ns']/1e9-utc_offset-offset
        if pub<0:continue
        assert pub<=t+.003
        pub=min(t,pub)
        pub_times.add(pub)
        age=g['policy_obs_age_s']
        events.append((t,5,pub,age))
    assert len(pub_times)>end*20, "Insufficient recorded guard publications"
    events.extend((v,4,v,None) for v in pub_times)
    events.sort()
    # Do not replay operator termination or restart navigation. Continue the measured
    # steady section when a simulated case outlasts the recording.
    first=rows[0]['t']-rows[0]['camera_host_age_s']
    period=end-first+1/camera['fps']
    output=[]
    for cycle in range(1+int(np.ceil(duration/period))):
        for t,k,i,arg in events:
            if cycle and (t<first or (k==5 and i<first)): continue
            output.append((2+t+cycle*period,k,(cycle,i),arg))
    return sorted(output), dict(run=("20261111_094517Z_ac10347e" if RUN == BASE else RUN.parent.name),vision_duration_s=end,frames=len(rows),plans=len(cmds),
        first_capture_s=first, steady_repeat_period_s=period,vision_monotonic_origin=offset,epoch_minus_monotonic_s=utc_offset)

def run(case, gui, hold):
    loaded=source_check()
    out=TEST_DIR
    track=Track('corner')
    spawn=cases[case]
    start=track.config['starts'][0]
    assert cases['full_01']==[*start['position'],np.deg2rad(start['yaw_degrees'])]
    empty=case.startswith(('small','large_exit')) or case=='sharp_entry'
    model,xml=visual_world(out,start,no_obstacles=empty,camera_profile=camera,bend_profile='corner')
    (out/'scene.xml').write_text(xml)
    robot=Robot(model,spawn,from_standby=True)
    rig=SensorRig(model,seed=args.seed,noise=True,camera_profile=camera)
    assert np.allclose(config['camera_position_body_m'],rig.camera_position,atol=1e-7)
    assert np.allclose(config['camera_rotation_optical_to_body'],rig.camera_rotation,atol=1e-7)
    referee=Referee(robot,track)
    assert len(referee.obstacles)==(0 if empty else 3)
    nav_params.load(OUT/'deploy/tuning.json')
    for item in args.set:
        key, _, val = item.partition('=')
        sec, _, name = key.partition('.')
        if not sec or not name:
            raise SystemExit(f'--set expects section.key=value, got {item!r}')
        try:
            value = float(val)
        except ValueError:
            value = val
        nav_params._P.setdefault(sec, {})[name] = value
    if args.set:
        print('overrides:', ', '.join(args.set))
    hud_params = ('off %.2f  spd %.2f  step %.1fs  side %.2f  vx %.2f  half %.2f' % (
        nav_params.get('avoid','strafe_offset'), nav_params.get('avoid','strafe_speed'),
        nav_params.get('avoid','strafe_s'), nav_params.get('avoid','follow_side'),
        nav_params.get('avoid','follow_vx'), nav_params.get('avoid','body_half')))
    print('tuning loaded from', OUT/'deploy/tuning.json')
    print('avoid:', hud_params, flush=True)
    def flat(d, pre=''):
        out={}
        for k,v in d.items():
            if isinstance(v,dict): out.update(flat(v,pre+k+'.'))
            else: out[pre+k]=v
        return out
    active=flat(nav_params.as_dict())
    baseline=flat(json.loads(args.baseline.read_text())) if args.baseline.exists() else {}
    ablations=[(k,baseline[k],active[k]) for k in active if k in baseline and active[k]!=baseline[k]]
    ablations.sort(key=lambda a:a[0])
    ablate_text=' '.join(f'{k.split(".")[-1]} {b:g}->{a:g}' for k,b,a in ablations)
    print('ABLATION vs', args.baseline.name, ':', ablate_text or '(none)', flush=True)
    for k,b,a in ablations:
        print(f'   {k}: {b} -> {a}')

    def build_table(rows):
        width, row_h = 440, 22
        height = 34+row_h*(len(rows)+1)
        img = Image.new('RGB', (width, height), '#0f1a20')
        pen = ImageDraw.Draw(img)
        pen.text((8, 6), '参数消融  RT16 → RT18', font=LABEL, fill='white')
        pen.line((6, 30, width-6, 30), fill='#2c414c')
        pen.text((10, 34), '参数', font=SMALL, fill='#9fb3bb')
        pen.text((300, 34), 'RT16', font=SMALL, fill='#9fb3bb')
        pen.text((370, 34), 'RT18', font=SMALL, fill='#9fb3bb')
        for i, (name, base, cur) in enumerate(rows):
            y = 34+row_h*(i+1)
            pen.text((10, y), name, font=SMALL, fill='white')
            pen.text((300, y), f'{base:g}', font=SMALL, fill='#c8d6db')
            pen.text((370, y), f'{cur:g}', font=SMALL, fill='#ffd479')
        return img

    table_img = build_table([(k.split('.')[-1], b, a) for k, b, a in ablations]) if ablations else None

    def show_overlays(view, panel, table):
        vp = view.viewport
        if vp is None or vp.width < 6 or vp.height < 24:
            return
        images = []
        if panel is not None:
            scale = min(vp.width*.27/PANEL_WIDTH, (vp.height-12)/PANEL_HEIGHT, 1.4)
            w, h = max(1, int(PANEL_WIDTH*scale)), max(1, int(PANEL_HEIGHT*scale))
            px = np.asarray(panel.resize((w, h), Image.Resampling.BILINEAR))
            images.append((mujoco.MjrRect(vp.left+vp.width-w-6, vp.bottom+6, w, h), px))
        if table is not None:
            tw, th = table.size
            images.append((mujoco.MjrRect(vp.left+6, vp.bottom+6, tw, th), np.asarray(table)))
        if images:
            view.set_images(images)
    nav=OnboardNavigation(rig.width,rig.height,rig.intrinsics,config['camera_position_body_m'],
                          config['camera_rotation_optical_to_body'],LOAD(RUN/'launch.json')['speed_m_s'],config['observation_lease_s'])
    if hasattr(nav,'worker'):
        nav.worker.shutdown()
        nav.worker=RecordedWorker()
    guard,cutoff=AMPCommandAdapter(),TurnCutoff()
    secs=LOAD(RUN/'launch.json')['maximum_duration_s'] if case.startswith('full_01') else 90.
    secs=args.limit or secs
    events,timing=timeline(secs)
    loco=robot.locomotion_metadata()
    for name,digest in release['factory_contract']['sha256'].items():
        assert loco['source_sha256'][name]==digest==LOAD(HW/'preflight.json')['factory_sha256'][name]
    evidence=dict(parent_release=release['release_id'],candidate=args.source.name,core_sha256=core_hashes(),loaded=loaded,dependencies=verify_dependencies(),camera=camera,camera_intrinsics_source=str(intrinsics_path),
                  timing=timing,recorded_run=str(RUN),locomotion=loco,seed=args.seed,settle_s=2,time_limit_after_settle_s=secs,
                  track_length_m=track.length,spawn=spawn,obstacles=[] if empty else track.config['obstacles'],
                  simulator_hashes={n:HASH(ROOT/n) for n in ['sim/robot.py','sim/device_amp.py','sim/sensors.py','sim/navigation.py','sim/scoring.py','sim/rt1_control.py','scripts/run_visual.py','scene_config.json','output/athletics_corner/geometry.json','output/athletics_corner/mujoco/field.xml']},
                  runner_hash=HASH(Path(__file__)),mujoco=mujoco.__version__)
    (out/'setup.json').write_text(json.dumps(evidence,indent=2))
    view=None
    if gui:
        import mujoco.viewer as viewer
        def on_key(code):
            global RESTART
            if code in (82, 114):   # R / r
                RESTART=True
        view=viewer.launch_passive(model,robot.data,show_left_ui=False,show_right_ui=False,
                                   key_callback=on_key)
        with view.lock():
            view.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
            view.cam.trackbodyid=robot.base
            view.cam.lookat[:]=robot.data.xpos[robot.base]
            view.cam.distance,view.cam.elevation,view.cam.azimuth=2.5,-32.,85.
        view.sync()
        print('MuJoCo GUI open',flush=True)
    def draw_target(view, robot, nav):
        """Green dots on the lane fit, red dot at the look-ahead target."""
        scn = view.user_scn
        scn.ngeom = 0
        if args.no_target or nav.lane is None or nav.support is None:
            return
        near, far = nav.support
        near = max(.15, float(near)); far = max(near, float(far))
        look = float(np.clip(near+.25, near, far))
        x, y, yaw = robot.pose
        c, s = np.cos(yaw), np.sin(yaw)
        def world(px, py):
            return np.array([x+c*px-s*py, y+s*px+c*py, .02])
        def dot(px, py, radius, rgba):
            nonlocal scn
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([radius, radius, radius]),
                                world(px, py), np.eye(3).reshape(-1),
                                np.array(rgba, dtype=np.float32))
            scn.ngeom += 1
        if nav.tracking_control:
            # Avoid mode: orange markers on the ride line, blue on the box.
            side = nav_params.get('avoid','follow_side')
            sign = 1. if (nav.box is not None and nav.box[1] <= 0) else -1.
            for d in np.linspace(.30, .80, 6):
                dot(float(d), sign*side, .012, (1., .55, .05, .95))
            if nav.box is not None:
                dot(float(nav.box[0]), float(nav.box[1]), .03, (.2, .4, 1., 1.))
            return
        for d in np.linspace(near, min(far, near+.7), 8):
            dot(float(d), float(np.polyval(nav.lane, d)), .012, (.1, .9, .3, .9))
        dot(look, float(np.polyval(nav.lane, look)), .03, (1., .15, .1, 1.))

    recorder = FrameWriter(out)
    panel = None
    panel_key = None
    capture_stamp = None
    frames,seen,published={},{},{}
    state_rows=[]; poses=[]; times=[]; decisions=[]
    telemetry=[];feedback_history=[]
    progress=0.;prev_s=track.s[track.tree.query(robot.pose[:2])[1]]
    passed_boxes=[]; entered=False; box_area=False; entered_sharp=False
    leg_samples=[]; next_leg_time=0.
    index=0; latest=None; pulse=None; desired=np.zeros(3); raw_cmd=np.zeros(3)
    active_plan=PublishedPlan()
    body_from_imu=np.array(config["body_from_imu"])
    pending_publications={}
    deadline=None
    if (OUT/'navigation/motion_deadline.py').exists():
        from navigation.motion_deadline import MotionDeadline
        deadline=MotionDeadline()
    motion_start=last_motion=None
    launch_stamp=None
    launch_direction=1.
    longest_zero=0.
    pending_site_start=None
    zero_publications=set()
    box_steps=[]
    last_feedback_time=None
    prev_request=np.zeros(3)
    result='time_limit'
    floor_fault_start=None
    progress_log=0.
    robot_geoms=np.flatnonzero((model.body_rootid[model.geom_bodyid]==robot.base)&(model.geom_contype!=0))
    def rearm_label(*args):
        pass # rt10 inhibit updates diagnostic labels and retains the published vector.
    try:
        while robot.data.time<secs+2-1e-7:
            t=float(robot.data.time)
            measurement=rig.proprioception(robot.data)
            q=measurement.orientation_wxyz
            attitude=quaternion_matrix_xyzw(np.r_[q[1:],q[0]]) @ np.array(config['body_from_imu']).T
            if hasattr(nav,'update_legs') and t+1e-8>=next_leg_time:
                leg_samples.append((t,dict(zip(JOINTS,measurement.joints)),attitude.copy()))
                next_leg_time+=1/18
            feedback_history.append(robot.device.command.copy())
            while index<len(events) and events[index][0]<=t+1e-8:
                stamp,kind,key,arg=events[index];index+=1
                if (kind in (1,2) or (kind==4 and hasattr(nav,'publish_velocity'))) and leg_samples:
                    due=[sample for sample in leg_samples if sample[0]<=stamp]
                    for leg_sample in due:nav.update_legs(*leg_sample)
                    leg_samples=leg_samples[len(due):]
                if kind==0:
                    latest=rig.camera(robot.data)
                    if (args.floor_only_for and floor_fault_start is None and nav.reason.startswith('box_')
                            and getattr(getattr(nav,'crates',None),'index',0)==0 and abs(prev_request[1])>.06):
                        floor_fault_start=stamp
                    if floor_fault_start is not None and stamp-floor_fault_start<args.floor_only_for:
                        # Preserve valid depth and the real simulated camera attitude, but
                        # remove all visual landmarks and obstacles from the sensor packet.
                        rays=nav.local.rays@nav.local.rotation32.T@attitude.T
                        origin=nav.local.position32@attitude.T
                        with np.errstate(divide='ignore',invalid='ignore'):
                            flat=(-robot.data.xpos[robot.base,2]-origin[2])/rays[...,2]
                        flat[(flat<.15)|(flat>4.)]=np.nan
                        latest=CameraFrame(stamp,np.full_like(latest.rgb,90),flat.astype('float32'))
                    capture_stamp=stamp
                    recorder.capture(stamp,t,latest.rgb,latest.depth)
                    head={name:(stamp,float(robot.data.joint(name).qpos[0])) for name in config['head_reference_rad']}
                    frames[key]=(CameraFrame(stamp,latest.rgb,latest.depth),attitude.copy(),stamp+arg,head)
                elif kind==1:
                    frame,orientation,imu_t,head=frames[key]
                    if hasattr(nav,'worker'):
                        nav.worker.available=arg
                    nav.set_attitude(stamp,attitude)
                    projection=frame_head_reasons(config,head,frame.time)
                    if stamp-frame.time>.2:projection=['stale_capture']
                    obs=None
                    if not projection:
                        if (case=='large_exit_pending' and pending_site_start is None and
                                nav.lane_time is not None and robot.pose[0]<3.90):
                            pending_site_start=stamp
                        if pending_site_start is not None and stamp-pending_site_start<6.:
                            # Reproduce six seconds of unresolved map evidence at the join.
                            # A live false hit invalidates a cell's previous clearance count.
                            if hasattr(nav,'tracked') and hasattr(nav,'unconfirmed_sites'):
                                pos=nav.tracked.xy
                                point=np.r_[pos+[0.,.9],.10]
                                key_xy=tuple(np.floor(point[:2]/.04).astype(int))
                                nav.unconfirmed_sites[key_xy]=(point,0,stamp)
                        obs=nav.observe(frame,orientation,imu_t,rearm_label,defer_mapping=True)
                    elif hasattr(nav,'invalidate_box_projection'):
                        nav.invalidate_box_projection(frame.time)
                    seen[key]=(stamp,obs,projection)
                elif kind==2:
                    if arg and hasattr(nav,'invalidate_box_projection'):
                        nav.invalidate_box_projection(stamp)
                    nav.set_attitude(stamp,attitude)
                    nav.command(stamp,True,cutoff.measurement)
                    turn=nav.lane_turn
                    next_pulse=(turn.started,nav.initial_yaw+turn.origin,turn.direction) if turn.started is not None and nav.reason=='ready' else None
                    next_vector=nav.desired.copy()
                    # RT16 dropped the RT10/RT13 launch pulse. From a centred
                    # start it asks for wz=0 and the factory AMP never leaves
                    # stand (probe: raw [0.4,0,0] stands, [0.4,0,0.6] walks).
                    # Re-add the proven pulse for 0.7 s at the desired level;
                    # --no-launch disables this driver aid.
                    if not args.no_launch:
                        if launch_stamp is None and next_vector[0] > .05:
                            launch_stamp = stamp
                            launch_direction = -1. if next_vector[2] < 0 else 1.
                        if launch_stamp is not None and 0. <= stamp-launch_stamp < .70:
                            next_vector = np.array([.22, 0., .24*launch_direction])
                    next_refinement=nav.freeze_publish_velocity() if hasattr(nav,'freeze_publish_velocity') else None
                    plan=PublishedPlan(tuple(next_vector), nav.deadline, next_pulse, nav.reason,
                                       nav.scene, nav.last_frame_time, key[1]>=0, next_refinement)
                    if deadline is not None:
                        side=nav.crates.recovery_direction() if nav.crates.frame is not None and not nav.crates.done else 0.
                        if not side and nav.steering_heading is not None:
                            side=np.sign(nav.steering_heading)
                        plan=replace(plan,recovery_direction=side)
                    # commands[].t records the instant just before command().
                    # With an image, the measured end of computation is known.
                    if key[1] < 0:
                        active_plan=plan
                        desired=next_vector
                    else:
                        pending_publications[key]=plan
                    rs=getattr(nav,'raised_seen',None)
                    if rs is None or len(rs)==0:
                        stats=(0,'','','')
                    else:
                        a=np.asarray(rs,dtype=float)
                        stats=(len(a),round(float(np.median(a[:,0])),2),
                               round(float(np.median(a[:,1])),2),round(float(np.abs(a[:,1]).max()),2))
                    decisions.append((stamp,nav.reason,*next_vector,
                        getattr(nav,'ride_count',None), getattr(nav,'ride_y',None),
                        getattr(nav,'ride_side',None), getattr(nav,'ride_now',None),
                        getattr(nav,'scene',None),*stats))
                elif kind==3:
                    active_plan=pending_publications.pop(key)
                    desired=np.array(active_plan.vector)
                    start_t,obs,projection=seen.pop(key)
                    frame,*_=frames.pop(key)
                    if obs is not None:
                        nav.dispatch_mapping(start_t,stamp,1/camera['fps'])
                    diag=nav.diagnostic()
                    state_rows.append(dict(t=stamp,truth=robot.pose.tolist(),progress=progress,boxes=None if obs is None else obs.obstacles.tolist(),
                                           lane=None if obs is None or obs.lane is None else obs.lane.tolist(),projection=projection,**diag))
                    if args.images and obs is not None:
                        images=out/'samples'; images.mkdir(exist_ok=True)
                        Image.fromarray(frame.rgb).save(images/f'{stamp:08.3f}.jpg')
                    if len(state_rows)==1 or len(state_rows)%12==0:
                        Image.fromarray(frame.rgb).save(out/'last_sample.jpg')
                elif kind==4:
                    # Use the release's shipping publisher, sampled with a fresh
                    # simulated AHRS packet at the guard's measured publication.
                    vector,_,_=active_plan.evaluate(stamp,(stamp,np.r_[q[1:],q[0]]),
                                                    body_from_imu,cutoff)
                    published[key]=np.asarray(vector).copy()
                    acknowledge=getattr(active_plan.box_servo,'published',None)
                    if acknowledge is not None:
                        acknowledge(stamp,published[key])
                        servo=nav.crates
                        step=getattr(getattr(servo,'steps',None),'active',None)
                        if not np.any(published[key]):
                            zero_publications.add(stamp)
                        elif (step is not None and hasattr(servo.frame,'paint_stamp') and
                              (not box_steps or box_steps[-1]['number']!=servo.steps.count)):
                            frame=servo.frame
                            zero=servo.steps.zero_stamp
                            paint=getattr(frame,'paint_stamp',(None,None))[int(servo._pass_side(frame)>0)]
                            paint=float(paint) if paint is not None and np.isfinite(paint) else None
                            axis=step['axis']
                            issues=[]
                            if box_steps and (zero not in zero_publications or
                                              not box_steps[-1]['sent']<zero<frame.stamp):
                                issues.append('missing_zero_or_new_capture')
                            if axis==1 and (paint is None or not 0<=frame.stamp-paint<=.50 or
                                            (zero is not None and paint<=zero)):
                                issues.append('missing_post_stop_paint')
                            depth=getattr(frame,'depth_status',None)
                            line_step=(servo.hold_reason=='line_follow_unseen_path' and
                                       (frame.depth_diagnostics or {}).get('reason')=='unseen_swept_path' and
                                       paint is not None and 0<=frame.stamp-paint<=.50 and
                                       (zero is None or paint>zero))
                            if axis==0 and depth not in ('depth_clear','depth_limited') and not line_step:
                                issues.append('missing_forward_evidence')
                            box_steps.append(dict(number=servo.steps.count,axis=axis,sent=stamp,zero=zero,
                                                  capture=frame.stamp,paint=paint,depth=depth,line_step=line_step,issues=issues))
                else:
                    prev_request=published[key]
                    feedback=None
                    if arg is not None:
                        ts=stamp-arg
                        if last_feedback_time is None or abs(ts-last_feedback_time)>1e-5:
                            feedback=feedback_history[min(len(feedback_history)-1,max(0,int(ts/robot.dt)))]
                            last_feedback_time=ts
                    raw_cmd=guard.update(prev_request,feedback,True)
                    active=active_plan.reason!='initializing'
                    if active and motion_start is None:
                        motion_start=stamp
                    if deadline is not None:
                        raw_cmd=deadline.output(stamp,raw_cmd,active,
                            getattr(active_plan,'recovery_direction',0.))
                        deadline.published(stamp,raw_cmd)
                    if motion_start is not None:
                        since=motion_start if last_motion is None else last_motion
                        longest_zero=max(longest_zero,stamp-since)
                        if np.any(raw_cmd):
                            last_motion=stamp
            recorder.plan(t,active_plan.reason)
            robot.step(raw_cmd,referee.contacts)
            referee.boundary()
            x,y,yaw=robot.pose
            s=track.s[track.tree.query([x,y])[1]]
            progress+=(s-prev_s+track.length/2)%track.length-track.length/2
            prev_s=s
            now=robot.data.time
            telemetry.append((now,x,y,yaw,progress,*desired,*prev_request,*raw_cmd,*robot.device.command))
            times.append(now); poses.append(robot.data.qpos.copy())
            if view is not None and view.is_running() and len(times)%4==1:
                view.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                'RT18 | '+case,
                                f'{now:.1f} s  {progress:.2f} m  {nav.reason}  |  '+hud_params))
                key = (capture_stamp, active_plan.reason)
                if panel_key != key:
                    panel = sensor_panel(latest.rgb if latest else None, millimeters(latest.depth) if latest else None, capture_stamp, active_plan.reason)
                    panel_key = key
                show_overlays(view, panel, table_img)
                draw_target(view, robot, nav)
                view.sync()
            entered=entered or (x<0 and y>0)
            entered_sharp=entered_sharp or (y<0 and x>=track.config['center_distance']-np.sqrt(track.config['large_radius']**2-track.config['small_radius']**2))
            if RESTART:
                result='restart';break
            if referee.reason:
                result=referee.reason;break
            if not empty and y>.90 and y<1.60 and x<2.85:
                box_area=True
                if len(passed_boxes)<3:
                    o=sorted(track.config['obstacles'],key=lambda b:-b['position'][0])[len(passed_boxes)]
                    far=o['position'][0]-.10
                    if x<far:
                        gm=robot_geoms
                        extent=robot.data.geom_xmat[gm].reshape(-1,3,3)[:,0,:]
                        bounds=model.geom_aabb[gm]
                        xmax=np.max(robot.data.geom_xpos[gm,0]+(extent*bounds[:,:3]+abs(extent)*bounds[:,3:]).sum(axis=1))
                        if xmax<far:passed_boxes.append(dict(id=o['id'],time_s=now,pose=[x,y,yaw]))
            if (case.startswith('full_01') and progress>=track.length and len(passed_boxes)==3 and entered
                    and x>=start['position'][0] and abs(y-start['position'][1])<=track.config['track_width']/2):
                result='lap_completed';break
            if case.startswith('boxes') and len(passed_boxes)==3 and entered and x>=.08 and y<-.9:
                result='boxes_and_small_exit';break
            if case.startswith('large_exit') and x<=2.35 and abs(y-1.25)<.18:
                result='large_exit_reached';break
            if case.startswith('small') and entered and x>=.08 and y<-.9:
                result='small_exit_reached';break
            if case=='sharp_entry' and x>=4.1 and y< -1.5:
                result='sharp_join_cleared';break
            if now>=progress_log:
                print(f'{case} t={now:.1f} progress={progress:.2f} xy=({x:.2f},{y:.2f}) {nav.reason} cmd={raw_cmd.round(2)}',flush=True); progress_log+=5
    finally:
        rig.close()
        nav.close(out/'floor_map.npz')
    header='t,x,y,yaw,progress,desired_vx,desired_vy,desired_wz,forward_vx,forward_vy,forward_wz,raw_vx,raw_vy,raw_wz,policy_vx,policy_vy,policy_wz'
    np.savetxt(out/'truth_and_commands.csv',telemetry,delimiter=',',header=header,comments='',fmt='%.7f')
    np.savez_compressed(out/'states.npz',t=times,qpos=poses)
    with (out/'frames.jsonl').open('w') as f:
        for r in state_rows: f.write(json.dumps(r)+'\n')
    with (out/'decisions.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['t','reason','vx','vy','wz','cells','ride_y','side','now','scene','n_raised','rx','ry','ry_max']);w.writerows(decisions)
    if latest is not None: Image.fromarray(latest.rgb).save(out/'last_camera.jpg')
    summary=dict(case=case,result=result,longest_zero_s=round(longest_zero,3),pending_site_added=pending_site_start is not None,time_s=times[-1],end_pose=robot.pose.tolist(),progress_m=progress,
                 max_progress_m=max(a[4] for a in telemetry),track_length_m=track.length,entered_sharp_join=bool(entered_sharp),
                 entered_box_area=bool(box_area),passed_boxes=passed_boxes,entered_small_bend=bool(entered),
                 referee=referee.reason,referee_time_s=referee.event_time,contact_detail=referee.event_detail,
                 overhang_ticks=referee.partial_foot_overhang_frames,frames=len(state_rows),box_observation_frames=sum(bool(r.get('boxes')) for r in state_rows),
                 reasons=dict(Counter(r['reason'] for r in state_rows)),last_reason=nav.reason,desired=desired.tolist(),last_raw=raw_cmd.tolist(),
                 vision_finished=bool(getattr(nav,'finished',False)),gui_opened=gui)
    if args.floor_only_for:
        assert floor_fault_start is not None, "Test did not exercise the first lateral request"
        end=floor_fault_start+args.floor_only_for
        window=[r for r in telemetry if floor_fault_start<=r[0]<end]
        assert window[-1][0]-window[0][0] > 1.
        moving=[r[0] for r in window if max(abs(v) for v in r[8:11])>.001]
        after_settle=[r for r in window if r[0]>floor_fault_start+1.]
        summary['floor_fault']=dict(start_s=floor_fault_start,end_s=end,
            last_requested_motion_after_fault_s=max(moving,default=floor_fault_start)-floor_fault_start,
            window_s=window[-1][0]-window[0][0],
            max_requested_after_one_second=max(abs(v) for r in after_settle for v in r[8:11]),
            displacement_after_one_second_m=float(np.linalg.norm(np.array(window[-1][1:3])-after_settle[0][1:3])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if box_steps:
        audit=dict(steps=box_steps,counts=dict(Counter(s['axis'] for s in box_steps)),
                   violations=sum(bool(s['issues']) for s in box_steps))
        (out/'step_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(summary),flush=True)
    recorder.close()
    if view is not None:
        if view.is_running():
            view.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,'rt16 | '+case+' | '+result,f'{times[-1]:.2f} s   {progress:.2f} m'))
            view.sync()
        while hold and view.is_running() and not RESTART: time.sleep(.25)
        view.close()
    return 'restart' if RESTART else 'done'

if __name__=='__main__':
    while True:
        RESTART=False
        if run(args.case,args.gui,args.hold)!='restart':
            break
