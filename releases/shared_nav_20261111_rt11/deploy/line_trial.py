"""Scoped real lane trial: original joystick translator + guarded sole publisher.

The factory controller and its policy stay running. The supervisor temporarily
restarts only joy_teleop using its unmodified installed script through a guarded
Publisher proxy, then restores the original command. No /joy publisher is added.
"""
import argparse,json,math,os,queue,runpy,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def proxy_role(args):
    import rospy,rosgraph,yaml
    from geometry_msgs.msg import Twist,TwistStamped
    from sensor_msgs.msg import Joy,Imu,JointState
    from std_msgs.msg import Int32,String,UInt8
    from sim2real_msg.msg import StateInfo,DebugData
    from deploy.line_follow_core import choose_command
    from deploy.head_profile import validate_body_only_amp
    from navigation.command import AMPCommandAdapter
    from navigation.motion_deadline import MotionDeadline
    original_publisher=rospy.Publisher;guard=[]

    from deploy.stop_archive import GuardStops
    class Guard:
        def __init__(self,real):
            self.real=real;self.lock=threading.RLock();self.cache={};self.begun=None;self.budget=0.;self.turn_budget=0.;self.last=time.monotonic();self.v=0.;self.omega=0.;self.last_command=[0.,0.,0.];self.deadline=self.last+args.duration+35.;self.baseline=None
            self.pub=original_publisher('/competition/line_guard',String,queue_size=10)
            self.policy=yaml.safe_load(Path('/home/nvidia/action_library/policy/amp/amp.yaml').read_text())
            self.turn_start=None;self.turn_completed=False
            self.shared_adapter=AMPCommandAdapter() if args.shared_navigation else None
            self.calibration=json.loads(args.config.read_text()) if args.shared_navigation else None
            self.motion_deadline=MotionDeadline()
            self.last_policy_receipt=None
            self.stops=GuardStops()
            self.stop_origin=None
            def on(key,msg):
                with self.lock:self.cache[key]=(time.monotonic(),msg)
            self.subs=[rospy.Subscriber('/joy',Joy,lambda m:on('joy',m),queue_size=1),
                rospy.Subscriber('/fsm_state',Int32,lambda m:on('fsm',m),queue_size=1),
                rospy.Subscriber('/sim2real_state_info',StateInfo,lambda m:on('state',m),queue_size=1),
                rospy.Subscriber('/sim2real_master_node/rbt_state',JointState,lambda m:on('joints',m),queue_size=1),
                rospy.Subscriber('/imu/data',Imu,lambda m:on('imu',m),queue_size=1),
                rospy.Subscriber('/battery_level',UInt8,lambda m:on('battery',m),queue_size=1),
                rospy.Subscriber('/sim2real_debug_data_obs',DebugData,lambda m:on('policy_obs',m),queue_size=1),
                rospy.Subscriber('/competition/lane_request',TwistStamped,self.on_nav,queue_size=1)]
            self.timer=rospy.Timer(rospy.Duration(.02),self.tick)
            rospy.on_shutdown(self.stop)
            signal.signal(signal.SIGHUP,lambda signum,frame:rospy.signal_shutdown('SSH session closed'))

        def __getattr__(self,name):return getattr(self.real,name)
        def publish(self,msg):
            with self.lock:self.cache['manual']=(time.monotonic(),msg)
        def on_nav(self,msg):
            if msg._connection_header.get('callerid')!='/competition_lane_trial':return
            modes={args.token:None,args.token+'/active':0.,
                   args.token+'/active_left':1.,args.token+'/active_right':-1.}
            mode=msg.header.frame_id
            if mode not in modes:return
            stamp=getattr(msg.header,'stamp',None)
            stamp_ns=stamp.to_nsec() if hasattr(stamp,'to_nsec') else None
            with self.lock:self.cache['nav']=(time.monotonic(),msg.twist,stamp_ns,modes[mode])
        def stop(self):
            for _ in range(5):
                try:self.real.publish(Twist())
                except Exception:pass
                time.sleep(.02)
        def tick(self,event):
            now=time.monotonic()
            with self.lock:
                c=dict(self.cache);attitude_metrics={};stop_reason=None
                try:
                    imu=c.get('imu')
                    if imu is not None:
                        q=imu[1].orientation;x,y,z,w=q.x,q.y,q.z,q.w
                        norm=math.sqrt(x*x+y*y+z*z+w*w)
                        if math.isfinite(norm) and abs(norm-1)<=.05 and math.isfinite(imu[1].angular_velocity.z):
                            g=(2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))
                            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                            if self.baseline is None:self.baseline=(g,yaw,[])
                            angle=math.acos(max(-1.,min(1.,sum(a*b for a,b in zip(g,self.baseline[0])))))
                            dyaw=(yaw-self.baseline[1]+math.pi)%(2*math.pi)-math.pi
                            absolute_tilt=math.acos(max(-1.,min(1.,g[2])))
                            attitude_metrics=dict(gravity_change_deg=math.degrees(angle),absolute_tilt_deg=math.degrees(absolute_tilt),yaw_change_deg=math.degrees(dyaw),yaw_rate_rad_s=imu[1].angular_velocity.z)
                    nav=c.get('nav');nv=[0.,0.,0.] if nav is None else [nav[1].linear.x,nav[1].linear.y,nav[1].angular.z]
                    if nav is not None and any(abs(v)>0 for v in nv) and self.begun is None:self.begun=now
                    elapsed=0. if self.begun is None else now-self.begun
                    if args.shared_navigation:
                        # RT10: nothing but the 300 s mission limit stops the walk.
                        # Sensor, calibration, AMP, joint, head-pose and request
                        # age are telemetry only; manual input is ignored.
                        command=[0.,0.,0.];source='stopped'
                        if args.execute and nav is not None and elapsed<args.duration:
                            feedback=None
                            policy_obs=c.get('policy_obs')
                            if policy_obs is not None:
                                receipt,observed=policy_obs
                                if receipt!=self.last_policy_receipt:
                                    feedback=list(observed.data[6:9]);self.last_policy_receipt=receipt
                            raw=self.shared_adapter.update(nv,feedback,True)
                            # A plain session token is startup, shutdown or a
                            # legacy caller. Only active navigation may recover.
                            direction=nav[3] if len(nav)>3 else None
                            command=self.motion_deadline.output(now,raw,direction is not None,direction).tolist()
                            source=('navigation_recovery' if self.motion_deadline.recovering else
                                    'navigation' if any(command) else 'stopped')
                        else:
                            self.motion_deadline.output(now,command,False)
                            stop_reason=('trial_duration' if elapsed>=args.duration
                                         else ('waiting_for_navigation' if nav is None else 'output_disabled'))
                    else:
                        joy=c['joy'][1];joints=c['joints'][1]
                        positions=dict(zip(joints.name,joints.position))
                        manual=c['manual'][1];manual_vector=[manual.linear.x,manual.linear.y,manual.angular.z]
                        active=any(abs(joy.axes[i])>.08 for i in (0,1,3,6,7)) or any(joy.buttons)
                        command,source=choose_command(now,manual_vector,nv,active,now-c['joy'][0],
                            float('inf') if nav is None else now-nav[0],elapsed,True,args.execute,
                            args.duration,args.speed,args.yaw_limit)
                    allowed=True;reason='ok'
                except Exception as exc:
                    command=list(self.last_command);source='hold';allowed=True;reason=str(exc);elapsed=0.
                dt=min(.05,max(.001,now-self.last));self.last=now
                if not args.shared_navigation and source=='navigation' and command[0]>self.v:command[0]=min(command[0],self.v+args.acceleration*dt)
                if not args.shared_navigation and source=='navigation':command[2]=max(self.omega-dt,min(self.omega+dt,command[2]))
                if args.calibrated_turns and source=='navigation' and nv[2]==0:command[2]=0.
                self.last_command=list(command)
                self.v=command[0];self.omega=command[2];self.budget+=math.hypot(command[0],command[1])*dt;self.turn_budget+=abs(command[2])*dt
                out=Twist();out.linear.x,out.linear.y,out.angular.z=command
                self.real.publish(out)
                emitted=time.monotonic()
                if args.shared_navigation:
                    self.motion_deadline.published(emitted,command)
                    if self.begun is None and any(command):self.begun=emitted
                policy_obs=c.get('policy_obs')
                policy_command=None if policy_obs is None else list(policy_obs[1].data[6:9])
                policy_obs_age=None if policy_obs is None else now-policy_obs[0]
                battery=c.get('battery')
                battery_metrics=dict(percent=None if battery is None else battery[1].data,
                                     age_s=None if battery is None else now-battery[0])
                # Keep the existing log keys for cumulative command integrals.
                # Battery, posture and these integrals are telemetry only.
                previous_stop = self.stops.stop_id
                stop_id, stopped_at = self.stops.update(emitted,command)
                nav = c.get('nav')
                stamp_ns = nav[2] if nav is not None and len(nav)>2 else None
                stop_reason = reason if not allowed else stop_reason
                if stop_id!=previous_stop:
                    self.stop_origin = dict(stop_reason=stop_reason, reason=reason, source=source,
                                            navigation_stamp_ns=stamp_ns)
                self.pub.publish(String(data=json.dumps(dict(t=now,emitted_s=emitted,stop_id=stop_id,stopped_at=stopped_at,
                    stop_reason=stop_reason,stop_origin=self.stop_origin,
                    navigation_stamp_ns=stamp_ns,allowed=allowed,reason=reason,source=source,command=command,elapsed=elapsed,distance_command_budget=self.budget,turn_command_budget=self.turn_budget,attitude=attitude_metrics,battery=battery_metrics,policy_command=policy_command,policy_obs_age_s=policy_obs_age,turn_pulse_completed=self.turn_completed))))

    def publisher(name,*a,**kw):
        real=original_publisher(name,*a,**kw)
        if rospy.resolve_name(name)=='/cmd_vel':
            proxy=Guard(real);guard.append(proxy);return proxy
        return real
    rospy.Publisher=publisher
    sys.argv=['/opt/ros/noetic/lib/joy_teleop/joy_teleop.py','__name:=joy_teleop']
    try:runpy.run_path(sys.argv[0],run_name='__main__')
    finally:
        for proxy in guard:proxy.stop()


def vision_role(args):
    if args.shared_navigation:
        from deploy.shared_navigation import run
        return run(args,publish=True)
    import cv2,numpy as np,rospy
    from geometry_msgs.msg import TwistStamped
    from std_msgs.msg import String
    from deploy.shadow import read_exact,capture_worker
    from deploy.line_follow_core import observe_lane,TurnThenWalk
    follower=TurnThenWalk()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    rospy.init_node('competition_lane_trial',disable_signals=True)
    pulse={}
    def on_pulse(msg):
        if msg._connection_header.get('callerid')=='/joy_teleop':pulse.update(json.loads(msg.data));pulse['receipt']=time.monotonic()
    rospy.Subscriber('/competition/line_guard',String,on_pulse,queue_size=1)
    pub=rospy.Publisher('/competition/lane_request',TwistStamped,queue_size=1)
    proc=None;rows=[];status='failed';video=None
    try:
        err=(out/'camera.log').open('w')
        proc=subprocess.Popen([str(ROOT/'deploy/realsense_stream')],stdout=subprocess.PIPE,stderr=err,bufsize=0)
        header=bytearray()
        while not header.endswith(b'\n'):header.extend(read_exact(proc.stdout,1,12))
        camera=json.loads(header);(out/'camera.json').write_text(json.dumps(camera,indent=2))
        latest=queue.Queue(maxsize=1);threading.Thread(target=capture_worker,args=(proc.stdout,camera['width'],camera['height'],latest),daemon=True).start()
        small=dict(camera,fx=camera['fx']/2,fy=camera['fy']/2,cx=(camera['cx']+.5)/2-.5,cy=(camera['cy']+.5)/2-.5)
        start=time.monotonic();begun=None;good=0;bad_since=None;last_frame=start
        video=cv2.VideoWriter(str(out/'camera_trial.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
        while time.monotonic()-start<args.duration+16.:
            if begun is None and time.monotonic()-start>16:raise RuntimeError('lane_warmup_deadline')
            try:frame=latest.get(timeout=.5)
            except queue.Empty:
                m=TwistStamped();m.header.frame_id=args.token;pub.publish(m)
                good=0;follower=TurnThenWalk()
                if begun is not None and time.monotonic()-last_frame>2:raise RuntimeError('camera_frames_stopped')
                continue
            if isinstance(frame,Exception):raise frame
            last_frame=time.monotonic()
            if time.monotonic()-frame.time>.18:
                m=TwistStamped();m.header.frame_id=args.token;pub.publish(m)
                good=0;follower=TurnThenWalk()
                if bad_since is None:bad_since=time.monotonic()
                if begun is not None and time.monotonic()-bad_since>8:raise RuntimeError('fresh_camera_recovery_timeout')
                continue
            rgb=cv2.resize(frame.rgb,(320,240),interpolation=cv2.INTER_AREA)
            depth=cv2.resize(frame.depth,(320,240),interpolation=cv2.INTER_NEAREST)
            tick=time.monotonic()
            inhibit_reason=None;terminal_error=None
            try:
                model=observe_lane(rgb,depth,small,speed=args.speed);good+=1
            except ValueError as exc:
                inhibit_reason=str(exc)
                good=0
                model=None
                if begun is not None and 'raised object in forward corridor' in str(exc):terminal_error='lane_lost: '+str(exc)
            now=time.monotonic()
            if model is not None and now-frame.time>.18:
                inhibit_reason='perception_latency_exceeded';model=None;good=0
                follower=TurnThenWalk()
            if model is None:
                follower=TurnThenWalk()
                if bad_since is None:bad_since=now
                if begun is not None and now-bad_since>8:terminal_error='lane_recovery_timeout: '+str(inhibit_reason)
            elif good>=3:bad_since=None
            if model is not None and args.calibrated_turns and good>=3:
                completed=pulse.get('turn_pulse_completed',False) and now-pulse.get('receipt',0)<.2
                model['command'],model['control_phase'],model['target_bearing_rad']=follower.command(model,args.speed,now,turn_completed=completed)
            if good>=3 and begun is None:begun=now
            request=TwistStamped();request.header.stamp=rospy.Time.now();request.header.frame_id=args.token
            if begun is not None and now-begun<args.duration and model is not None and good>=3:
                request.twist.linear.x,request.twist.linear.y,request.twist.angular.z=model['command']
            pub.publish(request)
            row=dict(time=now,frame_age_s=now-frame.time,processing_ms=1000*(now-tick),request=[request.twist.linear.x,request.twist.linear.y,request.twist.angular.z],model=model,inhibit_reason=inhibit_reason)
            rows.append(row)
            with (out/'vision.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            text='lane missing' if model is None else 'offset %.3f m | yaw %.3f'%(model['camera_center_offset_m'],model['command'][2])
            cv2.putText(image,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            if video.isOpened():video.write(image)
            np.save(out/'latest_depth_m.npy',depth)
            cv2.imwrite(str(out/'latest_rgb.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/'latest.jpg'),image)
            # Send the zero request before disk I/O, then retain the actual
            # rejected frame for diagnosis after an unreliable SSH link returns.
            if terminal_error:
                (out/'failed_observation.json').write_text(json.dumps(dict(reason=terminal_error,observation=row),indent=2))
                raise RuntimeError(terminal_error)
            if begun is not None and now-begun>=args.duration:status='vision_trial_completed';break
        if begun is None:raise RuntimeError('no_stable_pair_of_boundaries')
    except Exception as exc:status='failed: '+type(exc).__name__+': '+str(exc)
    finally:
        for _ in range(5):
            m=TwistStamped();m.header.frame_id=args.token;pub.publish(m);time.sleep(.02)
        if proc is not None:
            proc.terminate()
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if video is not None:video.release()
        (out/'vision_summary.json').write_text(json.dumps(dict(status=status,frames=len(rows),valid_frames=sum(r['model'] is not None for r in rows)),indent=2))
        print(json.dumps(dict(status=status,frames=len(rows))),flush=True);rospy.signal_shutdown('vision finished')
    return 0 if status=='vision_trial_completed' else 2


def turn_test_role(args):
    import rospy
    from geometry_msgs.msg import TwistStamped
    from std_msgs.msg import String
    rospy.init_node('competition_lane_trial',disable_signals=True)
    pub=rospy.Publisher('/competition/lane_request',TwistStamped,queue_size=1)
    latest={};rospy.Subscriber('/competition/line_guard',String,lambda m:latest.update(json.loads(m.data)),queue_size=1)
    start=time.monotonic();rows=[]
    # The smaller pulse reproduces the old input. The larger pulses test the
    # factory filter/deadzone; each ends at ten measured degrees or its timeout.
    phases=[(1.,2.),(0.,.8),(-1.,2.),(0.,.8)]
    try:
        for yaw,seconds in phases:
            begun=time.monotonic();initial=latest.get('attitude',{}).get('yaw_change_deg',0.)
            while time.monotonic()-begun<seconds:
                if time.monotonic()-start>args.duration:break
                current=latest.get('attitude',{}).get('yaw_change_deg',0.)
                if yaw and abs(current-initial)>=10:break
                m=TwistStamped();m.header.frame_id=args.token;m.header.stamp=rospy.Time.now();m.twist.angular.z=yaw;pub.publish(m)
                rows.append(dict(t=time.monotonic(),requested_yaw=yaw,observed_yaw_deg=current));time.sleep(.05)
    finally:
        for _ in range(5):
            m=TwistStamped();m.header.frame_id=args.token;pub.publish(m);time.sleep(.02)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'turn_test.json').write_text(json.dumps(rows,indent=2))
        (args.output/'vision_summary.json').write_text(json.dumps(dict(status='turn_test_completed',frames=len(rows))))
        rospy.signal_shutdown('turn test ended')
    return 0


def supervisor_role(args):
    if args.shared_navigation:
        from navigation.calibration import calibration_reasons
        from deploy.shared_navigation import preflight
        config=json.loads(args.config.read_text())
        reasons=calibration_reasons(config)
        if reasons:raise ValueError('Motion inhibited before changing control ownership: '+', '.join(reasons))
        preflight(config)
    import fcntl
    (ROOT/'runs').mkdir(parents=True,exist_ok=True)
    # All releases must contend on the same lock, including retry launches.
    control_lock=Path('/tmp/competition_navigation_control.lock').open('a+')
    try:fcntl.flock(control_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        args.output.mkdir(parents=True,exist_ok=True)
        report=dict(status='control_busy_no_motion_started',execute=args.execute)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        return 3
    import rospy,rosgraph,rosnode
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    rospy.init_node('competition_trial_supervisor',disable_signals=True)
    out=args.output;out.mkdir(parents=True,exist_ok=True);last={};commands=[];guard_process=None;vision=None;zero=None;changed=False;status='failed';original=None
    final_zero=None;restored=False
    def on_guard(msg):
        last.update(json.loads(msg.data));last['receipt']=time.monotonic()
        with (out/'guard.jsonl').open('a') as f:f.write(msg.data+'\n')
    def on_cmd(msg):commands.append([time.monotonic(),msg.linear.x,msg.linear.y,msg.angular.z])
    subs=[rospy.Subscriber('/competition/line_guard',String,on_guard,queue_size=20),rospy.Subscriber('/cmd_vel',Twist,on_cmd,queue_size=10)]
    master=rosgraph.Master(rospy.get_name())
    def publishers(topic):return dict(master.getSystemState()[0]).get(topic,[])
    def wait_for(predicate,seconds,message):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.05)
        raise RuntimeError(message)
    def stop_process(p):
        if p is None or p.poll() is not None:return
        p.terminate()
        try:p.wait(timeout=2)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    def original_teleop():
        log=(out/'restored_teleop.log').open('ab')
        return subprocess.Popen(['/usr/bin/python3','/opt/ros/noetic/lib/joy_teleop/joy_teleop.py','__name:=joy_teleop'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    def interrupted(signum,frame):raise RuntimeError('trial interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    panel=None
    try:
        if args.shared_navigation and args.execute:
            try:
                from deploy.box_panel import start_panel
                panel=start_panel(out)
            except (OSError,ValueError) as exc:
                print('RT11 调参面板启动失败：'+str(exc),flush=True)
        if publishers('/cmd_vel')!=['/joy_teleop'] or publishers('/joy_msg')!=['/joy_teleop']:raise RuntimeError('unexpected control ownership before trial')
        wait_for(lambda:len(commands)>=5,3,'no current manual velocity')
        if any(max(abs(v) for v in r[1:])>.005 for r in commands[-5:]):raise RuntimeError('manual velocity is not neutral')
        params=rospy.get_param('/teleop');(out/'original_teleop.json').write_text(json.dumps(params,indent=2))
        if params['walk']['topic_name']!='cmd_vel':raise RuntimeError('unexpected teleop mapping')
        zero=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        for _ in range(5):zero.publish(Twist());time.sleep(.02)
        changed=True;rosnode.kill_nodes(['/joy_teleop'])
        wait_for(lambda:not publishers('/joy_msg'),3,'original joystick translator did not exit')
        cmd=[sys.executable,str(Path(__file__)),'proxy','--output',str(out),'--token',args.token,'--duration',str(args.duration),'--speed',str(args.speed)]
        if args.execute:cmd+=['--execute']
        if args.turn_test:cmd+=['--turn-test']
        if args.calibrated_turns:cmd+=['--calibrated-turns']
        if args.shared_navigation:cmd+=['--shared-navigation','--config',str(args.config)]
        log=(out/'proxy.log').open('wb');guard_process=subprocess.Popen(cmd,stdout=log,stderr=log)
        wait_for(lambda:bool(last) and '/joy_teleop' in publishers('/cmd_vel'),8,'guarded joystick translator did not start')
        zero.unregister();zero=None
        wait_for(lambda:last.get('allowed') and last.get('source')=='stopped' and time.monotonic()-last.get('receipt',0)<.2,5,'guard readiness failed: '+str(last))
        if publishers('/cmd_vel')!=['/joy_teleop'] or publishers('/joy_msg')!=['/joy_teleop']:raise RuntimeError('control ownership is not unique')
        log=(out/'vision.log').open('wb')
        vision_cmd=[sys.executable,str(Path(__file__)),'vision','--output',str(out/'vision'),'--token',args.token,'--duration',str(args.duration),'--speed',str(args.speed)]
        if args.turn_test:vision_cmd+=['--turn-test']
        if args.calibrated_turns:vision_cmd+=['--calibrated-turns']
        if args.shared_navigation:vision_cmd+=['--shared-navigation','--config',str(args.config)]
        vision_env = None
        if args.shared_navigation:
            # The lane loop and the mapping worker already run concurrently. Bound
            # native worker pools before Python loads NumPy and OpenCV on the Jetson.
            vision_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1')
        vision=subprocess.Popen(vision_cmd,stdout=log,stderr=log,env=vision_env)
        deadline=time.monotonic()+args.duration+28
        while vision.poll() is None:
            if guard_process.poll() is not None or time.monotonic()-last.get('receipt',0)>.2:raise RuntimeError('velocity watchdog unavailable')
            if not last.get('allowed') or last.get('source')=='manual':raise RuntimeError('trial stopped: '+str(last.get('reason'))+' / '+str(last.get('source')))
            if time.monotonic()>deadline:raise RuntimeError('trial deadline')
            if publishers('/cmd_vel')!=['/joy_teleop']:raise RuntimeError('velocity ownership changed during trial')
            time.sleep(.03)
        if vision.returncode:raise RuntimeError((out/'vision/vision_summary.json').read_text())
        status=('full_lap_completed' if args.shared_navigation else 'limited_line_trial_completed') if args.execute else 'zero_output_validation_completed'
    except Exception as exc:status='failed: '+str(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGHUP,signal.SIG_IGN)
        stop_process(vision)
        if changed:
            # Removing the navigation worker makes the independent guard stop.
            time.sleep(.25);stop_process(guard_process)
            try:wait_for(lambda:not publishers('/joy_msg'),3,'guarded joystick translator did not exit')
            except Exception as exc:status+='; '+str(exc)
            if zero is None:zero=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
            for _ in range(10):
                zero.publish(Twist())
                if final_zero is None:final_zero=time.monotonic()
                time.sleep(.02)
            original=original_teleop()
            try:
                wait_for(lambda:'/joy_teleop' in publishers('/cmd_vel') and '/joy_teleop' in publishers('/joy_msg'),6,'manual translator restoration failed')
                restored=original.poll() is None
            except Exception as exc:status+='; '+str(exc)
        if zero is not None:zero.unregister()
        report=dict(status=status,execute=args.execute,max_requested_speed_m_s=args.speed,maximum_trial_s=args.duration,
                    acceleration_limit_m_s2=args.acceleration,command_distance_budget_m=None,command_turn_budget_rad=None,
                    nonzero_output_samples=sum(max(abs(x) for x in r[1:])>1e-5 for r in commands),
                    output_peak_vx=max((r[1] for r in commands),default=0.),output_peak_abs_yaw=max((abs(r[3]) for r in commands),default=0.),
                    manual_translator_restored=(not changed) or restored,manual_translator_untouched=not changed,controller_mode_changed=False,body_joint_commands_sent=False)
        for sub in subs:sub.unregister()
        if panel is not None:
            panel.close()
        control_lock.close()
        if args.shared_navigation:
            from deploy.stop_archive import reconcile
            try:reconcile(out, final_zero, status.splitlines()[0][:300])
            except Exception as exc:report['stop_archive_error']=str(exc)
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        rospy.signal_shutdown('trial supervisor finished')
    return 0 if status.endswith('_completed') else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('role',choices=['supervise','proxy','vision']);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--token',default='lane-'+str(time.time_ns()));p.add_argument('--duration',type=float);p.add_argument('--speed',type=float)
    p.add_argument('--turn-test',action='store_true')
    p.add_argument('--calibrated-turns',action='store_true')
    p.add_argument('--shared-navigation',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'deploy/pro_config.json')
    args=p.parse_args()
    if args.duration is None:args.duration=300. if args.shared_navigation else 6.
    if args.speed is None:args.speed=.22 if args.shared_navigation else .04
    if args.shared_navigation and (args.calibrated_turns or args.turn_test):p.error('Shared navigation cannot be combined with legacy turn trials')
    maximum_duration=300 if args.shared_navigation else (180 if args.calibrated_turns else 30)
    if not 0<args.duration<=maximum_duration:p.error('duration exceeds this trial mode limit')
    if args.turn_test and args.duration>8:p.error('turn test duration must be at most 8 seconds')
    if not math.isfinite(args.speed) or not 0<args.speed<=.4:p.error('speed must be within (0,0.4] m/s')
    args.acceleration=.04 if args.speed<=.04 else .4
    args.yaw_limit=1. if args.turn_test or args.calibrated_turns or args.shared_navigation else (.04 if args.speed<=.04 else .5)
    if args.shared_navigation and args.role=='supervise' and not args.execute:
        from deploy.shared_navigation import run
        return run(args,publish=False)
    if args.role=='proxy':return proxy_role(args)
    if args.role=='vision':return turn_test_role(args) if args.turn_test else vision_role(args)
    return supervisor_role(args)


if __name__=='__main__':raise SystemExit(main())
