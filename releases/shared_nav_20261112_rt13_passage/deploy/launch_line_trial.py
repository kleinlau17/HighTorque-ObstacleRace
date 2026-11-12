"""Launch one bounded on-robot trial, independent of SSH connection lifetime.

The per-output launch marker prevents an uncertain SSH reply from starting the
same trial twice. The supervisor separately holds the exclusive control lock.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--speed',type=float)
p.add_argument('--duration',type=float)
p.add_argument('--execute',action='store_true')
p.add_argument('--turn-test',action='store_true')
p.add_argument('--calibrated-turns',action='store_true')
p.add_argument('--shared-navigation',action='store_true')
p.add_argument('--config',type=Path,default=ROOT/'deploy/pro_config.json')
args=p.parse_args()
if args.duration is None:args.duration=300. if args.shared_navigation else 30.
if args.speed is None:args.speed=.22 if args.shared_navigation else .4
if args.shared_navigation and (args.turn_test or args.calibrated_turns):p.error('Shared navigation cannot use legacy turn modes')
maximum_duration=300 if args.shared_navigation else (180 if args.calibrated_turns else 30)
if not 0<args.speed<=.4 or not 0<args.duration<=maximum_duration:p.error('speed/duration outside trial limits')
out=(ROOT/args.output).resolve()
if (ROOT/'runs').resolve() not in out.parents:p.error('output must be below this release runs directory')
out.mkdir(parents=True,exist_ok=True);marker=out/'launch.json'
if marker.exists():
    print(json.dumps(dict(status='already_recorded_no_duplicate_launch',launch=json.loads(marker.read_text()))),flush=True)
    raise SystemExit(0)
# A completed zero-output check may still be exiting its ROS node. Wait for
# the previous owner to release control before recording a new launch.
with Path('/tmp/competition_navigation_control.lock').open('a+') as control:
    deadline=time.monotonic()+8
    while True:
        try:fcntl.flock(control.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise SystemExit('Control is still owned; no new task launched')
            time.sleep(.05)
    fcntl.flock(control.fileno(),fcntl.LOCK_UN)
try:fd=os.open(str(marker),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
    print(json.dumps(dict(status='already_recorded_no_duplicate_launch',launch=json.loads(marker.read_text()))),flush=True)
    raise SystemExit(0)
with os.fdopen(fd,'w') as f:json.dump(dict(status='launching',time=time.time()),f)
command=[sys.executable,str(ROOT/'deploy/line_trial.py'),'supervise','--speed',str(args.speed),'--duration',str(args.duration),'--output',str(out)]
if args.execute:command.append('--execute')
if args.turn_test:command.append('--turn-test')
if args.calibrated_turns:command.append('--calibrated-turns')
if args.shared_navigation:command+=['--shared-navigation','--config',str(args.config)]
with (out/'mission.log').open('ab') as log:
    proc=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
report=dict(status='launched',pid=proc.pid,time=time.time(),speed_m_s=args.speed,maximum_duration_s=args.duration,execute=args.execute,output=str(out))
temporary=marker.with_suffix('.new');temporary.write_text(json.dumps(report,indent=2));os.replace(str(temporary),str(marker))
print(json.dumps(report,indent=2),flush=True)
