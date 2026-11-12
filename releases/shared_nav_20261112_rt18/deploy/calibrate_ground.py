"""Capture real RGB-D and passive robot feedback, then estimate a floor plane."""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
import rospy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.shadow import Sensors,read_exact,capture_worker
from deploy.ground_calibration import fit_ground


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=8.)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rospy.init_node('competition_floor_calibration',anonymous=True,disable_signals=True)
    sensors=Sensors();proc=None;rows=[];clouds=[];planes=[];error=None;stale_frames=0
    try:
        with (args.output/'camera.log').open('w') as err:
            proc=subprocess.Popen([str(ROOT/'deploy/realsense_stream')],stdout=subprocess.PIPE,stderr=err,bufsize=0)
            header=bytearray()
            while not header.endswith(b'\n'):
                header.extend(read_exact(proc.stdout,1,12))
                if len(header)>4096:raise ValueError('Invalid camera header')
            camera=json.loads(header)
            (args.output/'camera_intrinsics.json').write_text(json.dumps(camera,indent=2))
            if np.any(np.abs(camera['coefficients'])>1e-9):raise ValueError('Nonzero distortion requires explicit rectification')
            w,h=camera['width'],camera['height'];u,v=np.meshgrid(np.arange(w),np.arange(h))
            rays=np.stack([(u-camera['cx'])/camera['fx'],(v-camera['cy'])/camera['fy'],np.ones_like(u)],-1)
            latest=queue.Queue(maxsize=1)
            threading.Thread(target=capture_worker,args=(proc.stdout,w,h,latest),daemon=True).start()
            start=time.monotonic()
            while time.monotonic()-start<args.duration:
                frame=latest.get(timeout=18.)
                if isinstance(frame,Exception):raise frame
                frame_age=time.monotonic()-frame.time
                if frame_age>.3:
                    # A delayed startup packet must never enter calibration.
                    # Discard it and keep waiting inside the bounded run.
                    stale_frames+=1
                    continue
                imu,joints,head=sensors.snapshot(frame.time)
                rgb=frame.rgb.astype(float);depth=frame.depth
                valid=np.isfinite(depth)&(depth>.15)&(depth<3.5)
                mask=valid&(rgb[:,:,1]>1.25*rgb[:,:,0])&(rgb[:,:,1]>1.10*rgb[:,:,2])&(v>h*.25)
                mask[1::2]=False;mask[:,1::2]=False
                p=(rays*depth[:,:,None])[mask]
                if len(p)>1800:p=p[np.linspace(0,len(p)-1,1800).astype(int)]
                if len(p)>=300:
                    clouds.append(p)
                    try:planes.append(fit_ground(p))
                    except ValueError:pass
                rows.append(dict(time=frame.time,frame_age_s=frame_age,imu_xyzw=imu[1].tolist(),imu_source_stamp=imu[2],
                    joints={n:val[1] for n,val in joints.items() if frame.time-val[0]<.3},
                    head={n:{'pos':val[1],'vel':val[2]} for n,val in head.items() if frame.time-val[0]<.3},
                    green_depth_points=len(p),valid_depth_fraction=float(valid.mean())))
                np.save(args.output/'latest_depth_m.npy',depth)
                cv2.imwrite(str(args.output/'latest_rgb.jpg'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                overlay=frame.rgb.copy();overlay[mask]=[255,100,255]
                cv2.imwrite(str(args.output/'floor_samples.jpg'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
            result=fit_ground(np.vstack(clouds)) if clouds else None
            if result is None:raise ValueError('No sufficiently visible green floor')
            if len(planes)<8:raise ValueError('Too few independently fitted floor frames')
            normals=np.array([p['normal_camera'] for p in planes]);mean=normals.mean(axis=0);mean/=np.linalg.norm(mean)
            angular=np.degrees(np.arccos(np.clip(normals@mean,-1,1)))
            heights=np.array([p['camera_height_m'] for p in planes])
            result.update(height_std_m=float(heights.std()),normal_angle_p95_deg=float(np.percentile(angular,95)),fitted_frames=len(planes))
            if heights.std()>.015 or np.percentile(angular,95)>2.:raise ValueError('Floor fit unstable across frames; head/camera may have moved')
            (args.output/'ground_plane.json').write_text(json.dumps(result,indent=2))
    except Exception as exc:error=str(exc)
    finally:
        if proc is not None:
            proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        rospy.signal_shutdown('calibration stopped')
        (args.output/'sensor_samples.json').write_text(json.dumps(rows))
        summary=dict(status='ground_plane_estimated' if error is None else 'failed',error=error,frames=len(rows),
            stale_frames_discarded=stale_frames,
            joint_complete_frames=sum(len(r['joints'])==20 for r in rows),head_complete_frames=sum(len(r['head'])==2 for r in rows),
            body_extrinsics_verified=False,control_config_modified=False,motor_publishers=0)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None else 2


if __name__=='__main__':raise SystemExit(main())
