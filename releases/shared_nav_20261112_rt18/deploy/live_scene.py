"""Continuously capture + recognise and publish one frame's scene for the live 3D view."""
import argparse, json, struct, subprocess, time
from pathlib import Path
import cv2, numpy as np
from navigation.visual import cluster_points, fit_box

W,H=640,480

def read_exact(stream,size,timeout=10.):
    data=bytearray(); start=time.monotonic()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk:
            if time.monotonic()-start>timeout: raise TimeoutError('camera read')
            time.sleep(.005); continue
        data.extend(chunk)
    return bytes(data)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='deploy/pro_model_config.json')
    ap.add_argument('--output',default='runs/live')
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--interval',type=float,default=.35)
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([str(Path(__file__).resolve().parent/'realsense_stream')],
                          stdout=subprocess.PIPE,stderr=open(out/'camera.log','w'),bufsize=0)
    header=bytearray()
    while not header.endswith(b'\n'): header.extend(read_exact(proc.stdout,1))
    cam=json.loads(header)
    k=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    scale=float(cfg.get('perception_scale',.5)); sw,sh=int(round(W*scale)),int(round(H*scale))
    intr=[k[0,0]*scale,k[1,1]*scale,(k[0,2]+.5)*scale-.5,(k[1,2]+.5)*scale-.5]
    dst=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
    maps=cv2.initUndistortRectifyMap(k,np.asarray(cam['coefficients']),None,dst,(sw,sh),cv2.CV_32FC1)
    from navigation.visual import CameraPerception
    perception=CameraPerception(sw,sh,80.,cfg['camera_position_body_m'],
                                cfg['camera_rotation_optical_to_body'],intr)
    # Real attitude from the AHRS; without it the level frame is wrong.
    import rospy
    from sensor_msgs.msg import Imu
    rospy.init_node('live_scene', anonymous=True)
    latest={'q':None}
    def on_imu(msg):
        q=msg.orientation; latest['q']=(q.x,q.y,q.z,q.w)
    rospy.Subscriber('/imu/data',Imu,on_imu,queue_size=1)
    body_from_imu=np.asarray(cfg.get('body_from_imu',np.eye(3)),dtype=float)
    def quat_matrix(x,y,z,w):
        n=x*x+y*y+z*z+w*w
        if n<1e-12: return np.eye(3)
        s=2./n
        return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                         [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                         [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
    def attitude_now():
        q=latest['q']
        if q is None: return np.eye(3)
        return quat_matrix(*q)@body_from_imu.T
    def grab():
        packet=read_exact(proc.stdout,8+W*H*7)
        ts=struct.unpack('<d',packet[:8])[0]
        rgb=np.frombuffer(packet,np.uint8,W*H*3,8).reshape(H,W,3).copy()
        depth=np.frombuffer(packet,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
        rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR); depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        depth[(depth<.15)|(depth>4.)]=np.nan
        return type('F',(),{'time':ts,'rgb':rgb,'depth':depth})(), rgb
    for _ in range(a.warmup): grab()
    while True:
        t0=time.monotonic()
        frame,rgb=grab()
        attitude=attitude_now()
        t=time.monotonic(); obs,scene=perception.inspect_lane(frame,attitude,None,inside_start=True)
        lane_ms=(time.monotonic()-t)*1e3
        if perception.analysis is not None:
            level,finite,height,xy,points,floor,z,yaw,tt,rgb0=perception.analysis
            bright,dark=rgb.max(axis=2),rgb.min(axis=2)
            white=(dark>115)&(bright-dark<.25*bright)&(xy[...,0]>.30)&floor
            route=obs.lane if obs.lane is not None else None
            cx=np.polyval(route,xy[...,0]) if route is not None else np.zeros_like(xy[...,0])
            in_path=(xy[...,1]>np.minimum(cx,0.)-.30)&(xy[...,1]<np.maximum(cx,0.)+.30); in_path|=abs(xy[...,1])<.33
            raised=finite&(height>.06)&(height<.65)&(xy[...,0]>.22)&(xy[...,0]<1.15)&in_path
            t=time.monotonic(); avoid,ascene=perception.inspect_avoid(perception.analysis); avoid_ms=(time.monotonic()-t)*1e3
            body=level@attitude
            w=body[white][:,:3]; wc=rgb[white]
            r=body[raised]; rc=rgb[raised]
            if len(r)>3000:
                step=max(1,len(r)//3000); r=r[::step]; rc=rc[::step]
            t=time.monotonic()
            boxes=[b for b in (fit_box(c) for c in cluster_points(body[raised])) if b is not None]
            box_ms=(time.monotonic()-t)*1e3
            for b in boxes: b['floor']=float(-z)
            payload=dict(t=time.time(),
                white=[[round(float(w[i,0]),3),round(float(w[i,1]),3),round(float(w[i,2]),3),int(wc[i,0]),int(wc[i,1]),int(wc[i,2])] for i in range(len(w))],
                raised=[[round(float(r[i,0]),3),round(float(r[i,1]),3),round(float(r[i,2]),3),int(rc[i,0]),int(rc[i,1]),int(rc[i,2])] for i in range(len(r))],
                boxes=boxes,floor=float(-z),
                timing=dict(lane_ms=lane_ms,avoid_ms=avoid_ms,box_ms=box_ms),
                scene=scene,basis=perception.lane_basis)
        else:
            payload=dict(t=time.time(),white=[],raised=[],boxes=[],floor=0.,
                         timing=dict(lane_ms=lane_ms,avoid_ms=None,box_ms=None),
                         scene=scene,basis=perception.lane_basis)
        cv2.imwrite(str(out/'live.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        tmp=out/'.scene.json.tmp'; tmp.write_text(json.dumps(payload)); tmp.replace(out/'scene.json')
        dt=time.monotonic()-t0
        time.sleep(max(0.,a.interval-dt))

if __name__=='__main__': main()
