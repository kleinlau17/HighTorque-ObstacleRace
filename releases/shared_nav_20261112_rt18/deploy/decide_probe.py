"""Capture one frame, run the current perception, and report the avoid decision."""
import json, struct, subprocess, time
from pathlib import Path
import cv2, numpy as np
W,H=640,480
def read_exact(stream,size,timeout=15.):
    data=bytearray(); start=time.monotonic()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk:
            if time.monotonic()-start>timeout: raise TimeoutError('camera')
            time.sleep(.005); continue
        data.extend(chunk)
    return bytes(data)
def main():
    cfg=json.loads(Path('deploy/pro_model_config.json').read_text())
    out=Path('runs/decide'); out.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['deploy/realsense_stream'],stdout=subprocess.PIPE,
                          stderr=open('runs/decide/camera.log','w'),bufsize=0)
    header=bytearray()
    while not header.endswith(b'\n'): header.extend(read_exact(proc.stdout,1))
    cam=json.loads(header)
    k=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    scale=float(cfg.get('perception_scale',.5)); sw,sh=int(round(W*scale)),int(round(H*scale))
    intr=[k[0,0]*scale,k[1,1]*scale,(k[0,2]+.5)*scale-.5,(k[1,2]+.5)*scale-.5]
    dst=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
    maps=cv2.initUndistortRectifyMap(k,np.asarray(cam['coefficients']),None,dst,(sw,sh),cv2.CV_32FC1)
    from navigation import params
    params.load('deploy/tuning.json')
    from navigation.visual import CameraPerception, cluster_points, fit_box
    perception=CameraPerception(sw,sh,80.,cfg['camera_position_body_m'],
                                cfg['camera_rotation_optical_to_body'],intr)
    import rospy
    from sensor_msgs.msg import Imu
    rospy.init_node('decide_probe', anonymous=True)
    latest={'q':None}
    rospy.Subscriber('/imu/data',Imu,lambda m: latest.update(q=(m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w)),queue_size=1)
    body_from_imu=np.asarray(cfg.get('body_from_imu',np.eye(3)),dtype=float)
    def quat_matrix(x,y,z,w):
        n=x*x+y*y+z*z+w*w
        if n<1e-12: return np.eye(3)
        s=2./n
        return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                         [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                         [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
    attitude=quat_matrix(*latest['q'])@body_from_imu.T if latest['q'] else np.eye(3)
    def grab():
        p=read_exact(proc.stdout,8+W*H*7)
        ts=struct.unpack('<d',p[:8])[0]
        rgb=np.frombuffer(p,np.uint8,W*H*3,8).reshape(H,W,3).copy()
        depth=np.frombuffer(p,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
        rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR); depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        depth[(depth<.15)|(depth>4.)]=np.nan
        return type('F',(),{'time':ts,'rgb':rgb,'depth':depth})(),rgb,depth
    for _ in range(20): grab()
    frame,rgb,depth=grab()
    proc.kill()
    obs,scene=perception.inspect_lane(frame,attitude,None,inside_start=True)
    avoid,ascene=perception.inspect_avoid(perception.analysis) if perception.analysis is not None else (obs,'clear')
    lines=[]
    if avoid is not None and avoid.edges is not None and len(avoid.edges)>=2: lines=[avoid.edges[0],avoid.edges[1]]
    elif avoid is not None and avoid.raw_lane is not None: lines=[avoid.raw_lane]
    sites=perception.raised_sites
    boxes=[]
    if sites is not None and len(sites):
        boxes=[b for b in (fit_box(c) for c in cluster_points(sites[:,:2])) if b is not None]
    # single-frame coloured scene for the 3D view
    scene_payload=None
    if perception.analysis is not None:
        _level,_finite,_height,_xy,_points,_floor,_z,_yaw,_t,_rgb=perception.analysis
        _bright,_dark=rgb.max(axis=2),rgb.min(axis=2)
        _white=((_dark>115)&(_bright-_dark<.25*_bright)&(_xy[...,0]>.30)&_floor)
        _route=obs.lane if obs is not None and obs.lane is not None else None
        _cx=np.polyval(_route,_xy[...,0]) if _route is not None else np.zeros_like(_xy[...,0])
        _in=(_xy[...,1]>np.minimum(_cx,0.)-.30)&(_xy[...,1]<np.maximum(_cx,0.)+.30); _in|=abs(_xy[...,1])<.33
        _raised=_finite&(_height>.06)&(_height<.65)&(_xy[...,0]>.22)&(_xy[...,0]<1.15)&_in
        _body=_level@attitude
        _w=_body[_white][:,:3]; _wc=rgb[_white]
        _r=_body[_raised][:,:3]; _rc=rgb[_raised]
        if len(_r)>4000:
            _step=max(1,len(_r)//4000); _r=_r[::_step]; _rc=_rc[::_step]
        scene_payload=dict(white=[[round(float(v),3) for v in _w[i]]+[int(_wc[i,0]),int(_wc[i,1]),int(_wc[i,2])] for i in range(len(_w))],
                           raised=[[round(float(v),3) for v in _r[i]]+[int(_rc[i,0]),int(_rc[i,1]),int(_rc[i,2])] for i in range(len(_r))],
                           boxes=[dict(center=[round(b['center'][0],3),round(b['center'][1],3)],yaw=round(b['yaw'],3),size=b['size'],excess=round(b['excess'],3),points=b['points']) for b in boxes],
                           floor=round(float(-_z),3))
        (out/'scene.json').write_text(json.dumps(scene_payload))
    rep=dict(scene=scene,avoid_scene=ascene,avoid_basis=perception.lane_basis,
             sites=0 if sites is None else int(len(sites)),
             boxes=[dict(center=[round(b['center'][0],3),round(b['center'][1],3)],yaw=round(b['yaw'],3),
                         excess=round(b['excess'],3),points=b['points']) for b in boxes],
             lines=len(lines))
    if boxes:
        by=float(boxes[0]['center'][1]); bx=float(boxes[0]['center'][0])
        info=[]
        for ln in lines:
            ly=float(np.polyval(ln,bx)); d=abs(ly-by)
            info.append(dict(line_y=round(ly,3),dist=round(d,3),connected=bool(d<0.06)))
        free=[i for i in info if not i['connected']]
        lateral=abs(by); body_half=params.get('avoid','body_half')
        if free and lateral>=body_half:
            decision='follow'
            y=float(np.polyval(lines[0],.25)); heading=float(np.arctan(np.polyval(np.polyder(lines[0]),.3)))
            target=[params.get('avoid','follow_vx'),float(np.clip(params.get('avoid','follow_ky')*y,-params.get('avoid','follow_vy_max'),params.get('avoid','follow_vy_max'))),float(np.clip(params.get('avoid','follow_kw')*heading,-params.get('avoid','follow_w_max'),params.get('avoid','follow_w_max')))]
        elif info or lateral<body_half:
            decision='strafe'
            nearest=min(info,key=lambda i:i['dist']) if info else None
            d=-1. if (nearest and nearest['line_y']>by) else 1.
            target=[0.,d*params.get('avoid','strafe_speed'),0.]
        else:
            decision='back'; target=[params.get('avoid','back_speed'),0.,0.]
        rep.update(box=(round(bx,3),round(by,3)),lateral=round(lateral,3),body_half=body_half,
                   info=info,free=len(free),decision=decision,target=[round(v,3) for v in target])
    else:
        rep['decision']='no_box (not an obstacle)'
    rep['depth_valid']=round(float(np.mean(np.isfinite(depth)&(depth>.15)&(depth<2.7))),3)
    cv2.imwrite('runs/decide/rgb.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    vis=np.where(np.isfinite(depth),depth,np.nan)
    lo,hi=np.nanpercentile(vis,[5,95]) if np.isfinite(vis).any() else (0,1)
    cv2.imwrite('runs/decide/depth.png',(np.clip((np.nan_to_num(vis,nan=hi)-lo)/max(1e-6,hi-lo),0,1)*255).astype(np.uint8))
    (out/'decision.json').write_text(json.dumps(rep,indent=1))
    print(json.dumps(rep,indent=1))
if __name__=='__main__': main()
