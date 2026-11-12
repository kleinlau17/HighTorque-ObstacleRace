"""Capture frames and visualize what perception sees (white tape + obstacles) with timings."""
import argparse, json, struct, subprocess, time
from pathlib import Path
import cv2, numpy as np

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
    ap.add_argument('--frames',type=int,default=3)
    ap.add_argument('--output',default='/tmp/perception_probe')
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([str(Path(__file__).resolve().parent/'realsense_stream')],
                          stdout=subprocess.PIPE,stderr=open(out/'camera.log','w'),bufsize=0)
    header=bytearray()
    while not header.endswith(b'\n'): header.extend(read_exact(proc.stdout,1))
    cam=json.loads(header); (out/'camera.json').write_text(json.dumps(cam,indent=2))
    k=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    scale=float(cfg.get('perception_scale',.5)); sw,sh=int(round(W*scale)),int(round(H*scale))
    intr=[k[0,0]*scale,k[1,1]*scale,(k[0,2]+.5)*scale-.5,(k[1,2]+.5)*scale-.5]
    dst=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
    maps=cv2.initUndistortRectifyMap(k,np.asarray(cam['coefficients']),None,dst,(sw,sh),cv2.CV_32FC1)
    from navigation.visual import CameraPerception
    perception=CameraPerception(sw,sh,80.,cfg['camera_position_body_m'],
                                cfg['camera_rotation_optical_to_body'],intr)
    attitude=np.eye(3); rows=[]
    for i in range(a.frames):
        packet=read_exact(proc.stdout,8+W*H*7)
        ts=struct.unpack('<d',packet[:8])[0]
        rgb=np.frombuffer(packet,np.uint8,W*H*3,8).reshape(H,W,3).copy()
        depth=np.frombuffer(packet,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
        rd=depth
        raw_valid=float(np.mean(np.isfinite(rd)&(rd>.15)&(rd<2.7)))
        raw_zero=float(np.mean(rd==0.))
        if i==0 or i%5==0:
            print(json.dumps(dict(i=i,raw_depth_min=float(np.nanmin(rd)) if np.isfinite(rd).any() else None,
                raw_depth_max=float(np.nanmax(rd)) if np.isfinite(rd).any() else None,
                raw_depth_nan=float(np.mean(~np.isfinite(rd))),
                raw_depth_zero=raw_zero,
                raw_depth_frac_valid=raw_valid)), flush=True)
        rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR); depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        depth[(depth<.15)|(depth>4.)]=np.nan
        frame=type('F',(),{'time':ts,'rgb':rgb,'depth':depth})()
        finite=np.isfinite(depth)&(depth>.15)&(depth<2.7)
        if i==0:
            d=np.where(np.isfinite(depth),depth,np.nan)
            lo,hi=np.nanpercentile(d,[5,95]) if np.isfinite(d).any() else (0,1)
            vis=np.clip((np.nan_to_num(d,nan=hi)-lo)/max(1e-6,hi-lo),0,1)
            cv2.imwrite(str(out/'depth_0.png'),(vis*255).astype(np.uint8))
            print('depth valid fraction', float(finite.mean()), 'p5/p95', float(lo), float(hi))
        t=time.monotonic(); obs,scene=perception.inspect_lane(frame,attitude,None,inside_start=True)
        lane_ms=(time.monotonic()-t)*1e3
        avoid=None; avoid_ms=None; avoid_scene=None
        if perception.analysis is not None:
            t=time.monotonic(); avoid,avoid_scene=perception.inspect_avoid(perception.analysis)
            avoid_ms=(time.monotonic()-t)*1e3
        if perception.analysis is None:
            cv2.imwrite(str(out/f'overlay_{i}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            rows.append(dict(i=i,scene=scene,lane_ms=round(lane_ms,2),raw_depth_frac_valid=round(raw_valid,3),
                             lane_basis=perception.lane_basis,white_px=None,raised_px=None))
            continue
        # recompute the pixel masks the same way inspect_lane does, for the overlay
        level,finite,height,xy,points,floor,z0,yaw,t0,rgb0=perception.analysis
        x,y=xy[...,0],xy[...,1]
        bright,dark=rgb.max(axis=2),rgb.min(axis=2)
        white=floor&(dark>115)&(bright-dark<.15*bright)&(x>.30)
        route=obs.lane if obs.lane is not None else None
        center=np.polyval(route,x) if route is not None else np.zeros_like(x)
        in_path=(y>np.minimum(center,0.)-.30)&(y<np.maximum(center,0.)+.30); in_path|=abs(y)<.33
        raised=finite&(height>.06)&(height<.65)&(x>.22)&(x<1.15)&in_path
        img=rgb.copy()
        img[white]=[0,255,0]          # tape: green
        img[raised]=[255,0,0]         # obstacle: red
        cv2.putText(img,f'{i} {scene} {perception.lane_basis} fit={lane_ms:.1f}ms av={avoid_ms}',
                    (4,12),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,0),1)
        cv2.imwrite(str(out/f'overlay_{i}.jpg'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        rows.append(dict(i=i,scene=scene,avoid_scene=avoid_scene,lane_basis=perception.lane_basis,
            ground_reason=perception.ground.result.reason, depth_valid=round(float(finite.mean()),3),
            lane_ms=round(lane_ms,2),avoid_ms=None if avoid_ms is None else round(avoid_ms,2),
            white_px=int(white.sum()),raised_px=int(raised.sum()),
            lane=None if obs.lane is None else obs.lane.tolist(),
            raw_lane=None if obs.raw_lane is None else obs.raw_lane.tolist(),
            edges=None if obs.edges is None else obs.edges.tolist()))
    (out/'result.json').write_text(json.dumps(rows,indent=2))
    proc.kill()
    try: proc.wait(timeout=2)
    except Exception: pass
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
