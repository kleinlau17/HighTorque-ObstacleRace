"""Warm up the camera, recognise tape + obstacles, draw them in 3D with timings."""
import argparse, json, struct, subprocess, time, math
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

def cluster_points(points, cell=.12, min_pts=25):
    """Connected components on a 2D grid, so each obstacle is fitted alone."""
    if points is None or len(points)<min_pts: return []
    xy=np.asarray(points,dtype=float)[:,:2]
    keys=np.floor(xy/cell).astype(np.int64)
    cells={}
    for i,(kx,ky) in enumerate(keys):
        cells.setdefault((int(kx),int(ky)),[]).append(i)
    seen=set(); clusters=[]
    neighbours=[(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy)!=(0,0)]
    for start in list(cells):
        if start in seen: continue
        stack=[start]; seen.add(start); members=[]
        while stack:
            cur=stack.pop(); members.extend(cells[cur])
            for dx,dy in neighbours:
                nb=(cur[0]+dx,cur[1]+dy)
                if nb in cells and nb not in seen:
                    seen.add(nb); stack.append(nb)
        if len(members)>=min_pts:
            clusters.append(np.asarray(points,dtype=float)[members])
    clusters.sort(key=len,reverse=True)
    return clusters


def fit_box(points, sx=.20, sy=.20, sz=.40):
    """Fit a fixed-size box (sx,sy,sz metres) to the obstacle points."""
    if points is None or len(points)<10: return None
    p=np.asarray(points,dtype=float)[:,:2]
    keep=np.ones(len(p),bool)
    for axis in range(2):
        lo,hi=np.percentile(p[:,axis],[2,98]); keep&=(p[:,axis]>=lo)&(p[:,axis]<=hi)
    p=p[keep]
    if len(p)<8: return None
    best=None
    for deg in range(0,91,3):
        th=np.deg2rad(deg); c,s=np.cos(th),np.sin(th)
        xr=p[:,0]*c+p[:,1]*s; yr=-p[:,0]*s+p[:,1]*c
        w=xr.max()-xr.min(); h=yr.max()-yr.min()
        excess=max(0.,w-sx)+max(0.,h-sy)
        if best is None or excess<best[0]-1e-9:
            best=(excess,deg,xr,yr)
    excess,deg,xr,yr=best
    cxr=.5*(xr.min()+xr.max()); cyr=.5*(yr.min()+yr.max())
    th=np.deg2rad(deg); c,s=np.cos(th),np.sin(th)
    cx=cxr*c-cyr*s; cy=cxr*s+cyr*c
    return dict(center=[float(cx),float(cy),None], yaw=float(th), size=[sx,sy,sz],
                excess=float(excess), points=int(len(p)))


def project(points,R,scale,offset):
    p=points@R.T
    return (p[:,:2]*scale+offset).astype(np.int32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='deploy/pro_model_config.json')
    ap.add_argument('--warmup',type=int,default=15)
    ap.add_argument('--frames',type=int,default=10)
    ap.add_argument('--output',default='runs/probe3d')
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
    attitude=np.eye(3)
    def grab():
        packet=read_exact(proc.stdout,8+W*H*7)
        ts=struct.unpack('<d',packet[:8])[0]
        rgb=np.frombuffer(packet,np.uint8,W*H*3,8).reshape(H,W,3).copy()
        depth=np.frombuffer(packet,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
        rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR); depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        depth[(depth<.15)|(depth>4.)]=np.nan
        return type('F',(),{'time':ts,'rgb':rgb,'depth':depth})(), rgb
    for _ in range(a.warmup): grab()
    lanes=[0.]; lane_ms=[]; avoid_ms=[]; box_ms=[]; white=[]; raised=[]; lanes_all=[]
    last_box=None; last_boxes=[]; last_frame=None
    z0=None
    for i in range(a.frames):
        frame,rgb=grab()
        t=time.monotonic(); obs,scene=perception.inspect_lane(frame,attitude,None,inside_start=True)
        lane_ms.append((time.monotonic()-t)*1e3)
        if perception.analysis is None: continue
        level,finite,height,xy,points,floor,zz,yaw,t0,rgb0=perception.analysis
        z0=float(zz)
        bright,dark=rgb.max(axis=2),rgb.min(axis=2)
        col=(dark>115)&(bright-dark<.25*bright)&(xy[...,0]>.30)
        print(json.dumps(dict(i=i,color_white=int(col.sum()),
            color_white_finite=int((col&finite).sum()),
            color_white_floor=int((col&floor).sum()),
            final_white=int(len(points)))),flush=True)
        white.append(np.column_stack([points[:,0],points[:,1],np.full(len(points),-z0)]))
        if perception.raised_sites is not None and len(perception.raised_sites):
            rs=perception.raised_sites
            raised.append(np.column_stack([rs[:,0],rs[:,1],-z0+rs[:,2]]))
            _t=time.monotonic()
            clusters=cluster_points(raised[-1])
            last_boxes=[b for b in (fit_box(c) for c in clusters) if b is not None]
            last_box=last_boxes[0] if last_boxes else None
            box_ms.append((time.monotonic()-_t)*1e3)
        t=time.monotonic(); avoid,ascene=perception.inspect_avoid(perception.analysis)
        avoid_ms.append((time.monotonic()-t)*1e3)
        if obs.lane is not None:
            xs=np.linspace(.15,1.1,20); lanes_all.append(np.column_stack([xs,np.polyval(obs.lane,xs),np.full(20,-z0)]))
        if obs.raw_lane is not None:
            xs=np.linspace(.15,1.1,20); lanes_all.append(np.column_stack([xs,np.polyval(obs.raw_lane,xs),np.full(20,-z0)]))
        # single-frame coloured clouds (one camera frame -> one coordinate frame)
        _level,_finite,_height,_xy,_points,_floor,_z,_yaw,_t,_rgb=perception.analysis
        _bright,_dark=rgb.max(axis=2),rgb.min(axis=2)
        _white=((_dark>115)&(_bright-_dark<.15*_bright)&(_xy[...,0]>.30)&_floor)
        _route=obs.lane if obs.lane is not None else None
        _cx=np.polyval(_route,_xy[...,0]) if _route is not None else np.zeros_like(_xy[...,0])
        _in=(_xy[...,1]>np.minimum(_cx,0.)-.30)&(_xy[...,1]<np.maximum(_cx,0.)+.30); _in|=abs(_xy[...,1])<.33
        _raised=_finite&(_height>.06)&(_height<.65)&(_xy[...,0]>.22)&(_xy[...,0]<1.15)&_in
        _w=_xy[_white]; _wc=rgb[_white]
        _r=_level[_raised]; _rc=rgb[_raised]
        last_frame=dict(
            white=[[float(_w[k,0]),float(_w[k,1]),float(-_z),int(_wc[k,0]),int(_wc[k,1]),int(_wc[k,2])] for k in range(len(_w))],
            raised=[[float(_r[k,0]),float(_r[k,1]),float(_r[k,2]),int(_rc[k,0]),int(_rc[k,1]),int(_rc[k,2])] for k in range(len(_r))],
            boxes=last_boxes, floor=float(-_z))
        cv2.imwrite(str(out/f'rgb_{i}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    proc.kill()
    def stats(a):
        if not a: return dict(n=0,mean=None,min=None,max=None,p95=None)
        return dict(n=len(a),mean=round(float(np.mean(a)),2),min=round(float(np.min(a)),2),
                    max=round(float(np.max(a)),2),p95=round(float(np.percentile(a,95)),2))
    timing=dict(lane_ms=stats(lane_ms),avoid_ms=stats(avoid_ms),box_ms=stats(box_ms))
    (out/'timing.json').write_text(json.dumps(timing,indent=2))
    (out/'scene.json').write_text(json.dumps(dict(
        white=np.vstack(white).tolist() if white else [],
        raised=np.vstack(raised).tolist() if raised else [],
        lanes=[l.tolist() for l in lanes_all],
        frame=last_frame,
        box=last_box if last_box is not None else (None if not raised else fit_box(np.vstack(raised))),
        boxes=last_boxes,
        floor=None if z0 is None else float(-z0),
        z0=None if z0 is None else float(z0),
        camera_position=cfg.get('camera_position_body_m'),
        timing=timing)))
    # ---- 3D view (manual orthographic projection) ----
    Wp,Hp=900,700
    img=np.full((Hp,Wp,3),24,np.uint8)
    az,el=math.radians(-40),math.radians(28)
    Rz=np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,math.cos(el),-math.sin(el)],[0,math.sin(el),math.cos(el)]])
    R=(Rx@Rz); scale_px=260.; offset=np.array([Wp*.45,Hp*.62])
    def draw_pts(pts,color,r):
        p=project(np.asarray(pts),R,scale_px,offset)
        for x,y in p: cv2.circle(img,(int(x),int(y)),r,color,-1)
    axis_len=.5
    for v,c,name in (((axis_len,0,0),(0,0,220),'x fwd'),((0,axis_len,0),(0,220,0),'y left'),((0,0,axis_len),(220,0,0),'z up')):
        p=project(np.array([[0,0,0],v]),R,scale_px,offset)
        cv2.line(img,tuple(p[0]),tuple(p[1]),c,2)
        cv2.putText(img,name,tuple(p[1]+np.array([4,-4])),cv2.FONT_HERSHEY_SIMPLEX,.45,c,1)
    if white: draw_pts(np.vstack(white),(0,180,0),1)
    if raised: draw_pts(np.vstack(raised),(0,0,255),2)
    for ln in lanes_all:
        p=project(ln,R,scale_px,offset)
        for j in range(len(p)-1): cv2.line(img,tuple(p[j]),tuple(p[j+1]),(200,120,0),2)
    cam_p=project(np.array([[0,0,0.0]]),R,scale_px,offset)[0]
    cv2.drawMarker(img,tuple(cam_p),(255,255,0),cv2.MARKER_TRIANGLE_UP,16,2)
    cv2.putText(img,'camera',tuple(cam_p+np.array([8,16])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
    grid=1.
    for g in np.arange(-grid,grid+1e-9,grid):
        p=project(np.array([[g,-grid,0],[g,grid,0]]),R,scale_px,offset)
        q=project(np.array([[-grid,g,0],[grid,g,0]]),R,scale_px,offset)
        cv2.line(img,tuple(p[0]),tuple(p[1]),(45,45,45),1); cv2.line(img,tuple(q[0]),tuple(q[1]),(45,45,45),1)
    cv2.putText(img,f'white {len(np.vstack(white)) if white else 0} pts  raised {len(np.vstack(raised)) if raised else 0} pts',(20,30),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
    cv2.putText(img,f'fit {timing["lane_ms"]["mean"]} ms  avoid {timing["avoid_ms"]["mean"]} ms',(20,58),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
    cv2.imwrite(str(out/'scene3d.png'),img)
    print(json.dumps(timing,indent=2))
if __name__=='__main__': main()
