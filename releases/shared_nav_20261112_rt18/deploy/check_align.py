"""Verify that the colour and depth images are pixel-aligned after the remap."""
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
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--output',default='runs/align')
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
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
    for i in range(a.warmup+3):
        packet=read_exact(proc.stdout,8+W*H*7)
        rgb=np.frombuffer(packet,np.uint8,W*H*3,8).reshape(H,W,3).copy()
        depth=np.frombuffer(packet,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
        if i<a.warmup: continue
        r=cv2.remap(rgb,*maps,cv2.INTER_LINEAR)
        d=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        valid=np.isfinite(d)&(d>.15)&(d<4.)
        # colour edges vs depth discontinuity (only where both valid)
        gray=cv2.cvtColor(r,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        g=cv2.magnitude(gx,gy)
        dd=np.where(valid,d,np.nan)
        dz=np.zeros_like(d); dz[1:]=np.abs(np.diff(np.nan_to_num(dd),axis=0)); dz[:,1:]=np.maximum(dz[:,1:],np.abs(np.diff(np.nan_to_num(dd),axis=1)))
        m=valid&(g>40)&(dz>0.02)
        # shift search: does shifting depth by (dx,dy) align better?
        best=None
        for dy in range(-4,5):
            for dx in range(-4,5):
                sd=np.roll(np.roll(dz,dy,axis=0),dx,axis=1)
                score=float(np.mean(sd[m])) if m.any() else 0.
                if best is None or score>best[0]: best=(score,dx,dy)
        print(json.dumps(dict(frame=i,valid_frac=float(valid.mean()),edge_pixels=int(m.sum()),
            best_shift_dxdy=[best[1],best[2]],best_score=round(best[0],4))),flush=True)
        if i==a.warmup:
            vis=cv2.cvtColor(r,cv2.COLOR_RGB2BGR).copy()
            edges=(dz>0.02)&valid
            vis[edges]=[0,0,255]
            cv2.imwrite(str(out/'overlay.png'),vis)
    proc.kill()
if __name__=='__main__': main()
