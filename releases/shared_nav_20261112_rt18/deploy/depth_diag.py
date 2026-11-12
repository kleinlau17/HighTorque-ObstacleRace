"""Check where the depth is invalid: overall, by image region, and on bright tape pixels."""
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
    proc=subprocess.Popen(['deploy/realsense_stream'],stdout=subprocess.PIPE,
                          stderr=open('runs/live/diag_cam.log','w'),bufsize=0)
    header=bytearray()
    while not header.endswith(b'\n'): header.extend(read_exact(proc.stdout,1))
    cam=json.loads(header)
    k=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    scale=float(cfg.get('perception_scale',.5)); sw,sh=int(round(W*scale)),int(round(H*scale))
    intr=[k[0,0]*scale,k[1,1]*scale,(k[0,2]+.5)*scale-.5,(k[1,2]+.5)*scale-.5]
    dst=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
    maps=cv2.initUndistortRectifyMap(k,np.asarray(cam['coefficients']),None,dst,(sw,sh),cv2.CV_32FC1)
    for _ in range(25):
        p=read_exact(proc.stdout,8+W*H*7)
    ts=struct.unpack('<d',p[:8])[0]
    rgb=np.frombuffer(p,np.uint8,W*H*3,8).reshape(H,W,3).copy()
    depth=np.frombuffer(p,'<f4',W*H,8+W*H*3).reshape(H,W).copy()
    rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR); depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
    proc.kill()
    valid=(np.isfinite(depth))&(depth>.15)&(depth<2.7)
    bright=rgb.max(axis=2); dark=rgb.min(axis=2)
    white=(dark>115)&((bright-dark)<.15*bright)
    h,w=valid.shape
    def frac(m): return round(float(valid[m].mean()),3)
    rep=dict(
      depth_valid_total=round(float(valid.mean()),3),
      left_half=frac(np.s_[:, :w//2]), right_half=frac(np.s_[:, w//2:]),
      bottom_half=frac(np.s_[h//2:, :]), top_half=frac(np.s_[:h//2, :]),
      bottom_left=frac(np.s_[h//2:, :w//2]), bottom_right=frac(np.s_[h//2:, w//2:]),
      white_px=int(white.sum()), white_depth_valid=frac(white),
      white_depth_min=float(np.nanmin(depth[white])) if white.any() else None,
      white_depth_max=float(np.nanmax(depth[white])) if white.any() else None)
    vis=np.where(valid,depth,np.nan)
    lo,hi=np.nanpercentile(vis,[5,95]) if np.isfinite(vis).any() else (0,1)
    img=np.clip((np.nan_to_num(vis,nan=hi)-lo)/max(1e-6,hi-lo),0,1)
    cv2.imwrite('runs/live/diag_depth.png',(img*255).astype(np.uint8))
    cv2.imwrite('runs/live/diag_rgb.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite('runs/live/diag_valid.png',(valid*255).astype(np.uint8))
    print(json.dumps(rep,indent=1))
if __name__=='__main__': main()
