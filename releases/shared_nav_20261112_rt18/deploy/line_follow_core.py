"""Short camera-centered lane-following trial, without global/body extrinsics.

Requires both boundaries. Ground geometry comes from each real RGB-D image.
This is a limited straight-lane experiment, not the full competition navigator.
"""
import math
import cv2
import numpy as np
from deploy.ground_calibration import fit_ground


class TurnThenWalk:
    """Camera feedback with hysteresis, using the measured effective turn input."""
    def __init__(self):
        self.turning=False;self.direction=0.;self.turn_started=0.;self.pause_until=0.;self.previous=[]

    def command(self, model, speed, now, turn_completed=False):
        path=model.get('path_samples_ground_m')
        x=max(.4,path[0][0]) if path else .4
        target=float(np.interp(x,np.asarray(path)[:,0],np.asarray(path)[:,1])) if path else float(np.polyval(model['center_line'],x))
        bearing=math.atan2(target,x)
        self.previous=(self.previous+[bearing])[-3:]
        filtered=float(np.median(self.previous))
        if self.turning and turn_completed:
            self.turning=False;self.pause_until=now+.5;self.previous=[]
        elif not self.turning and now>=self.pause_until and abs(filtered)>math.radians(6):
            self.turning=True
            self.direction=math.copysign(1.,filtered)
            self.turn_started=now
        if now<self.pause_until:return [0.,0.,0.],'settling',filtered
        if self.turning:return [0.,0.,self.direction],'turning',filtered
        return [float(speed),0.,0.],'walking',filtered


def sample_lane_boundaries(clouds):
    """Preserve tape corners instead of extrapolating one quadratic per side."""
    paths=[]
    for points in clouds:
        bins=np.floor(points[:,0]/.04).astype(int);rows=[]
        for key in np.unique(bins):
            p=points[bins==key]
            if len(p)>=3:rows.append(np.median(p,axis=0))
        if len(rows)<6:raise ValueError('sampled boundary lacks support')
        paths.append(np.asarray(rows))
    low=max(p[0,0] for p in paths);high=min(p[-1,0] for p in paths)
    if high-low<.22 or low>.45:raise ValueError('common near-field boundaries not visible')
    xs=np.linspace(low,high,max(7,int((high-low)/.035)))
    sides=[np.interp(xs,p[:,0],p[:,1]) for p in paths]
    sides.sort(key=lambda y:float(np.median(y)))
    center=(sides[0]+sides[1])/2
    slope=np.gradient(center,xs)
    slope=np.array([np.median(slope[max(0,i-1):i+2]) for i in range(len(slope))])
    widths=(sides[1]-sides[0])/np.sqrt(1+slope*slope)
    if (np.any(sides[1]<=sides[0]) or not .40<float(np.median(widths))<.62
        or np.percentile(widths,5)<.32 or np.percentile(widths,95)>.75
        or np.percentile(widths,90)-np.percentile(widths,10)>.20):
        raise ValueError('sampled boundaries do not form the expected lane')
    near=xs<=low+.12
    near_line=np.polyfit(xs[near],center[near],1)
    return dict(center=np.polyfit(xs,center,2),lines=[np.polyfit(xs,y,2) for y in sides],
                widths=widths,heading=math.atan(float(near_line[0])),offset=float(near_line[1]),
                path=np.column_stack((xs,center)))


def observe_lane(rgb, depth, camera, speed=.04):
    rgb=np.asarray(rgb);depth=np.asarray(depth)
    h,w=depth.shape
    v,u=np.indices((h,w))
    rays=np.stack([(u-camera['cx'])/camera['fx'],(v-camera['cy'])/camera['fy'],np.ones_like(depth)],-1)
    valid=np.isfinite(depth)&(depth>.2)&(depth<2.8)
    f=rgb.astype(float)
    green=valid&(f[:,:,1]>1.25*f[:,:,0])&(f[:,:,1]>1.1*f[:,:,2])&(v>h*.3)
    cloud=rays*depth[:,:,None]
    sample=cloud[green&((u%6)==0)&((v%6)==0)]
    if len(sample)<300:raise ValueError('insufficient visible green floor')
    if len(sample)>800:sample=sample[np.linspace(0,len(sample)-1,800).astype(int)]
    plane=fit_ground(sample,iterations=60)
    if not .4<plane['camera_height_m']<.85 or not 25<plane['downward_angle_deg']<55:
        raise ValueError('camera/floor pose outside trial envelope')
    rotation=np.asarray(plane['camera_to_ground_rotation'])
    level=cloud@rotation.T;level[:,:,2]+=plane['camera_height_m']
    white=(valid&(abs(level[:,:,2])<.03)&(f.min(2)>135)
           &((f.max(2)-f.min(2))<.25*f.max(2)))
    keep=white&(level[:,:,0]>.30)&(level[:,:,0]<.90)&(abs(level[:,:,1])<.8)
    points=level[keep,:2]
    if len(points)<30:raise ValueError('two white boundaries not visible')
    _,ids=np.unique(np.round(points/.008).astype(int),axis=0,return_index=True)
    remaining=points[ids];rng=np.random.default_rng(8);lines=[]
    # Most images contain two separate continuous tape components. Keep their
    # identities so RANSAC cannot accidentally join parts of opposite curves.
    joined=cv2.morphologyEx(keep.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels=cv2.connectedComponents(joined)
    components=[];raw_components=[]
    for label in range(1,count):
        p=level[keep&(labels==label),:2]
        if len(p)<20 or np.ptp(p[:,0])<.20:continue
        raw_components.append(p.copy())
        _,which=np.unique(np.round(p/.008).astype(int),axis=0,return_index=True);p=p[which]
        linear=np.polyfit(p[:,0],p[:,1],1)
        degree=1 if np.percentile(abs(np.polyval(linear,p[:,0])-p[:,1]),90)<.008 else 2
        for _ in range(3):
            curve=np.r_[np.zeros(2-degree),np.polyfit(p[:,0],p[:,1],degree)]
            residual=abs(np.polyval(curve,p[:,0])-p[:,1]);p=p[residual<.02]
            if len(p)<12:break
        if len(p)>=12 and np.ptp(p[:,0])>=.20 and abs(curve[0])<3 and abs(curve[1])<2:
            components.append(curve)
    sampled=None
    if len(raw_components)==2:
        try:sampled=sample_lane_boundaries(raw_components)
        except ValueError:pass
    if sampled is not None:lines=sampled['lines']
    elif len(components)==2:lines=components
    for _ in range(0 if len(lines)==2 else 2):
        best=None;score=0
        for trial in range(90):
            if len(remaining)<12:break
            degree=1 if trial<40 else 2
            chosen=remaining[rng.choice(len(remaining),degree+1,replace=False)]
            if np.ptp(chosen[:,0])<.20:continue
            if degree==2 and np.min(np.diff(np.sort(chosen[:,0])))<.025:continue
            try:curve=np.r_[np.zeros(2-degree),np.polyfit(chosen[:,0],chosen[:,1],degree)]
            except np.linalg.LinAlgError:continue
            if abs(curve[0])>3 or abs(curve[1])>2:continue
            inliers=abs(remaining[:,1]-np.polyval(curve,remaining[:,0]))<.015
            quality=float(inliers.sum())-abs(curve[0])*2
            if quality>score:best=inliers;score=quality;best_degree=degree
        if best is None or score<12 or np.ptp(remaining[best,0])<.25:
            raise ValueError('boundary fit lacks longitudinal support')
        line=np.r_[np.zeros(2-best_degree),np.polyfit(remaining[best,0],remaining[best,1],best_degree)]
        lines.append(line);remaining=remaining[~best]
    lines.sort(key=lambda c:float(np.polyval(c,.65)))
    center=np.mean(lines,axis=0);samples=np.array([.35,.55,.75])
    widths=(np.polyval(lines[1],samples)-np.polyval(lines[0],samples))/np.sqrt(1+np.polyval(np.polyder(center),samples)**2)
    if sampled is None and (np.any((widths<.40)|(widths>.62)) or np.ptp(widths)>.10):
        raise ValueError('boundary pair does not form the expected lane')
    heading=math.atan(float(center[1]));offset=float(center[2])
    if sampled is not None:
        center=sampled['center'];widths=sampled['widths'];heading=sampled['heading'];offset=sampled['offset']
    if abs(offset)>.15:
        raise ValueError('lane offset outside tracking envelope: offset=%.4f m heading=%.2f deg'%(offset,math.degrees(heading)))
    center_y=(np.interp(level[:,:,0],sampled['path'][:,0],sampled['path'][:,1]) if sampled is not None else np.polyval(center,level[:,:,0]))
    ahead=(level[:,:,0]>.25)&(level[:,:,0]<.95)&(abs(level[:,:,1]-center_y)<.30)
    if np.count_nonzero(valid&ahead&(level[:,:,2]>.08))>20:
        raise ValueError('raised object in forward corridor')
    lookahead=.55;target_y=float(np.polyval(center,lookahead))
    yaw_limit=.04 if speed<=.04 else .5
    yaw=float(np.clip(2.*speed*target_y/(lookahead**2+target_y**2),-yaw_limit,yaw_limit))
    return dict(command=[float(speed),0.,yaw],center_line=center.tolist(),boundaries=[c.tolist() for c in lines],
                path_samples_ground_m=None if sampled is None else sampled['path'].tolist(),fit_method='sampled_boundaries' if sampled is not None else 'polynomial',
                lane_width_m=float(widths.mean()),camera_center_offset_m=offset,heading_rad=heading,large_heading=abs(heading)>math.radians(20),
                floor=plane,white_pixels=int(keep.sum()),frame='camera-ground; body optical extrinsics unverified')


def choose_command(now, manual, nav, manual_active, joy_age, nav_age, trial_elapsed,
                   allowed, execute=True, duration=6.,speed_limit=.04,yaw_limit=.04):
    """Pure watchdog/arbitration decision, independent of image processing."""
    if not math.isfinite(speed_limit) or not 0<speed_limit<=.4:return [0.,0.,0.],'invalid_speed_limit'
    if not math.isfinite(yaw_limit) or not 0<yaw_limit<=1.:return [0.,0.,0.],'invalid_yaw_limit'
    if not allowed or joy_age>.2:return [0.,0.,0.],'inhibited'
    if manual_active:
        values=np.asarray(manual,dtype=float)
        if values.shape!=(3,) or not np.isfinite(values).all():return [0.,0.,0.],'invalid_manual'
        return list(manual),'manual'
    if not execute or nav_age>.2 or trial_elapsed>=duration:return [0.,0.,0.],'stopped'
    values=np.asarray(nav,dtype=float)
    if values.shape!=(3,) or not np.isfinite(values).all():return [0.,0.,0.],'invalid'
    return [float(np.clip(values[0],0,speed_limit)),0.,float(np.clip(values[2],-yaw_limit,yaw_limit))],'navigation'
