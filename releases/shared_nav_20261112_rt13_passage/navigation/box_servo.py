"""Body-relative box-face measurements and a metric servo for staggered crates."""
from dataclasses import dataclass
from threading import Lock

from navigation.box_view_recovery import BoxViewRecovery, VIEW_LEASE_S, ViewTrack
from navigation.box_body import limb_points, station_window
from navigation.box_passage import PassageChoice, floor_edges
from navigation.leg_odometry import ProLegOdometry
import cv2
import numpy as np


# A direct crate match must survive two perception cycles (~.32-.40 s each
# plus jitter). .55 s expired on the first slow frame and latched
# first_crate_lost for 93% of a run, which held the servo at zero.
FIRST_BOX_LEASE_S = .9


def rot(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])


def wrap(a):
    return (a+np.pi)%(2*np.pi)-np.pi


def wall_headings(level, height, keep, components):
    """Measure square-box axes from vertical surfaces in each depth component."""
    if not len(components):
        return {}
    component_image=np.zeros(keep.shape,np.int32)
    component_image[keep]=components
    stride=2
    du=level[stride:-stride,2*stride:]-level[stride:-stride,:-2*stride]
    dv=level[2*stride:,stride:-stride]-level[:-2*stride,stride:-stride]
    normals=np.cross(du,dv)
    magnitude=np.linalg.norm(normals,axis=2)
    normal_z=np.abs(normals[...,2])/np.maximum(magnitude,1e-12)
    inside=(height[stride:-stride,stride:-stride]<.365)&(normal_z<.25)
    labels=component_image[stride:-stride,stride:-stride][inside]
    normals=normals[inside]
    result={}
    for label in np.unique(labels):
        if not label:
            continue
        walls=normals[labels==label]
        if len(walls)<25:
            continue
        angles=np.arctan2(walls[:,1],walls[:,0])
        votes=np.exp(4j*angles)
        # Both visible walls of a cube give the same fourth harmonic.
        # Reject components with no dominant square wall orientation.
        mean=np.mean(votes)
        if abs(mean)<.68:
            continue
        inliers=np.real(votes*np.conj(mean)/abs(mean))>.87
        if np.sum(inliers)>=20:
            result[label]=np.angle(np.mean(votes[inliers]))/4
    return result


@dataclass(frozen=True)
class BoxFrame:
    position: np.ndarray
    velocity: np.ndarray
    centers: tuple
    heading: float
    side: float
    stamp: float
    leg_anchor: object
    kick_until: float
    first_seen: float = float('-inf')
    first_relative: object = None
    first_seen_streak: int = 0
    corroborated_relative: object = None
    corroborated_stamp: float = float('-inf')
    previous_first_seen: float = float('-inf')
    neighbor_range: object = None
    passages: tuple = ()
    passage_seen: tuple = ()
    view: object = None
    exit_lane_seen: bool = True
    passage_boundary_seen: tuple = ()
    exit_lane_heading: object = None
    trailing_foot_sweep: bool = False

    def pass_offset(self, index):
        if self.passages:
            return self.passages[index] if index < len(self.passages) else 0.
        return self.side*.42 if index==0 else 0.


class BoxServo:
    def __init__(self):
        self.control_lock=Lock()
        self.leg_lock=Lock()
        self.latest_leg=None
        self.limb_sample=None
        self.cross_cue=None
        self.trailing_foot_crossing=None
        self.frame=None
        self.pending_frame=None
        self.heading=None
        self.pos=np.zeros(2)
        self.velocity=np.zeros(2)
        self.boxes=[]
        self.index=0
        self.phase='enter'
        self.along_started=None
        self.final_approach_still=None
        self.side=0
        self.passages=[]
        self.view_track=ViewTrack()
        self.view_recovery=BoxViewRecovery()
        self.view_override=False
        self.exit_position_reached=False
        self.exit_search_used=False
        self.exit_lane_sample=None
        self.exit_lane_votes=0
        self.far_align_since=None
        self.shift_admitted=-1
        self.passage_wait=False
        self.legs=ProLegOdometry()
        self.leg_anchor=None
        self.last_time=None
        self.last_odom=None
        self.visible=[]
        self.done=False
        self.last_cmd=np.zeros(3)
        self.target=None
        self.error=None
        self.track_mid=None
        self.corrections=[]
        self.travel=0.
        self.stationary=0.
        self.first_seen=float('-inf')
        self.first_relative=None
        self.first_seen_streak=0
        self.previous_first_seen=float('-inf')
        self.neighbor_estimates=[]
        self.spacing_samples={}
        self.spacing_corrections={}
        self.third_pair_samples=[]
        self.refined_third=False
        self.entry_hold=None
        self.entry_epoch=0
        self.recovered_entry=False
        self.restart_witness=None
        self.restart_until=float('-inf')
        self.restart_used=False
        self.restart_origin=None
        self.restart_direction=None
        self.restart_phase=None
        self.hold_since=None
        self.hold_error=None
        self.progress_stamp=None
        self.progress_error=None
        self.drive_samples=[]
        self.drive_target=None
        self.drive_visual_error=None
        self.drive_visual_speed=0.
        self.drive_floor=0.
        self.drive_rebased=False
        self.drive_anchor=None
        self.drive_witness_since=None
        self.drive_jump_stamp=None
        self.drive_moving_away=False
        self.drive_nominal=None
        self.kick_until=0.

    def _refine_spacing(self,t,associations):
        if len(self.boxes)<2 or self.phase not in ('enter','along','shift'):
            return
        measured={}
        for box,rel in associations:
            key=id(box)
            if key not in measured or np.linalg.norm(self.pos+rel-box)<np.linalg.norm(self.pos+measured[key]-box):
                measured[key]=rel
        for i in range(self.index,len(self.boxes)-1):
            back,front=self.boxes[i:i+2]
            pair=(id(self.passages[i]),id(self.passages[i+1]))
            if pair in self.spacing_corrections:
                continue
            samples=[(ts,v) for ts,v in self.spacing_samples.get(pair,()) if t-ts<4.]
            rear=measured.get(id(back))
            ahead=measured.get(id(front))
            mapped=front-back
            if rear is not None and ahead is not None and .43<rear[0]<.87:
                delta=ahead-rear
                if abs(delta[0]-mapped[0])<.045:
                    samples.append((t,float(delta[1])))
            self.spacing_samples[pair]=samples
            if self.phase!='shift' or i!=self.index or len(samples)<4 or samples[-1][0]-samples[0][0]<.44:
                continue
            distances=[v for _,v in samples]
            correction=float(np.median(distances)-mapped[1])
            if np.ptp(distances)>=.018 or not .035<abs(correction)<.105:
                continue
            # Preserve the association and the observed passage at this index;
            # the old array may still be referenced by a published observation.
            updated=front.copy()
            updated[1]+=correction
            self.boxes[i+1]=updated
            associations[:]=[(updated if box is front else box,rel) for box,rel in associations]
            self.spacing_corrections[pair]=(i,i+1,correction,t)
            if i==0:
                self.track_mid=(back[1]+updated[1])/2

    def observe(self, frame, attitude, camera, tracked, origin_yaw, lane, exit_lane_seen=None, exit_lane_heading=None):
        yaw=np.arctan2(attitude[1,0],attitude[0,0])
        plane=camera.ground.result
        if self.done:return
        xyz=(camera.rays*frame.depth[...,None])@camera.rotation32.T+camera.position32
        level=xyz@attitude.T
        if plane.valid:
            height=plane.height(level)
        else:
            floor=level[::4,::4,2]; floor=floor[np.isfinite(floor)]
            if len(floor)<120:return
            height=level[...,2]-np.percentile(floor,12)
        keep=np.isfinite(height)&(height>.065)&(height<.48)&(xyz[...,0]>.15)&(frame.depth<2.6)
        points3=level[keep]
        grid=np.zeros((240,240),np.uint8)
        cells=np.clip(np.floor(points3[:,:2]/.025+120).astype('int32'),0,239)
        grid[cells[:,1],cells[:,0]]=1
        n,labels=cv2.connectedComponents(cv2.dilate(grid,np.ones((3,3),np.uint8)))
        groups=labels[cells[:,1],cells[:,0]]
        headings=wall_headings(level,height,keep,groups) if self.heading is None else {}
        clouds=[]
        for j in range(1,n):
            points=points3[groups==j]
            if len(points)<25:continue
            pts=points[:,:2]
            if np.max(np.ptp(pts,axis=0))>.39 or np.ptp(points[:,2])<.055:continue
            rect=cv2.minAreaRect(pts[::max(1,len(pts)//1300)].astype('float32'))
            if max(rect[1])<.078:continue
            axes=cv2.boxPoints(rect);v=axes[1]-axes[0]
            a=np.arctan2(v[1],v[0])
            if self.heading is None:
                if j not in headings:
                    continue
                a=headings[j]
            a=yaw+((a-yaw+np.pi/4)%(np.pi/2)-np.pi/4)
            clouds.append((pts,a))
        if self.heading is None:
            if not clouds:return
            candidates=[(pts,a) for pts,a in clouds if np.linalg.norm(np.median(pts,axis=0))<1.60 and abs(wrap(a-yaw))<.70]
            if not candidates:return
            pts,self.heading=min(candidates,key=lambda c:np.linalg.norm(np.median(c[0],axis=0)))
        basis=rot(self.heading)
        odom=tracked.xy@rot(origin_yaw).T@basis
        dt=0. if self.last_time is None else max(0.,frame.time-self.last_time)
        delta=np.zeros(2) if self.last_odom is None else odom-self.last_odom
        if dt and (not tracked.odometry.accepted or np.linalg.norm(delta)>dt*.7):
            delta=self.velocity*dt
        last_anchor=self.leg_anchor
        with self.leg_lock:
            self.leg_anchor=self.legs.at(frame.time)
        if last_anchor is not None and self.leg_anchor is not None:
            delta=(self.leg_anchor-last_anchor)@basis
        # The controller confirmed this position against two independent
        # crate and passage-boundary captures. Transfer only the correction for
        # the exact prior pose, before integrating new displacement or velocity.
        anchor=self.drive_anchor
        if anchor is not None and anchor[0]==self.last_time:
            self.pos[1]=anchor[1]
        old=self.pos.copy();self.pos+=delta
        self.last_odom=odom;self.last_time=frame.time
        measured=[]
        lens=(camera.position32@attitude.T)[:2]@basis
        for pts,a in clouds:
            p=pts@basis
            lo,hi=np.percentile(p,[1,99],axis=0)
            if np.any(hi-lo>.255):continue
            # A square gives its longitudinal center directly from a front face.
            # Use the toe-facing lateral side when the outline is clipped.
            center=(lo+hi)/2
            center[0]=lo[0]+.10
            if hi[1]-lo[1]<.17:
                center[1]=lo[1]+.10 if center[1]>lens[1] else hi[1]-.10
            measured.append(center)
        hits=[];new=[]
        for rel in measured:
            p=self.pos+rel
            errors=[np.linalg.norm(b-p) for b in self.boxes]
            if errors and min(errors)<.24:
                i=int(np.argmin(errors));hits.append((i,rel,self.boxes[i]-p))
            elif not self.boxes or rel[0]>.40:
                new.append(rel)
        # Use only measured vertical face normals. A clipped top or a visible
        # rear slat is not a measurement of the near face of the cube.
        depth_points=level.copy()
        depth_points[...,:2]=level[...,:2]@basis
        step=2
        ddx=depth_points[step:-step,2*step:]-depth_points[step:-step,:-2*step]
        ddy=depth_points[2*step:,step:-step]-depth_points[:-2*step,step:-step]
        norm=np.cross(ddx,ddy); mag=np.linalg.norm(norm,axis=2)
        norm=norm/np.maximum(mag[...,None],1e-12)
        wall=keep[step:-step,step:-step]&(height[step:-step,step:-step]<.365)
        xy=depth_points[step:-step,step:-step,:2]
        floor_mask=(np.isfinite(height)&(abs(height)<.035)).astype(np.uint8)
        rim_mask=(height>.365)&(height<.44)&cv2.dilate(floor_mask,np.ones((5,5),np.uint8)).astype(bool)
        rim=depth_points[rim_mask,:2]
        self.corrections=[]
        for axis in range(2):
            surf=xy[wall&(abs(norm[...,axis])>.91)]
            err=[]
            for box in self.boxes:
                predicted=box-self.pos
                face=predicted[axis]+np.sign(lens[axis]-predicted[axis])*.1
                dist=surf[:,axis]-face
                match=(abs(dist)<.075)&(abs(surf[:,1-axis]-predicted[1-axis])<.13)
                vals=dist[match]
                if len(vals)<12:continue
                bins=np.round(vals/.004).astype(int)
                modal=np.argmax(np.bincount(bins+20))-20
                near=vals[abs(vals-.004*modal)<.008]
                if len(near)>=10:err.append(float(np.median(near)))
            if not err:
                # At close range the head sees the top outline after losing
                # the front wall. A rim touching observed floor supplies an
                # independent metric edge even during a lateral crossing.
                for box in self.boxes:
                    predicted=box-self.pos
                    for sign in (-1.,1.):
                        face=predicted[axis]+sign*.1
                        dist=rim[:,axis]-face
                        ids=(abs(dist)<.052)&(abs(rim[:,1-axis]-predicted[1-axis])<.106)
                        ds=dist[ids]
                        if len(ds)<9:continue
                        bins=np.rint(ds/.003).astype(int)+20
                        peak=(np.argmax(np.bincount(bins))-20)*.003
                        on=abs(ds-peak)<.007
                        if on.sum()>=9 and np.ptp(rim[ids,1-axis][on])>.046:
                            err.append(float(np.median(ds[on])))
            if err:self.pos[axis]-=np.clip(np.median(err),-.018,.018)*.58
            self.corrections.append(err)
        if self.track_mid is not None and plane.valid:
            rgb=frame.rgb
            white=(abs(height)<.02)&(rgb.min(axis=2)>115)&(rgb.max(axis=2)-rgb.min(axis=2)<.15*rgb.max(axis=2))
            floor=depth_points[white,:2]
            center=self.track_mid-self.pos[1]
            line_err=[]
            for v in (center-.25,center+.25):
                pick=floor[(abs(floor[:,1]-v)<.065)&(floor[:,0]<1.1)&(floor[:,0]>.22)]
                if len(pick)>15 and np.ptp(pick[:,0])>.18:
                    # Exclude the start of a curve and transverse tape.
                    fit=np.polyfit(pick[:,0],pick[:,1],1)
                    if abs(fit[0])<.045 and np.percentile(abs(pick[:,1]-v-np.median(pick[:,1]-v)),85)<.018:
                        line_err.append(np.median(pick[:,1]-v))
            if line_err:self.pos[1]-=.55*np.clip(np.median(line_err),-.018,.018)
        self.visible=[]
        associations=[(self.boxes[i],rel) for i,rel,_ in hits]
        for i,rel,_ in hits:self.visible.append(i)
        for rel in sorted(new,key=lambda p:p[0]):
            p=self.pos+rel
            if self.boxes and min(np.linalg.norm(b-p) for b in self.boxes)<.23:continue
            if self.boxes and p[0]<self.boxes[0][0]-.26 and self.index==0 and self.phase in ('enter','along'):
                self.boxes.insert(0,p);associations.append((p,rel));self.side=0
                self.passages.insert(0,PassageChoice())
                self.visible=[0]+[index+1 for index in self.visible]
                self.first_seen_streak=0
                continue
            if self.boxes and p[0]<self.boxes[-1][0]+.26:continue
            self.boxes.append(p);associations.append((p,rel));self.visible.append(len(self.boxes)-1)
        if not self.boxes:self.heading=None;return
        while len(self.passages)<len(self.boxes):
            self.passages.append(PassageChoice())
        self._refine_spacing(frame.time,associations)
        first_matches=[rel for box,rel in associations if box is self.boxes[0]]
        if first_matches:
            self.previous_first_seen=self.first_seen if self.first_seen_streak else float('-inf')
            self.first_relative=min(first_matches,key=lambda rel: np.linalg.norm(rel+self.pos-self.boxes[0])).copy()
            self.first_seen_streak=(self.first_seen_streak+1 if 0 < frame.time-self.first_seen <= FIRST_BOX_LEASE_S else 1)
            self.first_seen=frame.time
        else:
            self.first_seen_streak=0
        # From beside crate 0 it can leave the lens while both forward crates
        # remain visible. Two distinct landmarks must independently agree on
        # the current relative position. They can certify entry completion only.
        inferred=[]
        for later in self.boxes[1:]:
            matches=[rel for box,rel in associations if box is later]
            if matches:
                rel=min(matches,key=lambda v: np.linalg.norm(v+self.pos-later))
                inferred.append(rel+self.boxes[0]-later)
        if (self.recovered_entry and not self.refined_third and self.index==0
                and self.phase in ('along','shift') and len(self.boxes)==3 and len(inferred)==2):
            # Re-observe lateral spacing of crates 1 and 2 simultaneously.
            # Their difference cancels the robot position and any common odometry drift.
            discrepancy=float(inferred[1][1]-inferred[0][1])
            self.third_pair_samples=[(t,e) for t,e in self.third_pair_samples if frame.time-t<1.2]
            self.third_pair_samples.append((frame.time,discrepancy))
            values=[e for t,e in self.third_pair_samples]
            if (self.phase=='shift' and len(values)>=4 and frame.time-self.third_pair_samples[0][0]>.5
                    and np.ptp(values)<.013 and .035<abs(np.median(values))<.10):
                corrected=self.boxes[2].copy()
                corrected[1]+=float(np.median(values))
                self.boxes[2]=corrected
                self.refined_third=True
        self.neighbor_estimates=[v.tolist() for v in inferred]
        corroborated=None
        if len(inferred)>=2:
            measurements=np.asarray(inferred)
            spread=np.ptp(measurements,axis=0)
            # A clipped lateral side can bias one inferred centre by half a
            # box width. Both estimates must put the body on the passing side.
            side_distances=-self.side*measurements[:,1]
            if spread[0]<.065 and spread[1]<=.10 and np.all((.355<side_distances)&(side_distances<.54)):
                corroborated=np.mean(measurements,axis=0)
        if self.track_mid is None and len(self.boxes)>1:
            self.track_mid=(self.boxes[0][1]+self.boxes[1][1])/2
        target_index=self.index+int(self.phase=='shift')
        selected=self.passages[target_index] if target_index<len(self.passages) else None
        turn_clear=False
        turn_block='ground_plane_unknown'
        if plane.valid:
            edges=floor_edges(frame.rgb,depth_points,height,np.isfinite(frame.depth)) if associations else []
            valid=np.isfinite(height) & (frame.depth>.15) & (frame.depth<2.6)
            xy=depth_points[valid,:2]
            heights=height[valid]
            # The stance already supports the robot. Check floor on both sides
            # of its current heading and newly visible obstacles close to the arms.
            near=xyz[...,0]
            side=xyz[...,1]
            floor=(valid & (abs(height)<.032) & (near>.20) & (near<.72))
            close=(valid & ((height>.10) | (height<-.055))
                   & (depth_points[...,0]**2+depth_points[...,1]**2<.35**2))
            on_both_sides=(np.count_nonzero(floor & (side>0.) & (side<.25))>=12
                           and np.count_nonzero(floor & (side<0.) & (side>-.25))>=12)
            turn_clear=on_both_sides and np.count_nonzero(close)<4
            turn_block='near_obstacle_or_drop' if np.count_nonzero(close)>=4 else 'near_floor_incomplete'
            for index,box in enumerate(self.boxes):
                matches=[rel for known,rel in associations if known is box]
                if not matches or index<self.index:
                    continue
                rel=min(matches,key=lambda v: np.linalg.norm(self.pos+v-box))
                lane_offset=None
                if lane is not None:
                    to_body=basis.T@rot(yaw)
                    body_rel=rel@to_body
                    midpoint=np.array([body_rel[0],np.polyval(lane,body_rel[0])])@to_body.T
                    lane_offset=midpoint[1]-rel[1]
                choice=self.passages[index]
                choice.view_active=(index==target_index)
                choice.view_position=self.pos.copy()
                choice.observe(frame.time,edges,xy,heights,rel,FIRST_BOX_LEASE_S,lane_offset)
        offsets=tuple(p.offset for p in self.passages)
        view=self.view_track.capture(frame.time,self.pos,
                    selected.seen_at if selected else None,selected.relative if selected else None,
                    selected,selected.offset if selected else 0.,target_index,self.phase,turn_clear=turn_clear,turn_block=turn_block)
        self.side=float(np.sign(offsets[0]))
        if dt:
            v=(self.pos-old)/dt
            self.velocity=.56*self.velocity+.44*np.clip(v,-.50,.50)
            self.travel+=np.linalg.norm(self.pos-old)
        if self.index==0 and self.travel<.08 and self.last_cmd[0]>.06 and abs(self.last_cmd[1])<.16:
            self.stationary+=dt
            if self.stationary>.5:
                self.kick_until=frame.time+.45;self.stationary=-.65

        # The real publisher runs concurrently with RGB-D processing. Publish
        # a complete capture result by one reference assignment; its arrays
        # are independent of the mutable next-frame tracker.
        trailing_foot_sweep=False
        if self.index+1<len(self.boxes):
            front=self.boxes[self.index+1]
            observations=[rel for known,rel in associations if known is front]
            if observations:
                rel=min(observations,key=lambda v: np.linalg.norm(self.pos+v-front))
                error=float(front[1]-rel[1]-self.pos[1])
                self.cross_cue=(front,frame.time,error)
            cue=self.cross_cue
            if cue is not None and cue[0] is front and frame.time-cue[1]<=1.:
                # If visual and propagated side positions differ by more than
                # the entry tolerance, protect the rear foot through its full sweep.
                trailing_foot_sweep=abs(cue[2])>.035
        self.pending_frame=BoxFrame(self.pos.copy(),self.velocity.copy(),tuple(self.boxes),self.heading,
                            self.side,self.last_time,self.leg_anchor,self.kick_until,
                            self.first_seen,self.first_relative,self.first_seen_streak,corroborated,frame.time,self.previous_first_seen,
                            float(inferred[0][0]) if inferred else None,offsets,tuple(p.seen_at for p in self.passages),
                            trailing_foot_sweep=trailing_foot_sweep,passage_boundary_seen=tuple(p.boundary_stamp for p in self.passages),
                            view=view,exit_lane_heading=exit_lane_heading,exit_lane_seen=(lane is not None if exit_lane_seen is None else exit_lane_seen))

    def commit_frame(self):
        if self.pending_frame is not None:
            with self.control_lock:
                self.frame,self.pending_frame=self.pending_frame,None

    def update_legs(self,t,positions,attitude):
        with self.leg_lock:
            accepted=self.legs.update(t,positions,attitude)
            if accepted:
                self.latest_leg=(t,self.legs.xy.copy(),self.legs.velocity.copy())
                body=limb_points(positions,attitude)
                if body is not None:
                    self.limb_sample=(t,body)
            return accepted

    def command(self,t,yaw):
        with self.control_lock:
            return self._command(t,yaw,self.frame)

    @staticmethod
    def entry_evidence_lease(frame):
        # In the final deceleration, a processed capture can arrive late even
        # though every RGB-D exposure still directly measures the same crate.
        # A missed match immediately returns to the ordinary pursuit lease.
        measured=frame.first_relative
        if (frame.first_seen_streak and measured is not None
                and 0.32<measured[0]<.75 and abs(measured[1]+frame.pass_offset(0))<.055):
            return .74
        return FIRST_BOX_LEASE_S

    def observation_deadline(self,frame,deadline):
        if self.view_override and frame is not None:
            # A new floor observation can authorize a pivot even after the
            # original pursuit lease has expired. This lease belongs only
            # to the recovery action and its own frozen camera evidence.
            return min(deadline,frame.view.stamp+VIEW_LEASE_S) if frame.view is not None else frame.stamp
        if self.phase=='enter' and frame is not None:
            if (self.drive_floor>0 or self.drive_rebased) and frame.passage_boundary_seen:
                return min(deadline,min(frame.first_seen,frame.passage_boundary_seen[0])+FIRST_BOX_LEASE_S)
            return min(deadline,frame.first_seen+self.entry_evidence_lease(frame))
        return deadline

    def publication(self,frame,deadline):
        """Bind the box measurements and their expiry to this published plan."""
        recovering=self.view_override
        boosted=self.drive_floor>0 or self.drive_rebased
        strict_frame=recovering or boosted
        entry_plan=self.phase=='enter' and not recovering
        epoch=self.entry_epoch
        target_index=self.index+int(self.phase=='shift')
        deadline=self.observation_deadline(frame,deadline)
        def velocity(stamp,planned,attitude):
            if strict_frame and (self.frame is not frame or self.view_override!=recovering):
                return np.zeros(3)
            if self.frame is not None and self.frame.pass_offset(target_index)!=frame.pass_offset(target_index):
                return np.zeros(3)
            if stamp>=deadline or epoch!=self.entry_epoch:
                return np.zeros(3)
            if entry_plan and stamp-frame.first_seen>=FIRST_BOX_LEASE_S and np.linalg.norm(planned[:2])>.085:
                return np.zeros(3)
            if not self.control_lock.acquire(blocking=False):
                return np.zeros(3) if strict_frame or (entry_plan and (self.entry_hold or epoch!=self.entry_epoch)) else planned
            try:
                # A planning thread may already have advanced to its next
                # observation. Never advance that plan using this older frame.
                if self.frame is not frame:
                    if strict_frame or self.view_override or self.drive_floor>0 or self.frame is None or self.frame.pass_offset(target_index)!=frame.pass_offset(target_index):
                        return np.zeros(3)
                    return np.zeros(3) if entry_plan and (self.entry_hold or epoch!=self.entry_epoch) else planned
                yaw=np.arctan2(attitude[1,0],attitude[0,0])
                guided=self._command(stamp,yaw,frame)
                if self.view_override!=recovering or self.done or (not boosted and self.drive_floor>0):
                    return np.zeros(3)
                return planned if guided is None else guided
            finally:
                self.control_lock.release()
        return velocity

    def _entry_feedback(self,t,frame):
        """Observe lateral motion against the directly measured crate in the passage frame."""
        offset=frame.pass_offset(0)
        self.drive_floor=0.
        self.drive_moving_away=False
        target=(tuple(frame.centers[0]),offset)
        if target!=self.drive_target:
            self.drive_samples.clear()
            self.drive_witness_since=None
            self.drive_rebased=False
            self.drive_target=target
        seen=frame.first_seen
        if (not offset or not frame.passage_boundary_seen or not frame.first_seen_streak
                or frame.first_relative is None or not np.all(np.isfinite(frame.first_relative))
                or not 0<=t-seen<FIRST_BOX_LEASE_S or frame.passage_boundary_seen[0]!=seen):
            self.drive_samples.clear()
            self.drive_visual_error=None
            self.drive_visual_speed=0.
            self.drive_witness_since=None
            return
        optical=float(frame.first_relative[1]+offset)
        distance=float(frame.first_relative[0])
        residual=float(frame.centers[0][1]+offset-frame.position[1]-optical)
        self.drive_visual_error=optical
        samples=self.drive_samples
        if samples and seen!=samples[-1][0]:
            prior,prev_error,prev_range,_=samples[-1]
            dt=seen-prior
            if (not 0<dt<=FIRST_BOX_LEASE_S or abs(distance-prev_range)>.65*dt+.025
                    or abs(optical-prev_error)>.60*dt+.020):
                samples.clear()
                self.drive_witness_since=None
                self.drive_rebased=False
                if 0<dt<=FIRST_BOX_LEASE_S:
                    self.drive_jump_stamp=seen
        if not samples or samples[-1][0]!=seen:
            self.drive_visual_speed=0.
            if samples:
                prior,prev_error,_,_=samples[-1]
                self.drive_visual_speed=float(np.clip((prev_error-optical)/(seen-prior),-.45,.45))
                if abs(prev_error-optical)>.004:
                    self.drive_witness_since=None
            samples.append((seen,optical,distance,residual))
            self.drive_samples[:]=[s for s in samples if seen-s[0]<=1.]
        if abs(optical)<=.035:
            self.drive_witness_since=None
            return
        recent=[s for s in samples if seen-s[0]<=.62]
        # Growing error over independent observations contradicts the requested
        # direction. Wait until the same measured corridor settles or converges.
        old_stamp,old_error,_,_=recent[0]
        self.drive_moving_away=(len(recent)>=3 and seen-old_stamp>=.30 and old_error*optical>0
                                and abs(optical)-abs(old_error)>.035
                                and (self.entry_hold=='visual_motion_away' or
                                     (self.drive_nominal is not None and self.drive_nominal*optical>0)))
        # Publisher ticks cannot add observations. A slow series of small steps
        # and a single observed step both cancel the stationary witness.
        if max(s[1] for s in recent)-min(s[1] for s in recent)>.010:
            self.drive_witness_since=None
            return
        if self.drive_witness_since is None:
            self.drive_witness_since=seen
        stalled=seen-self.drive_witness_since
        if len(samples)>=3 and stalled>=.48:
            self.drive_floor=min(.32,.18+.30*(stalled-.48))

    def _entry_allowed(self,t,frame,lateral):
        # A newly processed RGB-D frame can still contain no observation of crate 0.
        # A latched stop requires new camera evidence. Elapsed time never rearms it.
        self._entry_feedback(t,frame)
        age=t-frame.first_seen
        fresh=0<=age<FIRST_BOX_LEASE_S or (0<=age<self.entry_evidence_lease(frame) and abs(lateral)<.055)
        visual_error=(frame.first_relative[1]+frame.pass_offset(0)
                      if frame.first_relative is not None else None)
        conflicting=(fresh and frame.first_seen_streak and visual_error is not None
                     and (abs(visual_error-lateral)>.11 or
                          (self.drive_visual_error is not None and lateral*visual_error<=0
                           and abs(visual_error-lateral)>.045)))
        supported=self.drive_visual_error is not None
        error=abs(self.drive_visual_error) if supported else abs(lateral)
        if self.entry_hold:
            observed=(0<=age<FIRST_BOX_LEASE_S and frame.first_seen_streak>=2
                      and frame.previous_first_seen>self.hold_since)
            agrees=(visual_error is not None and abs(visual_error-lateral)<.065)
            samples=self.drive_samples
            confirmed=(supported and len(samples)>=2 and samples[-2][0]>self.hold_since
                       and abs(samples[-1][3]-samples[-2][3])<.04)
            if observed and confirmed and (conflicting or not agrees):
                # These new crate-and-passage observations independently give
                # the same pose correction. Servo on the direct measurement.
                self.drive_rebased=True
            if self.drive_rebased:
                agrees=confirmed
            progressed=(self.entry_hold!='no_progress' or confirmed or error<=self.hold_error-.018 or error<.045)
            if self.entry_hold=='visual_motion_away' and (not confirmed or self.drive_moving_away):
                return False
            if not (observed and agrees and progressed):
                return False
            self.entry_hold=None
            self.entry_epoch+=1
            self.recovered_entry=True
            self.restart_witness=None
            self.progress_stamp=None
        if not fresh or (self.drive_rebased and not supported):
            reason='first_crate_lost'
        elif self.drive_moving_away:
            reason='visual_motion_away'
        elif self.drive_jump_stamp==frame.first_seen or (conflicting and not self.drive_rebased):
            reason='visual_pose_disagreement'
        else:
            if self.progress_stamp is None or error<.06 or error<=self.progress_error-.018:
                self.progress_stamp=t
                self.progress_error=error
            reason='no_progress' if not supported and t-self.progress_stamp>=.95 else None
        if reason:
            self.entry_hold=reason
            self.entry_epoch+=1
            self.hold_since=t
            self.hold_error=error
            return False
        return True

    def _restart_stalled_entry(self,t,frame,pos,vel,dx,lateral,vx):
        # After stopping, the stock gait can remain in double support even as
        # forward velocity is requested. Authorize one bounded yaw pulse only
        # when live crate range and odometry both confirm no forward progress.
        measured_range=None
        if self.phase=='enter' and frame.first_seen_streak and 0<=t-frame.first_seen<=.40:
            measured_range=frame.first_relative[0]
        elif self.phase=='along' and self.index==0 and 0<=t-frame.corroborated_stamp<=.40:
            measured_range=frame.neighbor_range
        if (self.restart_used or not self.recovered_entry or measured_range is None
                or dx<.16 or abs(lateral)>.08 or vx<.16):
            self.restart_witness=None
            return
        if self.restart_witness is not None:
            since,first_x,first_range=self.restart_witness
            if abs(pos[0]-first_x)<.030 and abs(measured_range-first_range)<.065:
                if t-since>=.65 and abs(vel[0])<.07 and abs(vel[1])<.08:
                    self.restart_until=t+.45
                    self.restart_used=True
                    self.restart_phase=self.phase
                return
        self.restart_witness=(t,pos[0],measured_range)

    def _view_command(self,t,yaw,frame,pos,offset,waiting):
        previous_side=self.view_recovery.last_side
        same_crate=(frame.view is not None and self.view_recovery.index==frame.view.index
                    and self.view_recovery.identity==frame.view.identity)
        command=self.view_recovery.command(t,yaw,frame.heading,pos,self.index,self.phase,offset,waiting,frame.view)
        self.view_override=command is not None
        if command is not None:
            self.last_cmd=command
            return command.copy()
        if (self.phase=='along' and same_crate and previous_side and offset*previous_side<0):
            # The alternative was authorized from in front of the crate. Put
            # the normal approach/crossing controller back in charge before
            # commanding translation; the old along target is beyond the box.
            if self.index==0:
                self.phase='enter'
                self.progress_stamp=None
            else:
                self.index-=1
                self.phase='shift'
                self.shift_admitted=-1
            self.entry_epoch+=1
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        return None

    def _command(self,t,yaw,frame):
        self.view_override=False
        if self.entry_hold and frame is not None and not frame.centers:
            return np.zeros(3)
        if frame is None or not frame.centers or self.done or self.index>=len(frame.centers):return None
        i=self.index
        center=frame.centers[i]
        leg=self.latest_leg
        vel=frame.velocity
        if frame.leg_anchor is not None and leg is not None and 0<=t-leg[0]<=.10:
            basis=rot(frame.heading)
            vel=leg[2]@basis
            pos=frame.position+(leg[1]-frame.leg_anchor)@basis+vel*(t-leg[0])
        else:
            pos=frame.position+frame.velocity*min(.30,max(0.,t-frame.stamp))
        offset=frame.pass_offset(i)
        next_offset=frame.pass_offset(i+1)
        self.passage_wait=self.phase not in ('lane_entry','shift') and not offset
        if self.phase=='shift':
            # Re-identify this specific crate before acting on its surveyed gap.
            if not next_offset:
                self.shift_admitted=-1
            elif i+1<len(frame.passage_seen) and 0<=t-frame.passage_seen[i+1]<FIRST_BOX_LEASE_S:
                self.shift_admitted=i+1
            self.passage_wait=not offset or self.shift_admitted!=i+1
        view_offset=next_offset if self.phase=='shift' else offset
        if not self.passage_wait:
            self.far_align_since=None
        if self.passage_wait:
            if self.phase=='enter':
                self._entry_feedback(t,frame)
                # A distant approaching crate gives the lane heading before
                # either side can be surveyed. Retain the initial alignment
                # at once; use the wider search if it fails to expose the gap.
                if self.far_align_since is None:
                    self.far_align_since=t
                angle=wrap(frame.heading-yaw)
                if (t-self.far_align_since<1.1 and frame.first_relative is not None
                        and frame.first_relative[0]>.75 and 0<=t-frame.first_seen<FIRST_BOX_LEASE_S
                        and frame.view is not None and frame.view.turn_space and abs(angle)>.08):
                    self.last_cmd=np.array([0.,0.,np.clip(2.4*angle,-.50,.50)])
                    return self.last_cmd.copy()
            recovery=self._view_command(t,yaw,frame,pos,view_offset,True)
            if recovery is not None:
                return recovery
            self.last_cmd=np.zeros(3)
            if (self.phase=='enter' and frame.first_relative is not None
                    and 0<=t-frame.first_seen<FIRST_BOX_LEASE_S and frame.first_relative[0]>.55):
                angle=wrap(frame.heading-yaw)
                if abs(angle)>.08:
                    self.last_cmd[2]=np.clip(2.4*angle,-.50,.50)
            return self.last_cmd.copy()
        y=center[1]+offset
        if i+1<len(frame.centers):
            gate=(center[0]+frame.centers[i+1][0])/2-.029
        else:
            gate=center[0]+.38
        if self.phase=='enter':
            target_x=center[0]-.36
        elif self.phase=='along':
            target_x=gate
        elif self.phase=='lane_entry':
            target_x=gate+.04
            y=.5*(frame.centers[-2][1]+frame.centers[-1][1])
        else:
            y=frame.centers[i+1][1]+next_offset
            # As the torso moves across the lane, the trailing outside foot
            # approaches the previous crate. Advance through the midpoint
            # gradually after the leading arm has crossed the next crate.
            total=abs(y-(center[1]+offset))
            fraction=np.clip(1.-abs(y-pos[1])/max(total,.001),0.,1.)
            target_x=gate+.063*np.clip((fraction-.15)/.58,0.,1.)
            # During a change of sides the two box faces constrain fore/aft
            # movement. Account for the current swing foot and protruding wrist.
            sample=self.limb_sample
            if sample is not None and 0<=t-sample[0]<=.13:
                if frame.trailing_foot_sweep:
                    self.trailing_foot_crossing=i
                low,high=station_window(sample[1],rot(frame.heading),pos,y,center,frame.centers[i+1],self.trailing_foot_crossing==i)
                if low>high:
                    self.passage_wait=True
                    recovery=self._view_command(t,yaw,frame,pos,view_offset,True)
                    if recovery is not None:
                        return recovery
                    self.last_cmd=np.zeros(3)
                    return self.last_cmd.copy()
                margin=.015
                if low+margin<=high-margin:
                    target_x=np.clip(target_x,low+margin,high-margin)
                elif np.isfinite(low+high):
                    target_x=(low+high)*.5
        dx=target_x-pos[0];lateral=y-pos[1]
        vx0,vy0=vel
        entry_ok=self.phase!='enter' or self._entry_allowed(t,frame,lateral)
        if self.phase=='enter' and entry_ok and self.drive_rebased:
            lateral=self.drive_visual_error
            vy0=self.drive_visual_speed
            self.drive_anchor=(frame.stamp,frame.centers[0][1]-frame.first_relative[1])
        elif self.phase!='enter':
            self.drive_rebased=False
        ahead=dx-.55*vx0
        across=lateral-.33*vy0
        vx=np.clip(1.7*ahead,-.18,.22)
        vy=np.clip(2.3*across,-.26,.26)
        self.drive_nominal=float(vy)
        if (self.phase=='enter' and entry_ok and self.drive_floor>abs(vy) and
                lateral*self.drive_visual_error>0):
            vy=np.sign(self.drive_visual_error)*max(abs(vy),self.drive_floor)
        else:
            self.drive_floor=0.
        if self.phase in ('shift','lane_entry'):
            vx=np.clip(3*(dx-.39*vx0)-.075*min(1.,abs(vy)/.15),-.20,.13)
        elif abs(ahead)<.020:vx=0.
        # The pulse governor starts above 5.5 cm, but the transition is
        # below 4 cm. A robot at rest between them must be able to finish.
        still_short=(self.phase=='along' and .04<=dx<=.055 and abs(vx0)<.04 and abs(vy)<.035)
        if not still_short:
            self.final_approach_still=None
        elif self.final_approach_still is None:
            self.final_approach_still=t
        settled_short=still_short and t-self.final_approach_still>=.55
        if self.phase=='along' and abs(vy)<.035 and 0.<vx<.15 and dx<.12:
            # The stock velocity adapter otherwise rounds every positive
            # straight request up to .20. Govern the final pulse explicitly.
            elapsed_us=round((t-self.along_started)*1_000_000)
            pulse_on=elapsed_us%300_000<105_000
            vx=.20 if (settled_short and ahead>.025) or (dx>.055 and ahead>.045 and pulse_on) else 0.
        self.target=[float(target_x),float(y)];self.error=[float(dx),float(lateral)]
        if self.phase=='enter' and not entry_ok:
            # Stop lateral drive; a measured neighboring pair may establish that
            # entry is already complete, without extending the lateral pursuit.
            pair=frame.corroborated_relative
            last=frame.first_relative
            if (self.entry_hold=='first_crate_lost' and pair is not None and last is not None
                    and 0<=t-frame.corroborated_stamp<=FIRST_BOX_LEASE_S and 0<=t-frame.first_seen<=2.5
                    and .24<last[0]<.80 and abs(last[1]+offset)<.12
                    and abs(pair[0]-.36)<.10 and abs(pair[1]+offset)<.06
                    and abs(lateral)<.035 and abs(dx)<.065 and abs(vy0)<.10):
                self.phase='along'
                self.along_started=t
                self.entry_hold=None
                self.recovered_entry=True
                self.restart_witness=None
                self.entry_epoch+=1
            else:
                recovery=self._view_command(t,yaw,frame,pos,view_offset,True)
                if recovery is not None:
                    return recovery
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        at_exit=(self.phase=='lane_entry' and abs(lateral)<.04 and abs(vy0)<.10 and pos[0]>center[0]+.30)
        if at_exit:
            self.exit_position_reached=True
        elif self.phase!='lane_entry' or pos[0]<center[0]+.26:
            self.exit_position_reached=False
        awaiting_lane=self.exit_position_reached and not frame.exit_lane_seen
        if awaiting_lane:
            self.exit_search_used=True
            self.exit_lane_sample=None
            self.exit_lane_votes=0
        elif self.exit_position_reached and self.exit_search_used:
            # Resume on two distinct consistent lane captures. A scan can be
            # turning away from the lane when the first edge reappears.
            angle=frame.exit_lane_heading
            if angle is not None and (self.exit_lane_sample is None or self.exit_lane_sample[0]!=frame.stamp):
                previous=self.exit_lane_sample
                consistent=(previous is not None and 0<frame.stamp-previous[0]<=.45
                            and abs(wrap(angle-previous[1]))<.21)
                self.exit_lane_votes=self.exit_lane_votes+1 if consistent else 1
                self.exit_lane_sample=(frame.stamp,angle)
            if (angle is None or self.exit_lane_votes<2 or abs(wrap(angle-yaw))>.14):
                self.view_override=True
                self.view_recovery.action='realign'
                self.last_cmd=(self.view_recovery.turn_to(t,yaw,angle,frame.view)
                               if angle is not None and frame.view is not None else np.zeros(3))
                return self.last_cmd.copy()
        recovery=self._view_command(t,yaw,frame,pos,view_offset,awaiting_lane)
        if recovery is not None:
            return recovery
        if awaiting_lane:
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        if (self.phase=='enter' and 0<=t-frame.first_seen<FIRST_BOX_LEASE_S and abs(lateral)<.035 and dx<.065
                and (self.drive_visual_error is None or abs(self.drive_visual_error)<.055)):
            self.phase='along'
            self.along_started=t
        elif self.phase=='along' and abs(vx0)<.09 and (abs(dx)<.040 or
                (settled_short and i==2 and frame.exit_lane_seen and pos[0]>center[0]+.30)):
            # With the third crate already behind and lane tape in view, the
            # lane-entry servo itself can finish the last 4–5.5 cm.
            if i+1<len(frame.centers):self.phase='shift'
            elif len(frame.centers)>=3:
                self.phase='lane_entry';return self.last_cmd.copy()
        elif self.phase=='shift' and abs(lateral)<.035 and abs(vy0)<.11:
            self.phase='along'
            self.along_started=t
            self.index+=1
        if self.phase=='lane_entry' and self.exit_position_reached and frame.exit_lane_seen:
            self.done=True;return None
        heading_error=wrap(frame.heading-yaw)
        body=np.array([vx,vy])@rot(frame.heading).T@rot(yaw)
        w=np.clip(2.4*heading_error,-.5,.5)
        self._restart_stalled_entry(t,frame,pos,vel,dx,lateral,vx)
        if t<frame.kick_until and abs(w)<.20:
            w=.23
        if self.phase==self.restart_phase and t<self.restart_until and abs(w)<.20:
            if self.phase=='enter':
                w=.23
            elif frame.neighbor_range is not None and 0<=t-frame.corroborated_stamp<=.40:
                if self.restart_origin is None:
                    self.restart_origin=yaw
                    to_axis=wrap(frame.heading-yaw)
                    self.restart_direction=np.sign(to_axis) if abs(to_axis)>.015 else frame.side
                w=self.restart_direction*.45 if abs(wrap(yaw-self.restart_origin))<.10 else 0.
        if self.phase=='enter' and t-frame.first_seen>=FIRST_BOX_LEASE_S and np.linalg.norm(body)>.085:
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        self.last_cmd=np.r_[body,w]
        return self.last_cmd.copy()

    def diagnostic(self):
        return dict(**self.view_recovery.diagnostic(),crate_phase=self.phase,crate_index=self.index,crate_side=float(self.side),
                    crate_passage_gaps=[p.widths for p in self.passages],crate_passage_offsets=[p.offset for p in self.passages],crate_passage_wait=self.passage_wait,
                    crate_entry_hold=self.entry_hold,crate_refined_third=self.refined_third,crate_spacing_corrections=list(self.spacing_corrections.values()),crate_restart_used=self.restart_used,crate_restart_until=self.restart_until if self.restart_used else None,crate_first_match_time=None if self.first_relative is None else self.first_seen,
                    crate_first_relative=None if self.frame is None or self.frame.first_relative is None else self.frame.first_relative.tolist(),
                    crate_drive_visual_error=self.drive_visual_error,crate_drive_visual_speed=self.drive_visual_speed,
                    crate_drive_floor=self.drive_floor,crate_drive_nominal=self.drive_nominal,crate_drive_rebased=self.drive_rebased,
                    crate_neighbors=self.neighbor_estimates,
                    crate_neighbor_relative=None if self.frame is None or self.frame.corroborated_relative is None else self.frame.corroborated_relative.tolist(),
                    crate_pose=self.pos.tolist(),crate_velocity=self.velocity.tolist(),crate_heading=self.heading,
                    crate_centers=[v.tolist() for v in self.boxes],crate_visible=self.visible,
                    crate_residuals=self.corrections,crate_target=self.target,crate_error=self.error,crate_done=self.done)
