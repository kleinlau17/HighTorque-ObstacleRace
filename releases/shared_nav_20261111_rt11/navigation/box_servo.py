"""Body-relative box-face measurements and a metric servo for staggered crates."""
from dataclasses import asdict, dataclass, replace
from threading import Lock

from navigation.leg_odometry import ProLegOdometry
from navigation.box_tape import BoxTape
from navigation.box_steps import BoxSteps
from navigation.box_corridor import BoundaryTracks, Edge, clipped_paint_pair, floor_edges, measured_pass_side, forward_clearance
from navigation.box_tuning import BoxTuning
from navigation.box_body import limb_points, station_window
import cv2
import numpy as np


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
    identified: tuple = ()
    axis_evidence: tuple = (False,False)
    tape_fix: bool = False
    tape_longitudinal: bool = False
    direct_axes: tuple = ((),())
    linked: tuple = ()
    visual_xy: tuple = (None,None)
    # Right and left boundaries are signed metric lines in the crate-heading frame.
    edges: tuple = (None,None)
    previous_edges: tuple = (None,None)
    previous_stamp: float = -np.inf
    forward_room: float = 0.
    depth_status: str = 'depth_unobserved'
    progress_x: object = None
    # Fresh range to a crate side that was measured together with this stripe.
    registered_edges: tuple = (None,None)
    previous_registered: tuple = (None,None)
    paint_stamp: tuple = (-np.inf,-np.inf)
    depth_diagnostics: object = None


class BoxServo:
    def __init__(self):
        self.control_lock=Lock()
        self.invalid_projection_time = -np.inf
        self.leg_lock=Lock()
        self.latest_leg=None
        self.frame=None
        self.pending_frame=None
        self.heading=None
        self.pos=np.zeros(2)
        self.velocity=np.zeros(2)
        self.boxes=[]
        self.index=0
        self.phase='enter'
        self.side=0
        self.legs=ProLegOdometry()
        self.leg_anchor=None
        self.last_time=None
        self.last_odom=None
        self.visible=[]
        self.done=False
        self.last_cmd=np.zeros(3)
        self.guard_report=None
        self.target=None
        self.error=None
        self.track_mid=None
        self.corrections=[]
        self.travel=0.
        self.stationary=0.
        self.kick_until=0.
        self.tape=BoxTape()
        self.boundaries=BoundaryTracks()
        self.edge_landmarks={}
        self.edge_proof={}
        self.paint_time=[-np.inf,-np.inf]
        self.entry_side_measurement=None
        self.steps=BoxSteps()
        self.previous_entrance_look=None
        self.tuning=BoxTuning()
        self.tuning_revision=0
        self.pending_tuning=None
        self.panel_gap=None
        self.release_requested=False
        self.hold_reason='initializing'
        self.step_count=0
        self.step_cap=None
        self.entry_station=None
        self.entry_clearance=None
        self.boundary_lock=None
        self.last_aligned_line=None
        self.aligned_box=None
        self.covisible=set()
        self.visual_odom_anchor=None
        self.body_sample=None
        self.last_body_gap=-np.inf
        self.station_bounds=None
        self.body_history=[]
        self.observation_since=None
        self.search_heading=None
        self.search_direction=1.
        self.search_pulses=0

    def observe(self, frame, attitude, camera, tracked, origin_yaw, lane, rgb_usable=None):
        yaw=np.arctan2(attitude[1,0],attitude[0,0])
        plane=camera.ground.result
        if self.done:return
        if self.frame is not None:
            missing='no_box_geometry'
            if not plane.valid:
                distances=frame.depth[::4,::4]
                missing='no_ground_plane'
                if not np.any((distances>.12)&(distances<4.)):
                    if rgb_usable is None:
                        pixels=(frame.rgb.max(axis=2)>5)&(frame.rgb.min(axis=2)<250)
                        rgb_usable=np.count_nonzero(pixels)>=frame.depth.size*.01
                    missing='no_depth_returns' if rgb_usable else 'no_usable_rgbd'
            self.pending_frame=replace(self.frame,stamp=frame.time,identified=(),axis_evidence=(False,False),
                                       tape_fix=False,tape_longitudinal=False,direct_axes=((),()),linked=(),visual_xy=(None,None),
                                       edges=(None,None),previous_edges=(None,None),previous_stamp=-np.inf,
                                       forward_room=0.,depth_status='depth_unobserved',progress_x=None,
                                       registered_edges=(None,None),previous_registered=(None,None),
                                       depth_diagnostics={'reason':missing})
        xyz=(camera.rays*frame.depth[...,None])@camera.rotation32.T+camera.position32
        level=xyz@attitude.T
        if not plane.valid:return
        height=plane.height(level)
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
        face_ids=[set(),set()]
        face_ranges=({}, {})
        visual_xy=[None,None]
        for axis in range(2):
            surf=xy[wall&(abs(norm[...,axis])>.91)]
            face_hits=[]
            for idx,box in enumerate(self.boxes):
                predicted=box-self.pos
                face=predicted[axis]+np.sign(lens[axis]-predicted[axis])*.1
                dist=surf[:,axis]-face
                match=(abs(dist)<.075)&(abs(surf[:,1-axis]-predicted[1-axis])<.13)
                vals=dist[match]
                if len(vals)<12:continue
                bins=np.round(vals/.004).astype(int)
                modal=np.argmax(np.bincount(bins+20))-20
                near=vals[abs(vals-.004*modal)<.008]
                if len(near)>=10:
                    face_hits.append((idx,float(np.median(near))))
            if not face_hits:
                # At close range the head sees the top outline after losing
                # the front wall. A rim touching observed floor supplies an
                # independent metric edge even during a lateral crossing.
                for idx,box in enumerate(self.boxes):
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
                            face_hits.append((idx,float(np.median(ds[on]))))
            if len(face_hits)>1 and np.ptp([delta for _,delta in face_hits])>.042:
                # Clipped near faces sometimes match a slat on a different box.
                # Averaging two conflicting ranges moves the station between them.
                supported=[hit for hit in face_hits if abs(hit[1])<.022]
                face_hits=supported if supported and np.ptp([v for _,v in supported])<.028 else []
            err=[delta for _,delta in face_hits]
            for idx,delta in face_hits:
                face_ranges[axis][id(self.boxes[idx])]=self.boxes[idx][axis]-self.pos[axis]+delta
            face_ids[axis].update(idx for idx,_ in face_hits)
            if err:
                visual_xy[axis]=float(self.pos[axis]-np.median(err))
                maximum,gain=(.038,.85) if axis==0 else (.018,.58)
                self.pos[axis]-=np.clip(np.median(err),-maximum,maximum)*gain
            self.corrections.append(err)
        line_err=[]
        floor=np.empty((0,2))
        if self.track_mid is not None and plane.valid:
            rgb=frame.rgb
            white=(abs(height)<.02)&(rgb.min(axis=2)>115)&(rgb.max(axis=2)-rgb.min(axis=2)<.15*rgb.max(axis=2))
            floor=depth_points[white,:2]
            center=self.track_mid-self.pos[1]
            for v in (center-.25,center+.25):
                pick=floor[(abs(floor[:,1]-v)<.065)&(floor[:,0]<1.1)&(floor[:,0]>.22)]
                if len(pick)>15 and np.ptp(pick[:,0])>.18:
                    # Exclude the start of a curve and transverse tape.
                    fit=np.polyfit(pick[:,0],pick[:,1],1)
                    if abs(fit[0])<.045 and np.percentile(abs(pick[:,1]-v-np.median(pick[:,1]-v)),85)<.018:
                        line_err.append(np.median(pick[:,1]-v))
            if line_err:
                if visual_xy[1] is None:visual_xy[1]=float(self.pos[1]-np.median(line_err))
                self.pos[1]-=.55*np.clip(np.median(line_err),-.018,.018)
        self.visible=[]
        for i,rel,_ in hits:self.visible.append(i)
        for rel in sorted(new,key=lambda p:p[0]):
            p=self.pos+rel
            if self.boxes and min(np.linalg.norm(b-p) for b in self.boxes)<.23:continue
            if self.boxes and p[0]<self.boxes[0][0]-.26 and self.index==0 and self.phase in ('enter','along'):
                self.boxes.insert(0,p);self.side=0
                self.covisible={(a+1,b+1) for a,b in self.covisible}
                self.visible=[v+1 for v in self.visible]+[0]
                face_ids=[{j+1 for j in ids} for ids in face_ids]
                continue
            if self.boxes and p[0]<self.boxes[-1][0]+.26:continue
            self.boxes.append(p);self.visible.append(len(self.boxes)-1)
        if not self.boxes:self.heading=None;return
        identified=set(self.visible)
        for idx,rel,err in hits:
            # An ID is usable as a visual anchor only within its own metric
            # footprint. The looser association above remains for tracking.
            if np.linalg.norm(err)>.105:
                identified.discard(idx)
        x_seen,y_seen=(bool(group) for group in self.corrections)
        current_reference=identified | (face_ids[0]&face_ids[1])
        implied=[self.boxes[idx]-rel for idx,rel,_ in hits if idx in identified]
        if implied:
            observed_position=np.median(implied,axis=0)
            for axis in range(2):
                if visual_xy[axis] is None:visual_xy[axis]=float(observed_position[axis])
        self.covisible.update((a,b) for a in current_reference for b in current_reference
                              if a!=b and abs(self.boxes[a][1]-self.boxes[b][1])<=.55)
        connected={self.index}
        for _ in self.boxes:
            connected.update(b for a,b in self.covisible if a in connected)
        tape_correction,tape_x=self.tape.match(floor,self.pos, bool(identified) and x_seen)
        if tape_correction is not None and not identified:
            if not x_seen and tape_x:
                visual_xy[0]=float(self.pos[0]+tape_correction[0])
                self.pos[0]+=.28*tape_correction[0]
            if not (y_seen or line_err):
                visual_xy[1]=float(self.pos[1]+tape_correction[1])
                self.pos[1]+=.28*tape_correction[1]
        if self.track_mid is None and len(self.boxes)>1:
            self.track_mid=(self.boxes[0][1]+self.boxes[1][1])/2
        if not self.side or (self.index==0 and self.steps.count==0):
            if len(self.boxes)>1:
                self.side=np.sign(self.boxes[1][1]-self.boxes[0][1])
            elif lane is not None:
                p=(self.boxes[0]-self.pos)@basis.T@rot(yaw)
                self.side=np.sign(np.polyval(lane,p[0])-p[1])
        if dt:
            v=(self.pos-old)/dt
            self.velocity=.56*self.velocity+.44*np.clip(v,-.50,.50)
            self.travel+=np.linalg.norm(self.pos-old)
        if self.index==0 and self.travel<.08 and self.last_cmd[0]>.06 and abs(self.last_cmd[1])<.16:
            self.stationary+=dt
            if self.stationary>.5:
                self.kick_until=frame.time+.45;self.stationary=-.65

        valid=np.isfinite(frame.depth)&(frame.depth>.12)&(frame.depth<4.)
        lines=floor_edges(frame.rgb,depth_points,height,valid)
        # Match current measurements after insertion/remapping of a nearer
        # first crate. A pre-insertion ID would associate the wrong white edge.
        relative={}
        errors={}
        for rel in measured:
            distances=np.linalg.norm(np.asarray(self.boxes)-self.pos-rel,axis=1)
            idx=int(np.argmin(distances))
            if distances[idx]<min(.105,errors.get(idx,np.inf)):
                relative[idx]=rel
                errors[idx]=distances[idx]
        if self.index==0 and (not self.side or self.steps.count==0):
            if 0 in relative:
                measured_side=measured_pass_side(lines,relative[0])
                if measured_side:
                    self.entry_side_measurement=(self.boxes[0],float(measured_side))
            if self.entry_side_measurement is not None and self.entry_side_measurement[0] is self.boxes[0]:
                self.side=self.entry_side_measurement[1]
        anchors=[(self.side*(-1 if idx%2 else 1),rel) for idx,rel in relative.items() if self.side]
        bounds=self.boundaries.observe(lines,frame.time,self.pos,anchors)
        ranged={}
        for idx,box in enumerate(self.boxes):
            ident=id(box)
            if ident in face_ranges[0] and ident in face_ranges[1]:
                ranged[idx]=(face_ranges[0][ident],face_ranges[1][ident])
        registered=self._registered_edges(frame.time,bounds,relative,depth_points,height,valid,ranged)
        # Use a far curved stripe only as a present paint witness. Its tangent
        # is unsuitable for projecting the body alignment back to the robot.
        clipped=clipped_paint_pair(lines,anchors)
        for k,edge in enumerate(bounds):
            reference=registered[k]
            if edge is not None or (reference and any(
                    abs(e.at(e.near)-reference.at(e.near))<(.030 if e.near<.88 else .050) and
                    abs(e.slope-reference.slope)<.40 and e.far-e.near>.069
                    for e in ([clipped[k]] if clipped[k] is not None else [])+lines)):
                self.paint_time[k]=frame.time
        body,depth_diagnostics=self._capture_body(frame.time)
        room,depth_status=forward_clearance(depth_points,height,valid,body,frame.depth,camera,
                                            attitude,basis,plane,bounds,self.tuning.forward_step/100.,anchors,
                                            diagnostics=depth_diagnostics)
        if body is None:
            depth_diagnostics['reason']=depth_diagnostics['body_evidence']
        # Longitudinal motion cannot be inferred from a straight tape. Use actual
        # range to a currently associated crate, or independently accepted visual odometry.
        progress_x=visual_xy[0]
        if progress_x is None and relative:
            idx=min(relative,key=lambda j:abs(j-self.index))
            progress_x=float(self.boxes[idx][0]-relative[idx][0])
        if tracked.odometry.accepted:
            if progress_x is not None:
                self.visual_odom_anchor=(frame.time,float(odom[0]),progress_x)
            elif self.visual_odom_anchor and 0<frame.time-self.visual_odom_anchor[0]<.6:
                _,origin,anchor=self.visual_odom_anchor
                progress_x=float(anchor+odom[0]-origin)
        else:
            self.visual_odom_anchor=None
        prior=self.frame
        prior_stamp=prior.stamp if prior is not None else -np.inf
        prior_edges=prior.edges if prior is not None else (None,None)

        # The real publisher runs concurrently with RGB-D processing. Publish
        # a complete capture result by one reference assignment; its arrays
        # are independent of the mutable next-frame tracker.
        self.pending_frame=BoxFrame(self.pos.copy(),self.velocity.copy(),tuple(self.boxes),self.heading,
                            self.side,self.last_time,self.leg_anchor,self.kick_until,
                            tuple(sorted(identified)),(x_seen,y_seen),bool(line_err) or tape_correction is not None,bool(tape_x),
                            tuple(tuple(sorted(v)) for v in face_ids),tuple(sorted(connected&current_reference)),tuple(visual_xy),
                            tuple(bounds),prior_edges,prior_stamp,room,depth_status,progress_x,
                            registered,prior.registered_edges if prior else (None,None),tuple(self.paint_time),
                            depth_diagnostics)

    def _registered_edges(self,stamp,bounds,relative,points,height,valid,ranged):
        # The width of a 20 cm crate gives a bounded center from a directly
        # measured front or roof. Reject a severely clipped outline. A side
        # plane is independently sufficient when its normal and extent agree.
        x,y=points[...,0],points[...,1]
        measured={}
        for idx,(cx,cy) in relative.items():
            surfaces=valid & (((abs(x-(cx-.10))<.031)&(height>.05)&(height<.377)) |
                              ((abs(x-cx)<.118)&(abs(height-.40)<.019))) & (abs(y-cy)<.125)
            if np.count_nonzero(surfaces)<36:continue
            lo,hi=np.percentile(y[surfaces],[1,99])
            if .164 < hi-lo < .225:
                measured[idx]=(cx,(hi+lo)*.5)
                continue
            pairs=[]
            for sign in (-1.,1.):
                local=surfaces & (abs(y-(cy+sign*.10))<.022) & (height<.377)
                if local.sum()<32:continue
                yy=y[local]
                bins=np.rint((yy-cy-sign*.10)/.003).astype(int)+8
                peak=cy+sign*.10+.003*(np.argmax(np.bincount(bins))-8)
                on=local & (abs(y-peak)<.004)
                if on.sum()>=32 and np.ptp(x[on])>.043 and np.ptp(height[on])>.065:
                    pairs.append(np.median(y[on])-sign*.10)
            if pairs and np.ptp(pairs)<.018:
                measured[idx]=(cx,float(np.median(pairs)))
        if self.index>=1:
            for idx,rel in ranged.items():
                measured.setdefault(idx,rel)
        result=[]
        for k,edge in enumerate(bounds):
            estimates=[]
            for idx,(cx,cy) in measured.items():
                key=(id(self.boxes[idx]),k)
                samples=self.edge_landmarks.setdefault(key,[])
                if edge is not None and edge.near<.87:
                    # Store a bounded sample history, replacing a single
                    # contradictory extrapolation only after agreement.
                    offset=edge.at(cx)-cy
                    samples.append((stamp,edge.slope,offset))
                    del samples[:-9]
                if len(samples)<2 or not 0<=stamp-samples[-1][0]<14.:
                    continue
                slopes=np.array([q[1] for q in samples])
                offsets=np.array([q[2] for q in samples])
                agree=abs(offsets-np.median(offsets))<.020
                if np.count_nonzero(agree)<2:continue
                slope=np.median(slopes[agree])
                est=np.median(offsets[agree])+cy-slope*cx
                estimates.append((idx,est,slope))
            # A single consistent metric ID is preferable to combining two
            # faces with different occlusion bias. Preserve the previous ID.
            prior=self.edge_proof.get(k)
            estimates.sort(key=lambda v: (v[0]!=(prior[0] if prior else None),abs(v[0]-self.index),v[0]))
            if estimates and (len(estimates)==1 or np.ptp([a[1] for a in estimates])<.055):
                i,offset,slope=estimates[0]
                self.edge_proof[k]=(i,stamp,float(offset),float(slope))
                same=(prior is not None and prior[0]==i and 0<stamp-prior[1]<=.55)
                result.append(Edge(float(offset),float(slope),0.,0.) if same else None)
            else:
                self.edge_proof.pop(k,None)
                result.append(None)
        return tuple(result)

    def commit_frame(self):
        if self.pending_frame is not None:
            with self.control_lock:
                self.frame,self.pending_frame=self.pending_frame,None

    def update_legs(self,t,positions,attitude):
        with self.leg_lock:
            accepted=self.legs.update(t,positions,attitude)
            if accepted:
                self.latest_leg=(t,self.legs.xy.copy(),self.legs.velocity.copy())
                points=limb_points(positions,attitude)
                if points is not None:
                    # Keep raw poses longer than a vision cycle; the 18 Hz ROS
                    # callback keeps running while the Jetson processes RGB-D.
                    self.body_history=[prev for prev in self.body_history if t-prev[0]<=1.5]
                    self.body_history.append((t,points,self.legs.xy.copy()))
                    if self.body_sample is not None and t-self.body_sample[0]>.13:
                        self.last_body_gap=t
                    self.body_sample=(t,np.vstack([p for ts,p,_ in self.body_history if t-ts<.22]))
            return accepted

    def _capture_body(self, stamp):
        with self.leg_lock:
            history=self.body_history
            latest=self.body_sample
            last_gap=self.last_body_gap
        # The individual arrays are immutable and update_legs replaces the list
        # on every update. Projection can finish without holding the ROS lock.
        info=dict(body_capture_age_s=None,body_latest_offset_s=None,body_evidence='missing_capture_body')
        if not history or latest is None:
            return None,info
        newest,live_sweep=latest
        info['body_latest_offset_s']=float(newest-stamp)
        before=[pose for pose in history if pose[0]<=stamp]
        if not before:
            return None,info
        capture_t,_,capture_xy=before[-1]
        info['body_capture_age_s']=float(stamp-capture_t)
        # One normal encoder period is 56 ms. A long gap cannot describe the
        # arms at exposure time, even if encoders are publishing again now.
        if stamp-capture_t>.10:
            info['body_evidence']='stale_capture_body'
            return None,info
        if capture_t<last_gap:
            info['body_evidence']='interrupted_body_history'
            return None,info
        if newest-capture_t<.22:
            info['body_evidence']='capture_in_live_sweep'
            return live_sweep,info
        between=[pose for pose in history if pose[0]>=capture_t]
        # For a delayed image, evaluate the sweep at exposure as well as all
        # subsequent observed poses. Translate them into the exposure frame
        # using support-foot odometry. Include current raw reach in addition.
        sweep=[p for ts,p,_ in before if stamp-ts<.22]
        for _,points,xy in between[1:]:
            moved=points.copy()
            moved[:,:2]+=xy-capture_xy
            sweep.append(moved)
        info['body_evidence']='capture_and_live_sweep'
        return np.vstack([*sweep,live_sweep]),info

    def command(self,t,yaw):
        with self.control_lock:
            return self._command(t,yaw,self.frame)

    def publication(self,frame,deadline):
        """Bind the box measurements and their expiry to this published plan."""
        def retain_look_lease(move):
            if move and move.get('measured_look') and move['frame_stamp']==frame.stamp:
                # A replacement plan must not lengthen the authorizing capture's lease.
                move['look_deadline']=min(move.get('look_deadline',deadline),deadline)
                move['until']=min(move['until'],move['look_deadline'])

        def velocity(stamp,planned,attitude):
            if stamp>=deadline:
                return np.zeros(3)
            if not self.control_lock.acquire(blocking=False):
                return np.zeros(3)
            try:
                # A planning thread may already have advanced to its next
                # observation. Never advance that plan using this older frame.
                if self.frame is not frame:
                    return np.zeros(3)
                yaw=np.arctan2(attitude[1,0],attitude[0,0])
                move=self.steps.active
                retain_look_lease(move)
                guided=self._command(stamp,yaw,frame)
                if self.steps.active is not move:
                    retain_look_lease(self.steps.active)
                return np.zeros(3) if guided is None else guided
            finally:
                self.control_lock.release()
        def published(stamp,vector):
            with self.control_lock:
                self.steps.published(stamp,vector)
        velocity.published=published
        return velocity

    def queue_tuning(self, revision, values):
        settings=BoxTuning.read(values)
        with self.control_lock:
            if revision<=self.tuning_revision or (self.pending_tuning and revision<=self.pending_tuning[0]):
                return
            self.pending_tuning=(revision,settings)
            if self.frame is None or self.done:
                self._adopt_tuning()

    def _adopt_tuning(self):
        if self.pending_tuning is not None:
            self.tuning_revision,self.tuning=self.pending_tuning
            self.pending_tuning=None

    def guard_publication(self,stamp,source):
        # The sole motor publisher can search even while perception is waiting.
        self.guard_report=(stamp,source)

    def panel_status(self, now):
        with self.control_lock:
            frame=self.frame
            fresh=(frame is not None and frame.stamp>self.invalid_projection_time and 0<=now-frame.stamp<=.50)
            move=self.steps.active
            moving=bool(move and move['sent'] and self.steps.last_publication_nonzero
                        and frame is not None and frame.stamp>self.invalid_projection_time
                        and now<move['until'] and (fresh or self.steps.completing_retry(now,frame) or
                            (self.hold_reason=='scan_box_area' and move['frame_stamp']==frame.stamp)))
            gap=self.panel_gap
            gate=self.hold_reason if fresh or moving else 'no_current_geometry'
            if (fresh or moving) and move is not None and gate in ('confirmed_path','lateral_step'):
                gate=self.steps.reason
            report=self.guard_report
            if report is not None and 0<=now-report[0]<=.20 and report[1]=='navigation_recovery':
                moving=True
                gate='scan_box_area'
            return dict(applied=asdict(self.tuning),revision=self.tuning_revision,
                        phase=self.phase,index=self.index,done=self.done,started=bool(frame and frame.centers),
                        gate=gate,
                        depth_detail=(frame.depth_diagnostics or {}).get('reason') if fresh else None,
                        gap=gap[1] if fresh and gap and gap[0]==frame.stamp else None,
                        count=self.steps.count,moving=moving)

    def invalidate_projection(self, stamp):
        with self.control_lock:
            self.invalid_projection_time = max(self.invalid_projection_time, stamp)

    def _command(self,t,yaw,frame):
        if frame is not None and frame.centers and not self.done and (
                frame.stamp<=self.invalid_projection_time or yaw is None or not np.isfinite(yaw)):
            self.hold_reason='no_current_geometry'
            if self.steps.active is not None or frame.stamp<=self.invalid_projection_time:
                self.steps.stop(t)
            self.last_cmd = np.zeros(3)
            return self.last_cmd.copy()
        if frame is None or not frame.centers or self.done or self.index>=len(frame.centers):return None
        leg=self.latest_leg
        vel=frame.velocity
        if frame.leg_anchor is not None and leg is not None and 0<=t-leg[0]<=.10:
            basis=rot(frame.heading)
            vel=leg[2]@basis
            pos=frame.position+(leg[1]-frame.leg_anchor)@basis+vel*(t-leg[0])
        else:
            pos=frame.position+frame.velocity*min(.30,max(0.,t-frame.stamp))
        reason=self.steps.check(t,frame,pos,self._motion_xy(frame),yaw)
        if reason:
            self.hold_reason=reason
            return np.zeros(3)
        if self.steps.active is None:
            self._adopt_tuning()
        phase=(self.phase,self.index)
        self.step_cap=None
        vector=self._drive(t,yaw,frame,pos,vel) if frame.side else self._hold('no_passage_side')
        if self.release_requested and self.steps.active is None:
            self.done=True
            return None
        if phase!=(self.phase,self.index) or self.release_requested:
            self._reset_observation()
            if self.phase=='shift' and self.entry_clearance is None and self.station_bounds is not None:
                # Calibrate settling compensation at the first stop between crates.
                # This is the measured remaining fore/aft margin of the limbs,
                # in the frame of the observed boxes, after braking on approach.
                self.entry_clearance=float(self.station_bounds[1]-frame.progress_x)
            self.steps.stop(t)
            self.hold_reason='phase_observation'
            return np.zeros(3)
        if vector is None:return None
        vector=self._observe_when_blocked(t,frame,yaw,vector)
        local=vector[:2]@rot(yaw).T@rot(frame.heading)
        axis=1 if self.hold_reason=='lateral_step' else 0
        if np.linalg.norm(local)<.015 and abs(vector[2])>.02:axis=2
        sign=np.sign(local[axis]) if axis<2 else np.sign(vector[2])
        step_cm=self.tuning.side_step if axis==1 else self.tuning.forward_step if axis==0 else 6.3
        requested_motion=bool(np.any(np.abs(vector)>1.e-5))
        distance=step_cm/100.
        if axis==0 and frame.depth_status=='depth_limited':
            distance=min(distance,frame.forward_room-.015)
        if axis==0 and self.step_cap is not None:
            distance=min(distance,self.step_cap)
        soft_range=self.hold_reason=='line_follow_unseen_path' and self.step_cap is not None
        previous_count=self.steps.count
        vector=self.steps.apply(t,frame,pos,vector,axis,sign,distance,self._motion_xy(frame),yaw,
                                soft_range=soft_range)
        if self.steps.count!=previous_count:
            previous=self.previous_entrance_look
            self.previous_entrance_look=None
            if (axis==2 and self.phase=='enter' and self.hold_reason=='look_for_opposite_line'
                    and frame.edges[int(self._pass_side(frame)>0)] is None):
                move=self.steps.active
                weak=0
                if (previous and previous['sent'] and previous['sign']==sign
                        and abs(wrap(yaw-previous['yaw']))<.035):
                    weak=previous['weak_looks']+1
                move['weak_looks']=weak
                move['measured_look']=weak>=2
                self.previous_entrance_look=move
        if requested_motion and self.steps.active is None and not np.any(vector):
            self.hold_reason=self.steps.reason
        self.step_count=self.steps.count
        self.last_cmd=vector.copy()
        return vector

    def _motion_xy(self,frame):
        # Both boundaries measure the same signed lateral coordinate, even
        # when the next box is on the other side of the lane.
        for side in (self._pass_side(frame),-self._pass_side(frame)):
            edge=frame.edges[int(side>0)]
            if edge is not None:
                return (frame.progress_x,side*.25-edge.offset/np.hypot(1.,edge.slope))
        if self.phase in ('along','shift','lane_entry'):
            side=self._pass_side(frame)
            edge=frame.registered_edges[int(side>0)]
            if edge is not None:
                return (frame.progress_x,side*.25-edge.offset/np.hypot(1.,edge.slope))
        return (frame.progress_x,None)

    def recovery_direction(self):
        # Prefer the established search or the known open side of this crate.
        if self.search_heading is not None:
            return self.search_direction
        return self._pass_side(self.frame) if self.frame is not None else 0.

    def _pass_side(self,frame):
        index=self.index+(self.phase=='shift')
        return frame.side*(-1 if index%2 else 1)

    def _hold(self,reason):
        self.hold_reason=reason
        return np.zeros(3)

    def _reset_observation(self):
        self.observation_since=None
        self.search_heading=None

    def _observe_when_blocked(self,t,frame,yaw,vector):
        # Keep the ordinary line/depth decision first, including immediate
        # braking if a formerly valid translation loses its evidence.
        if np.any(vector):
            # A newly measured box gives a specific direction to look for the
            # stripe. Give that turn the same priority as a verified step.
            self._reset_observation()
            return vector
        move=self.steps.active
        completing=(move is not None and move['axis']==2 and move['frame_stamp']==frame.stamp and t<move['until'])
        if t<frame.stamp or (t-frame.stamp>.50 and not completing):
            return self._hold('no_current_geometry')
        if (frame.depth_diagnostics or {}).get('reason')=='no_usable_rgbd':
            self.observation_since=None
            return self._hold('no_current_geometry')
        if self.search_heading is None:
            if self.observation_since is None:
                self.observation_since=frame.stamp
            if frame.stamp-self.observation_since<.90:
                return vector
            self.search_heading=yaw
            self.search_pulses=0
            toward_axis=wrap(frame.heading-yaw)
            self.search_direction=(np.sign(toward_axis) if abs(toward_axis)>.10 else
                                   self._pass_side(frame) or 1.)
        # The IMU limit is relative to the start of this search, including
        # across repeated pulses and +/-pi wraparound. Each individual turn
        # also goes through BoxSteps' actual-zero and post-stop capture gate.
        offset=wrap(yaw-self.search_heading)
        if self.search_direction*offset>=.25:
            self.search_direction*=-1.
            self.search_pulses=0
        if move is not None:
            if move['axis']!=2:
                return vector
            if move['sign']!=self.search_direction or abs(wrap(yaw-move['yaw']))>=.15:
                return self._hold('observe_after_turn')
        elif self.steps.zero_stamp is not None and frame.stamp-self.steps.zero_stamp<.22:
            return self._hold('observe_after_turn')
        else:
            # A planted gait can unwind most of a turn while settling. Explore
            # the other side too even if repeated pulses never reach the limit.
            if self.search_pulses>=3 and self.search_direction*offset>-.25:
                self.search_direction*=-1.
                self.search_pulses=0
            self.search_pulses+=1
        self.hold_reason='scan_box_area'
        return np.array([0.,0.,self.search_direction*.80])

    def _look_for_edge(self,frame,yaw,pos):
        # Only rotate while the camera still sees a box. Aim at the empty-side
        # tape ahead of the body; never authorize translation from this estimate.
        if not frame.identified and not frame.linked:
            return self._hold('wait_opposite_line')
        side=self._pass_side(frame)
        idx=self.index+(self.phase=='shift')
        if idx>=len(frame.centers):
            return self._hold('wait_next_box')
        lateral=frame.centers[idx][1]-pos[1]+side*.40
        look_yaw=frame.heading+np.clip(np.arctan2(lateral,.70),-.46,.46)
        error=wrap(look_yaw-yaw)
        if abs(error)<.09:
            return self._hold('wait_opposite_line')
        self.hold_reason='look_for_opposite_line'
        return np.array([0.,0.,np.copysign(.80,error)])

    def _drive(self,t,yaw,frame,pos,vel):
        # A slow capture can still complete the repeat it authorized. It
        # cannot authorize another step after the robot stops.
        move=self.steps.active
        if move and move.get('measured_look'):
            # Two same-direction looks gained less than two degrees each.
            # Finish this look on IMU feedback while its original capture owns it.
            if frame.edges[int(self._pass_side(frame)>0)] is not None:
                return self._hold('observe_after_turn')
            if (frame.identified or frame.linked) and frame.stamp<=t<move.get('look_deadline',move['frame_stamp']+.50):
                self.hold_reason='look_for_opposite_line'
                return np.array([0.,0.,move['sign']*.80])
        if not 0<=t-frame.stamp<=.50 and not self.steps.completing_retry(t,frame):
            return self._hold('no_current_geometry')
        i=self.index
        has_next=i+1<len(frame.centers)
        gate=None
        if self.phase=='along':
            center=frame.centers[i]
            gate=(center[0]+frame.centers[i+1][0])/2-.060 if has_next else center[0]+.38
            body=self.body_sample
            self.station_bounds=None
            if has_next and body is not None and 0<=t-body[0]<=.22:
                target_y=frame.centers[i+1][1]-self._pass_side(frame)*.40
                lo,hi=station_window(body[1],rot(frame.heading),pos,target_y,center,frame.centers[i+1])
                self.station_bounds=(float(np.clip(lo,-10,10)),float(np.clip(hi,-10,10)))
                if lo<=hi:
                    gate=float(np.clip(gate,lo,hi))
            # The old stripe may disappear behind the next crate just as we
            # reach the measured crossing station. Switching the target only
            # stops and looks; the new stripe must authorize the first sidestep.
            if (has_next and self.aligned_box==i and self.station_bounds is not None and
                    frame.progress_x is not None and self.station_bounds[0]<=frame.progress_x<=self.station_bounds[1]
                    and abs(frame.progress_x-gate)<.040 and abs(vel[0])<.11):
                self.phase='shift'
                self.entry_station=None
                return self._hold('phase_observation')
        side=self._pass_side(frame)
        k=int(side>0)
        blind_patch=(frame.depth_status=='depth_unobserved' and
                     (frame.depth_diagnostics or {}).get('reason')=='unseen_swept_path')
        line_returned=(blind_patch and self.phase=='along' and self.aligned_box==i and
                       self.boundary_lock is not None and any(edge is not None for edge in frame.edges))
        # A sliver can become visible during a search turn, then disappear as the
        # turning gait settles. Finish the bounded turn before requesting zero.
        move=self.steps.active
        if (not line_returned and self.search_heading is None and move and move['axis']==2 and (frame.identified or frame.linked) and
                move['yaw'] is not None and abs(wrap(yaw-move['yaw']))<.16):
            self.hold_reason='look_for_opposite_line'
            return np.array([0.,0.,move['sign']*.80])
        index=self.index+(self.phase=='shift')
        gap=self.tuning.first_gap/100. if index==0 else .39
        # A first gap of 39 cm asks for a body centre 1 cm inside the opposite stripe.
        inset=.25 if self.phase=='lane_entry' else .40-gap
        key=(index,side,inset)
        edge=frame.edges[k]
        prior=frame.previous_edges[k]
        obscured=edge is None
        finishing_on_last_line=False
        line_correction=0.
        if obscured and self.phase in ('shift','lane_entry') and frame.registered_edges[k] and frame.previous_registered[k]:
            # This exact stripe and this same crate were measured together.
            # Every step still needs two current captures of the crate's side.
            edge=frame.registered_edges[k]
            prior=frame.previous_registered[k]
            obscured=False
        if obscured:
            # After an actual opposite-stripe alignment, the next crate can hide it.
            # The other measured boundary can maintain an existing forward pass, but
            # never initiate alignment or authorize a lateral step.
            if self.phase!='along' or self.boundary_lock!=key:
                return self._look_for_edge(frame,yaw,pos)
            if frame.edges[1-k] is not None:
                edge=frame.edges[1-k]
                prior=frame.previous_edges[1-k]
                line_correction=side*.50
            else:
                edge=frame.registered_edges[k]
                prior=frame.previous_registered[k]
                last=self.last_aligned_line
                finishing_on_last_line=(blind_patch and move is not None and move['axis']==0 and
                                        last is not None and last[0]==key and 0<frame.stamp-last[1]<=.45)
                if edge is None:
                    if not finishing_on_last_line:
                        return self._look_for_edge(frame,yaw,pos)
                    # Finish the already-started short step across a momentary
                    # dropout. No new grant can start from the remembered line.
                    edge=last[2]
                    prior=edge
                    line_correction=last[3]
        norm=np.hypot(1.,edge.slope)
        offset=edge.offset/norm+line_correction
        error=offset-side*inset
        if index==0 and self.phase!='lane_entry':
            self.panel_gap=(frame.stamp,float(100*(.40-side*offset)))
        tangent=np.array([1.,edge.slope])/norm
        normal=np.array([-tangent[1],tangent[0]])
        # A lateral grant cannot contain a forward component along the tape.
        transform=rot(frame.heading).T@rot(yaw)
        angle=wrap(frame.heading+np.arctan(edge.slope)-yaw)
        turn=float(np.clip(2.*angle,-.38,.38))
        self.error=[None,float(error)]
        self.target=None
        # Establish alignment with two captures. Once locked, a newly observed
        # aligned stripe can resume the same forward pass after a brief dropout.
        zero=self.steps.zero_stamp
        tolerance=.040 if self.boundary_lock==key else .025
        current_line=frame.edges[k] is not None or (obscured and frame.edges[1-k] is not None)
        following_line=(self.phase=='along' and self.boundary_lock==key and self.aligned_box==index
                        and abs(error)<=tolerance and current_line and blind_patch)
        if (prior is None or not 0<frame.stamp-frame.previous_stamp<=.55 or
                frame.previous_stamp<=self.invalid_projection_time or
                (zero is not None and frame.previous_stamp<=zero)):
            if not (following_line or finishing_on_last_line):
                return self._hold('confirm_opposite_line')
            prior=edge
        prior_error=prior.offset/np.hypot(1.,prior.slope)+line_correction-side*inset
        if abs(prior_error-error)>.075 or abs(prior.slope-edge.slope)>.22:
            return self._hold('confirm_opposite_line')
        paint=frame.stamp if frame.edges[k] is not None else frame.paint_stamp[k]
        saw_target=0<=frame.stamp-paint<=.50 and (zero is None or paint>zero)
        if abs(error)>tolerance:
            if obscured:
                return self._look_for_edge(frame,yaw,pos)
            self.boundary_lock=None
            if abs(prior_error)<=tolerance or error*prior_error<0:
                return self._hold('confirm_opposite_line')
            if not saw_target:
                return self._look_for_edge(frame,yaw,pos)
            self.hold_reason='lateral_step'
            # Hold our measured fore/aft station during the lateral gait. If
            # the previous entrance step barely moved sideways, re-anchor at
            # the current station before pushing sideways again.
            if self.entry_station is None or (self.phase=='enter' and self.steps.active is None and
                    self.steps.little_response(1,np.sign(error),side*.25-offset)):
                self.entry_station=float(pos[0])
            axial=float(np.clip(2.2*(self.entry_station-pos[0])-.55*vel[0]-.055,-.19,0.))
            speed=min(self.tuning.side_speed/100.,max(.18,2.3*abs(error)))
            lateral=np.sign(error)*speed
            if self.phase=='shift':
                back,front=frame.centers[self.index:self.index+2]
                body=self.body_sample
                if body is None or t-body[0]>.22 or frame.progress_x is None:return self._hold('wait_station_geometry')
                before=back[1]-side*.39
                after=front[1]+side*.39
                ratio=np.clip((pos[1]-before)/(after-before),0.,1.)
                lo,hi=station_window(body[1],rot(frame.heading),pos,after,back,front)
                self.station_bounds=[float(np.clip(lo,-10,10)),float(np.clip(hi,-10,10))]
                station=(back[0]+front[0])/2-.043+.079*np.clip((ratio-.36)/.52,0.,1.)
                if lo<=hi:station=float(np.clip(station,lo,hi))
                # Reserve 18 mm for settling as the leading wrist sweeps past
                # the next box. Fade the reserve before the trailing foot exits.
                margin=.035 if self.entry_clearance is None else self.entry_clearance
                tight=float(np.clip((.028-margin)/.008,0.,1.))
                reserve=float(np.clip((.063-margin)/.012,0.,1.))*(1.-tight)
                station-=reserve*.018*np.clip((ratio-.37)/.18,0.,1.)*np.clip((1.-ratio)/.10,0.,1.)
                lateral=np.clip(2.3*(error-.33*vel[1]),-self.tuning.side_speed/100.,self.tuning.side_speed/100.)
                compensate=-0.012*np.clip((ratio-0.34)/0.18,0.,1.)*np.clip((1.-ratio)/.09,0.,1.)
                axial=float(np.clip(3*(station-pos[0]-.39*vel[0])-0.115*min(1.,abs(lateral)/.15)-compensate,-.30,.06))
                if tight>0.:
                    # When backing up further would crowd the trailing heel, turn the
                    # leading wrist away from the box while translating along the tape.
                    blend=np.clip((ratio-.30)/.17,0.,1.)*np.clip((.84-ratio)/.14,0.,1.)
                    bias=-.20*blend*tight if side>0 else 0.
                    turn=float(np.clip((2.+1.2*tight)*wrap(angle+bias),-.38-.14*tight,.38+.14*tight))
            return np.r_[(lateral*normal+axial*tangent)@transform,turn]
        if abs(prior_error)>tolerance:
            return self._hold('confirm_opposite_line')
        if self.phase in ('enter','shift','lane_entry') and not saw_target:
            return self._look_for_edge(frame,yaw,pos)
        self.boundary_lock=key
        self.aligned_box=index
        if current_line:
            self.last_aligned_line=(key,frame.stamp,edge,line_correction)
        self.entry_station=None
        if self.phase=='lane_entry':
            gate=frame.centers[self.index][0]+.35
            if frame.progress_x is not None and frame.progress_x>frame.centers[self.index][0]+.30:
                self.release_requested=True
                return self._hold('lane_exit_confirmation')
        if self.phase=='shift':
            self.index+=1
            self.phase='along'
            return self._hold('phase_observation')
        if self.phase=='enter':
            self.phase='along'
            return self._hold('phase_observation')
        x=frame.progress_x
        if x is None:
            return self._hold('wait_range')
        dx=gate-x
        self.error=[float(dx),float(error)]
        self.target=[float(gate),float(pos[1]+error)]
        if dx<.040:
            if abs(vel[0])>=.11:
                return self._hold('phase_observation')
            if has_next:
                self.phase='shift'
            elif i==2:
                self.phase='lane_entry'
            else:
                return self._hold('wait_next_box')
            return self._hold('phase_observation')
        partial=frame.depth_status=='depth_limited' and frame.forward_room>=.035
        follow_blind_patch=blind_patch and (current_line or finishing_on_last_line)
        if not follow_blind_patch and (frame.depth_status not in ('depth_clear','depth_limited') or
                (frame.forward_room+1.e-6<self.tuning.forward_step/100. and not partial)):
            return self._hold(frame.depth_status)
        # On a lateral-to-forward switch at the edge of the alignment
        # tolerance, confirm with one more capture. An unchanged position on
        # that capture is valid: place no movement requirement on this check.
        previous=self.steps.previous
        if previous and previous['axis']==1 and abs(error)>.025:
            first_check=previous.setdefault('forward_check_stamp',frame.stamp)
            if frame.stamp<=first_check:
                return self._hold('confirm_opposite_line')
        if frame.depth_diagnostics is not None and not follow_blind_patch:
            live=self.body_sample
            if live is None or not 0<=t-live[0]<=.13 or frame.stamp<self.last_body_gap:
                return self._hold('stale_live_body')
        # A zero-command gait coasts. Cap displacement on the lower-latency
        # leg sample while the fresh capture still owns the target station.
        rate=float(np.clip(frame.velocity[0],0.,.27))
        allowance=gate-pos[0]-.40*rate-.014
        if dx<.105 and allowance<=.022:
            return self._hold('brake_at_station')
        self.step_cap=max(.022,dx-.032) if dx<.10 else None
        self.hold_reason='line_follow_unseen_path' if follow_blind_patch else 'confirmed_path'
        speed=min(self.tuning.forward_speed/100.,max(.22,1.65*dx))
        if follow_blind_patch and dx<.45:
            self.step_cap=min(self.step_cap or self.tuning.forward_step/100.,.030)
            speed=min(speed,.18)
        return np.r_[(speed*tangent)@transform,turn]

    def diagnostic(self):
        depth=(self.frame.depth_diagnostics if self.frame is not None and
               self.frame.stamp>self.invalid_projection_time else None)
        return dict(crate_phase=self.phase,crate_index=self.index,crate_side=float(self.side),
                    crate_gate=self.hold_reason,crate_steps=self.step_count,crate_entry_clearance=self.entry_clearance,
                    crate_step_retry=bool(self.steps.active and self.steps.active['retry']),
                    crate_paint=[v if np.isfinite(v) else None for v in self.paint_time],crate_boundary_lock=self.boundary_lock,
                    crate_boundary_offsets=None if self.frame is None else [None if e is None else e.offset for e in self.frame.edges],
                    crate_forward_depth=None if self.frame is None else self.frame.depth_status,
                    crate_depth_detail=(depth or {}).get('reason'),crate_depth_diagnostics=depth,
                    crate_forward_room=None if self.frame is None else self.frame.forward_room,
                    crate_progress_x=None if self.frame is None else self.frame.progress_x,
                    crate_evidence=None if self.frame is None else [self.frame.identified,self.frame.axis_evidence,self.frame.tape_fix,self.frame.tape_longitudinal],
                    crate_visual_xy=None if self.frame is None else self.frame.visual_xy,
                    crate_station_bounds=self.station_bounds,crate_zero_stamp=self.steps.zero_stamp,
                    crate_step_until=None if self.steps.active is None else self.steps.active['until'],
                    crate_reference=None if self.frame is None else [self.frame.direct_axes,self.frame.linked],
                    crate_pose=self.pos.tolist(),crate_velocity=self.velocity.tolist(),crate_heading=self.heading,
                    crate_centers=[v.tolist() for v in self.boxes],crate_visible=self.visible,
                    crate_residuals=self.corrections,crate_target=self.target,crate_error=self.error,crate_done=self.done)
