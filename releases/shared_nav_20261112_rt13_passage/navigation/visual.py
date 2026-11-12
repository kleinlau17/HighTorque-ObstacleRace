"""Camera-only local lane/obstacle perception and receding-horizon navigation.

No simulator, field configuration, saved route, or global map is accessible here.
Input: RGB-D, calibrated intrinsics/extrinsics, and estimated onboard odometry.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import cv2
import numpy as np
from navigation.surface_odometry import SurfaceOdometry
from navigation.gap_controller import GapController
from navigation.ground import GroundEstimator
from navigation.floor_odometry import FloorOdometry


def rotation2(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])


@dataclass
class Perception:
    time: float
    lane: np.ndarray | None
    white: np.ndarray
    obstacles: np.ndarray
    finish: np.ndarray
    ground_height: float
    overlay: np.ndarray
    current_white: np.ndarray | None = None
    lane_support: tuple[float, float] | None = None
    lane_observed: bool = True
    steering_heading: float | None = None
    # Local RT13 patch: white tape points measured closer to the body (x>0.18)
    # for box-zone tape following. Lane fitting keeps using current_white.
    near_white: object = None


class RGBDOdometry:
    """Feature-flow translation with IMU rotation; retain pose on lost tracking."""
    def __init__(self,width,height,fovy,camera_position,camera_rotation,intrinsics=None):
        f=height/2/np.tan(np.deg2rad(fovy/2))
        self.fx,self.fy,self.cx,self.cy=intrinsics if intrinsics is not None else (f,f,(width-1)/2,(height-1)/2)
        self.translation,self.rotation=camera_position,camera_rotation
        self.position32=np.asarray(camera_position,dtype=np.float32)
        self.rotation32=np.asarray(camera_rotation,dtype=np.float32)
        self.previous=None;self.previous_time=None;self.inliers=0;self.accepted=False;self.map_reset=False
        u,v=np.meshgrid(np.arange(width),np.arange(height))
        self.forward_ray=(u-self.cx)/self.fx*self.rotation[0,0]+(v-self.cy)/self.fy*self.rotation[0,1]+self.rotation[0,2]
        rays=np.stack([(u-self.cx)/self.fx,(v-self.cy)/self.fy,np.ones_like(u)],-1).astype(np.float32)
        self.geometry=SurfaceOdometry(rays,camera_position,camera_rotation)
        self.floor=FloorOdometry((self.fx,self.fy,self.cx,self.cy),camera_position,camera_rotation)
        self.previous_ground=None
        self.previous_geometry=None
        self.surface_reference=None
        self.recent_surface_reference=None
        self.candidate_surface_reference=None
        self.method='initializing'
        self.relocalized=False

    def points(self,uv,depth,attitude):
        pixels=np.rint(uv).astype(int)
        pixels[:,0]=np.clip(pixels[:,0],0,depth.shape[1]-1);pixels[:,1]=np.clip(pixels[:,1],0,depth.shape[0]-1)
        z=depth[pixels[:,1],pixels[:,0]]
        pc=np.column_stack([(uv[:,0]-self.cx)/self.fx*z,(uv[:,1]-self.cy)/self.fy*z,z]).astype(np.float32)
        return (pc@self.rotation32.T+self.position32)@np.asarray(attitude,dtype=np.float32).T

    @staticmethod
    def feature_mask(gray, depth):
        # Low-contrast carpet grain and rasterized straight edges can pass
        # forward/backward flow while their along-edge motion is ambiguous.
        quality=cv2.cornerMinEigenVal(gray,3)
        smooth=cv2.GaussianBlur(gray.astype(np.float32)/255.,(0,0),1.5)
        structure=cv2.cornerEigenValsAndVecs(smooth,9,3)[...,:2]
        directional=structure.min(axis=2)>.05*structure.max(axis=2)
        finite=np.isfinite(depth)
        filled=np.where(finite,depth,0).astype(np.float32)
        kernel=np.ones((3,3),np.uint8)
        lower=cv2.erode(filled,kernel)
        upper=cv2.dilate(filled,kernel)
        continuous=(lower>0)&(upper-lower<np.maximum(.025,.03*filled))
        return (quality>=1e-4)&directional&continuous

    def update(self,frame,attitude,kinematic_xy):
        gray=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
        result=kinematic_xy.copy();self.inliers=0;self.accepted=False;self.map_reset=False
        self.relocalized=False
        self.method='unobserved'
        geometry=self.geometry.project(frame,attitude)
        self.floor.select_window(frame.depth, attitude, self.geometry.ground.result)
        candidate=self.candidate_surface_reference
        self.candidate_surface_reference=None
        if self.previous is None:
            self.surface_reference=None
            self.recent_surface_reference=None
            candidate=None
        if self.previous is not None:
            old,depth,rotation,xy=self.previous
            references=[item for item in (self.surface_reference,self.recent_surface_reference,candidate) if item is not None]
            for reference_record in references:
                reference,origin,stamp=reference_record
                displacement=self.geometry.register(reference,geometry,initial=kinematic_xy-origin)
                innovation=None if displacement is None else origin+displacement-kinematic_xy
                if displacement is not None and np.linalg.norm(displacement)<min(.25,.8*(frame.time-stamp)+.01) and np.linalg.norm(innovation)<.09:
                    result=origin+displacement;self.accepted=True
                    self.inliers=self.geometry.support;self.method='rgbd_surfaces'
                    self.relocalized=bool(np.linalg.norm(innovation)>min(.12,.45*(frame.time-self.previous_time)+.012))
                    if reference_record is candidate:
                        self.surface_reference=candidate
                    break
            if not self.accepted:
                displacement=self.floor.estimate(old,depth,rotation,self.previous_ground,
                    gray,frame.depth,attitude,self.geometry.ground.result,frame.time-self.previous_time)
                if displacement is not None:
                    result=xy+displacement
                    self.accepted=True
                    self.inliers=self.floor.support
                    self.method='rgbd_floor'
                    self.relocalized=bool(frame.time-self.previous_time>.25)
            # Self geometry is not a stationary visual landmark.
            mask=(self.feature_mask(old,depth)&(depth<3)&(depth*self.forward_ray+self.translation[0]>.25)).astype(np.uint8)*255
            points=None if self.accepted else cv2.goodFeaturesToTrack(old,400,.001,3,mask=mask)
            if points is not None and len(points)>=3:
                current,ok,error=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(21,21),maxLevel=3)
                if current is None or ok is None:return result
                back,bok,_=cv2.calcOpticalFlowPyrLK(gray,old,current,None,winSize=(21,21),maxLevel=3)
                if back is None or bok is None:return result
                keep=ok.ravel().astype(bool)&bok.ravel().astype(bool)&(error.ravel()<20)&(np.linalg.norm((back-points)[:,0],axis=1)<.6)
                current_pixels=np.rint(current[:,0]).astype(int)
                inside=(current_pixels[:,0]>=0)&(current_pixels[:,0]<gray.shape[1])&(current_pixels[:,1]>=0)&(current_pixels[:,1]<gray.shape[0])
                keep&=inside
                valid_current=self.feature_mask(gray,frame.depth)
                selected=np.flatnonzero(inside)
                keep[selected]&=valid_current[current_pixels[selected,1],current_pixels[selected,0]]
                a=self.points(points[:,0],depth,rotation);b=self.points(current[:,0],frame.depth,attitude)
                delta=a-b
                keep&=np.isfinite(delta).all(axis=1)
                candidates=delta[keep]
                if len(candidates)>=3:
                    median=np.median(candidates,axis=0)
                    good=np.linalg.norm(candidates-median,axis=1)<.012
                    self.inliers=int(good.sum())
                    # With attitude measured, only translation is estimated.
                    # Three widely distributed metric correspondences suffice;
                    # sparse support must have stronger consensus and span two
                    # image directions, rather than one ambiguous edge.
                    uv=points[keep,0][good]
                    distributed=len(uv)>=3 and np.linalg.eigvalsh(np.cov(uv.T))[0]>=25.
                    enough=(self.inliers>=6 and self.inliers>=len(candidates)*.45) or (self.inliers>=3 and self.inliers>=len(candidates)*.7 and distributed)
                    if enough:
                        displacement=np.median(candidates[good],axis=0)[:2]
                        if np.linalg.norm(displacement)<min(.12,.45*(frame.time-self.previous_time)+.012):
                            # Metric image correspondences are the only source
                            # of translation; no commanded-velocity integration.
                            result=xy+displacement
                            self.accepted=True
                            self.method='rgbd_features'
        # Retain the last accepted image through short failures. Replacing it
        # with an untracked image would permanently discard motion in that
        # interval, compressing the estimated separation between static boxes.
        initializing=self.previous is None and np.count_nonzero(np.isfinite(frame.depth))>=100
        if self.accepted or initializing:
            self.previous=(gray.copy(),frame.depth.copy(),attitude.copy(),result.copy())
            self.previous_time=float(frame.time)
            self.previous_geometry=geometry
            self.previous_ground=self.geometry.ground.result
            # Keep a metric keyframe over a short spatial baseline. Replacing
            # it at every noisy image accumulates millimetre-scale edge bias
            # even while the robot is stationary.
            if initializing or self.method=='rgbd_surfaces':
                self.recent_surface_reference=(geometry,result.copy(),float(frame.time))
                renew=self.surface_reference is None or np.linalg.norm(result-self.surface_reference[1])>.10
                if renew and self.geometry.observable(geometry):
                    self.surface_reference=(geometry,result.copy(),float(frame.time))
            elif self.method=='rgbd_floor' and self.geometry.observable(geometry):
                # A measured floor segment can outgrow both old references.
                # Hold its newly visible geometry for one more image before
                # admitting it as a registered metric anchor. Existing anchors
                # retain priority, and sparse feature flow cannot seed one.
                self.candidate_surface_reference=(geometry,result.copy(),float(frame.time))
        return result


class CameraPerception:
    def __init__(self, width, height, fovy, camera_position, camera_rotation, intrinsics=None):
        f = height/2/np.tan(np.deg2rad(fovy/2))
        fx,fy,cx,cy=intrinsics if intrinsics is not None else (f,f,(width-1)/2,(height-1)/2)
        u,v = np.meshgrid(np.arange(width), np.arange(height))
        self.rays = np.stack([(u-cx)/fx,(v-cy)/fy,np.ones_like(u)],-1).astype(np.float32)
        self.camera_position, self.camera_rotation = camera_position, camera_rotation
        self.position32 = np.asarray(camera_position, dtype=np.float32)
        self.rotation32 = np.asarray(camera_rotation, dtype=np.float32)
        self.white_memory = []
        self.obstacle_memory = []
        self.rng = np.random.default_rng(4)
        self.box_rng = np.random.default_rng(17)
        self.lane_basis = 'no_fit'
        self.start_edge = None
        self.intrinsics = fx, fy, cx, cy
        self.raised_sites = None
        self.floor_pixels = None
        self.clearance_depth = None
        self.previous_lane = None
        self.last_pose = None
        self.ground = GroundEstimator()
        self.map_translation = np.zeros(2)

    def box_surfaces(self, cloud, heights, camera_xy):
        """Recover a square footprint from vertical faces, including clipped ones.

        A face constrains the center only along its normal. Its visible tangent
        interval bounds the center but must not drag an occluded track toward
        the midpoint of that interval. No inter-box spacing is assumed.
        """
        walls=cloud[heights<np.percentile(heights,99)-.015]
        if len(walls)<30:return None
        sample=walls[np.linspace(0,len(walls)-1,min(len(walls),700)).astype(int)]
        best=None;score=0
        for _ in range(70):
            a,b=sample[self.box_rng.choice(len(sample),2,replace=False)]
            tangent=b-a;length=np.linalg.norm(tangent)
            if length<.04:continue
            normal=np.array([-tangent[1],tangent[0]])/length
            mask=abs((sample-a)@normal)<.003
            if mask.sum()>score:best,score=normal,mask.sum()
        if best is None or score<max(25,len(sample)*.2):return None
        # Refine the dominant face, excluding its recessed seams and other faces.
        projections=sample@best
        # Use a dense interval to recover the offset belonging to the RANSAC line.
        ordered=np.sort(projections);left=np.searchsorted(ordered,ordered-.006)
        i=np.argmax(np.arange(len(ordered))-left)
        offset=np.median(ordered[left[i]:i+1])
        inliers=walls[abs(walls@best-offset)<.004]
        if len(inliers)<25:return None
        mean=inliers.mean(axis=0);_,_,vh=np.linalg.svd(inliers-mean,full_matrices=False)
        normal=vh[-1]
        if (mean-camera_xy)@normal<0:normal=-normal
        tangent=np.array([-normal[1],normal[0]])
        if np.ptp(inliers@tangent)<.05:return None
        near=float(np.median(inliers@normal))
        normals=[normal];offsets=[near+.10]
        # A second perpendicular face makes both center coordinates observable.
        side=walls[(walls@normal>near+.018)&(walls@normal<near+.22)]
        if len(side)>=25 and np.ptp(side@normal)>.05:
            coordinate=side@tangent
            ordered=np.sort(coordinate);left=np.searchsorted(ordered,ordered-.006)
            i=np.argmax(np.arange(len(ordered))-left)
            side_offset=float(np.median(ordered[left[i]:i+1]))
            support=side[abs(coordinate-side_offset)<.004]
            if len(support)>=25 and len(support)>.35*len(side) and np.ptp(support@normal)>.05:
                direction=1 if (support.mean(axis=0)-camera_xy)@tangent>0 else -1
                normals.append(tangent*direction);offsets.append(side_offset*direction+.10)
        normals=np.asarray(normals);offsets=np.asarray(offsets)
        lo,hi=np.percentile(cloud@tangent,[1,99])
        # All observed points must lie inside the known 20 cm footprint.
        lower,upper=hi-.102,lo+.102
        if lower>upper:return None
        center=normal*offsets[0]+tangent*((lo+hi)/2)
        if len(normals)==2:center=np.linalg.solve(normals,offsets)
        size=np.maximum(.20,.20*(abs(normal)+abs(tangent)))
        return center,size,normals,offsets,tangent,lower,upper

    @staticmethod
    def box_size(tracked, rotation):
        """Project the observed axes of a stationary 20 cm square into heading."""
        axes = tracked['axes'] @ rotation
        return .20 * np.abs(axes).sum(axis=0)

    @staticmethod
    def known_top_surfaces(outline, cloud, tracked, predicted, rotation):
        """Use visible top-to-floor edges to refine an already tracked box."""
        if len(outline) < 8:
            return None
        axes = tracked.get('axes', np.eye(2)) @ rotation
        normals, offsets = [], []
        for axis in axes:
            tangent = np.array([-axis[1], axis[0]])
            candidates = []
            for side in (-1, 1):
                expected = predicted @ axis + side * .10
                selected = outline[abs(outline @ axis - expected) < .035]
                if len(selected) < 8 or np.ptp(selected @ tangent) < .04:
                    continue
                edge = float(np.median(selected @ axis))
                if np.percentile(abs(selected @ axis - edge), 90) > .006:
                    continue
                # The inward normal locates the center 10 cm from this edge.
                candidates.append((len(selected), -side * axis, -side * edge + .10))
            if candidates:
                _, normal, offset = max(candidates, key=lambda value: value[0])
                normals.append(normal)
                offsets.append(offset)
        if not normals:
            return None
        normals, offsets = np.asarray(normals), np.asarray(offsets)
        center = predicted + normals.T @ (offsets - normals @ predicted)
        tangent = np.array([-normals[0, 1], normals[0, 0]])
        lower, upper = np.percentile(cloud @ tangent, [99, 1]) + [-.102, .102]
        if lower > upper:
            return None
        size = CameraPerception.box_size(tracked, rotation)
        return center, size, normals, offsets, tangent, lower, upper

    def _lane_candidates(self, points):
        """Score the fixed 43 straight and 29 curved RANSAC candidates.

        The straight trials dominate the loop, so they are drawn in the original
        order but solved and scored as one matrix: a two-point fit is just the
        line through those two points, and the greedy selection only keeps the
        first strictly better score, which argmax reproduces. Quadratic trials
        stay scalar because there are few and they are cheap.
        """
        x, y = points[:, 0], points[:, 1]
        count = len(points)
        pairs = np.empty((43, 2), np.int64)
        for index in range(43):
            pairs[index] = self.rng.choice(count, 2, replace=False)
        curves = [points[self.rng.choice(count, 3, replace=False)] for _ in range(29)]
        x1, y1 = points[pairs[:, 0], 0], points[pairs[:, 0], 1]
        x2, y2 = points[pairs[:, 1], 0], points[pairs[:, 1], 1]
        dx = x2-x1
        usable = abs(dx) >= .18
        slope = np.divide(y2-y1, dx, out=np.zeros_like(dx), where=usable)
        usable &= abs(slope) <= 1.5
        intercept = y1-slope*x1
        masks = abs(slope[:, None]*x[None, :]+intercept[:, None]-y[None, :]) < .018
        scores = np.where(usable, masks.sum(axis=1), -np.inf)
        winner = int(np.argmax(scores))
        bestscore = float(scores[winner])
        best = masks[winner] if bestscore > 0 else None
        bestdegree = 1
        for sample in curves:
            if np.ptp(sample[:, 0]) < .4: continue
            try: c = np.polyfit(sample[:, 0], sample[:, 1], 2)
            except np.linalg.LinAlgError: continue
            if abs(c[0]) > 1.5 or abs(c[1]) > 1.5: continue
            mask = abs((c[0]*x+c[1])*x+c[2]-y) < .018
            score = mask.sum()-abs(c[0])*5
            if score > bestscore: best, bestscore, bestdegree = mask, score, 2
        return best, bestscore, bestdegree

    @staticmethod
    def _edge_sections(points, line, span):
        """Take measured tape centres in 4 cm slices, starting at its near support."""
        x, y = points.T
        mask = (x>=span[0]-.03)&(x<=span[1]+.03)&(abs(y-np.polyval(line,x))<.045)
        tape = points[mask]
        if len(tape)<12:
            return None
        bins = np.floor((tape[:,0]-tape[:,0].min())/.04).astype(int)
        sections = []
        for i in np.unique(bins):
            group = tape[bins==i]
            if len(group)<2:
                continue
            point = np.median(group,axis=0)
            if sections and point[0]-sections[-1][0]>.105:
                break
            sections.append(point)
        if len(sections)<5:
            return None
        return np.asarray(sections)

    @staticmethod
    def _steering_heading(path):
        # Use only the measured near arc. Aim at its centre and follow its local
        # direction; the tape may have two different tangents on either side of a join.
        x,y = path.T
        if len(path)<5 or np.any(np.diff(x)<=0) or x[-1]-x[0]<.14:
            return None
        look = min(x[0]+.23,x[-1]-.05)
        lo,hi = max(x[0],look-.09),min(x[-1],look+.09)
        direction = np.arctan2(np.interp(hi,x,y)-np.interp(lo,x,y),hi-lo)
        bearing = np.arctan2(np.interp(look,x,y),look)
        return float(.65*direction+.35*bearing)

    def fit_lane(self, points, reference=None, require_known_side=False, inside_start=False):
        self.lane_basis = 'no_fit'
        self.steering_heading = None
        self.lane_support=None
        points = points[(points[:,0]>-.12)&(points[:,0]<1.4)&(abs(points[:,1])<1.4)]
        if len(points)<20:
            return None
        # Trial scoring is O(trials x points); a uniform cap keeps the same
        # evidence spread while bounding the per-frame cost on the robot.
        if len(points)>512:
            points=points[np.linspace(0,len(points)-1,512).astype(int)]
        lines=[]
        spans=[]
        remaining=points.copy()
        for side in range(2):
            if len(remaining)<15: break
            best,bestscore,bestdegree=self._lane_candidates(remaining)
            if best is None or bestscore<12: break
            line=np.r_[np.zeros(2-bestdegree),np.polyfit(remaining[best,0],remaining[best,1],bestdegree)]
            if np.ptp(remaining[best,0])<.18:break
            spans.append(np.percentile(remaining[best,0],[5,95]))
            lines.append(line);remaining=remaining[~best]
        if len(lines)==2:
            # A curved boundary can produce two RANSAC segments. They only
            # form a lane pair if both are observed over the same forward
            # interval and have lane-width separation within that interval.
            lower=max(0.,spans[0][0],spans[1][0])
            upper=min(spans[0][1],spans[1][1])
            if upper-lower>=.18:
                samples=np.linspace(lower,upper,7)
                slope=np.polyval(np.polyder(np.mean(lines,axis=0)),samples)
                separation=abs(np.polyval(lines[0],samples)-np.polyval(lines[1],samples))/np.sqrt(1+slope*slope)
                if np.all((separation>.36)&(separation<.8)):
                    self.lane_support=(float(lower),float(upper))
                    self.lane_basis = 'paired_edges'
                    return np.mean(lines,axis=0)
        if lines:
            # One boundary can be hidden by a box. Offset the visible boundary
            # by the known lane half-width, along its normal, not along image Y.
            line=lines[0];x=np.linspace(spans[0][0],spans[0][1],40)
            slope=np.polyval(np.polyder(line),x)
            # A bypass may put the body outside one boundary while one foot
            # remains inside. The boundary's sign relative to the body then
            # changes; its identity relative to the tracked lane does not.
            identity_x=np.clip(.4,spans[0][0],spans[0][1])
            reference_y=0. if reference is None else np.polyval(reference,identity_x)
            distance = (np.polyval(line,identity_x)-reference_y)/np.hypot(1.,np.polyval(np.polyder(line),identity_x))
            side=1 if distance>0 else -1
            sections = self._edge_sections(points,line,spans[0]) if require_known_side else None
            self.lane_basis = 'single_edge'
            if require_known_side and reference is not None:
                if abs(abs(distance)-.25)>.12:
                    self.lane_basis = 'edge_disagrees_with_history'
                    return None
                self.lane_basis = 'edge_with_history'
            elif require_known_side:
                # At an in-lane race start the near tape identifies its side.
                # The far end of a bend may cross the camera centre even though
                # the near part of this same marking stays on the right or left.
                self.lane_basis = 'edge_side_unknown'
                if not inside_start or sections is None or sections[0,0]>.70:
                    return None
                near = sections[sections[:,0]<=sections[0,0]+.20]
                if len(near)<4 or np.ptp(near[:,0])<.11:
                    return None
                tangent, intercept = np.polyfit(near[:,0],near[:,1],1)
                residual = near[:,1]-(tangent*near[:,0]+intercept)
                # Signed distance from the base to the measured near tangent.
                # An oblique boundary often crosses y=0 within the image.
                body_distance = intercept/np.hypot(1.,tangent)
                if (np.max(abs(residual))>.018 or not .055<abs(body_distance)<.515
                        or abs(tangent)>1.65):
                    return None
                side = 1 if body_distance>0 else -1
                self.lane_basis = 'initial_left_edge' if side>0 else 'initial_right_edge'
            if sections is not None:
                slopes = np.gradient(sections[:,1],sections[:,0])
                slopes = np.array([np.median(slopes[max(0,i-1):i+2]) for i in range(len(slopes))])
                normals = np.column_stack([-slopes,np.ones(len(slopes))])/np.hypot(1.,slopes)[:,None]
                self.steering_heading = self._steering_heading(sections-side*.25*normals)
            nx=-slope/np.sqrt(1+slope*slope);ny=1/np.sqrt(1+slope*slope)
            center_x=x-side*.25*nx
            self.lane_support=(float(center_x.min()),float(center_x.max()))
            return np.polyfit(center_x,np.polyval(line,x)-side*.25*ny,2)
        return None

    def previous_sites_clear(self, sites, attitude, lane):
        """Return positive clearance evidence separately for each old footprint.

        Horizontal coordinates are current body offsets in AHRS axes; the last
        component is the observed return's height above its measured floor.
        """
        clear = np.zeros(len(sites), dtype=bool)
        if self.floor_pixels is None or not len(sites) or lane is None:
            return clear
        xy = sites[:,:2]
        yaw = np.arctan2(attitude[1, 0], attitude[0, 0])
        x, y = (xy@rotation2(yaw)).T
        center = np.polyval(lane, np.clip(x, .30, 1.15))
        relevant = ((x>-.35) & (x<1.30) & (y>=np.minimum(0., center)-.40)
                    & (y<=np.maximum(0., center)+.40))
        clear[~relevant] = True
        if not np.any(relevant):
            return clear
        xy = xy[relevant]
        plane = self.ground.result
        ground_z = -(xy@plane.normal[:2]+plane.offset)/plane.normal[2]
        fx, fy, cx, cy = self.intrinsics
        height, width = self.floor_pixels.shape
        du = np.tile([-1, 0, 1], 3)
        dv = np.repeat([-1, 0, 1], 3)

        def evidence(above_floor, check_floor):
            z = ground_z+above_floor/plane.normal[2]
            optical = (np.column_stack((xy,z))@attitude-self.position32)@self.rotation32
            depth = optical[:,2]
            uv = optical[:,:2]/np.maximum(depth,.15)[:,None]*[fx,fy]+[cx,cy]
            u, v = np.rint(uv).astype(int).T
            inside = (depth>.15)&(u>=1)&(u<width-1)&(v>=1)&(v<height-1)
            u = np.clip(u,1,width-2)[:,None]+du
            v = np.clip(v,1,height-2)[:,None]+dv
            measured = self.clearance_depth[v,u]
            valid = np.isfinite(measured)&(measured>.15)&(measured<2.7)
            if check_floor:
                support = self.floor_pixels[v,u]
            else:
                # A real return farther down the same ray sees through the old
                # raised position. A missing return or an occluding one cannot.
                support = valid & (measured>depth[:,None]+.035)
            foreground = valid & (measured<depth[:,None]-.035)
            return inside & support[:,4] & (support.sum(axis=1)>=6) & ~foreground.any(axis=1)

        ground_seen = evidence(0., True)
        # If the old footprint is at the image boundary, independently verify
        # clearance through both the old surface and the low collision band.
        if not np.all(ground_seen):
            ground_seen |= evidence(sites[relevant,2], False) & evidence(.06, False)
        clear[relevant] = ground_seen
        return clear

    def inspect_lane(self, frame, attitude, reference=None, inside_start=False):
        """Inspect this image in the level body frame without reading or writing map coordinates."""
        rgb, depth = frame.rgb, frame.depth
        body = (self.rays*depth[..., None])@self.rotation32.T+self.position32
        level = body@np.asarray(attitude, dtype=np.float32).T
        yaw = np.arctan2(attitude[1, 0], attitude[0, 0])
        finite = np.isfinite(depth) & (depth>.15) & (depth<2.7)
        # Decimate the plane fit only. Keep thin white tape at sensor resolution.
        plane = self.ground.estimate(level[::2, ::2], (finite & (body[..., 0]>.30))[::2, ::2], timestamp=frame.time)
        empty = np.empty((0, 2))
        result = Perception(frame.time, None, empty, np.empty((0, 4)), empty,
                            -plane.z0, rgb, lane_observed=False)
        self.lane_basis = 'no_fit'
        self.pivot_clear = False
        self.pivot_wide_clear = False
        self.raised_sites = self.floor_pixels = None
        if not plane.valid:
            self.start_edge = None
            return result, 'unknown_depth'
        height = plane.height(level)
        xy = level[..., :2]@rotation2(yaw).astype(np.float32)
        x, y = xy[..., 0], xy[..., 1]
        floor = finite & (abs(height)<.025)
        near=floor & (x>.20) & (x<.72)
        sweep=finite & (x*x+y*y<.35**2)
        hazard=(height>.10) | (height<-.055)
        self.pivot_clear=bool(np.count_nonzero(near & (y>0.) & (y<.25))>=12
                              and np.count_nonzero(near & (y<0.) & (y>-.25))>=12
                              and np.count_nonzero(sweep & hazard)<4)
        self.pivot_wide_clear=self.pivot_clear and np.count_nonzero(finite & (x*x+y*y<.49**2) & hazard)<4
        self.floor_pixels = floor
        self.clearance_depth = depth
        bright, dark = rgb.max(axis=2), rgb.min(axis=2)
        white_near = floor & (dark>115) & (bright-dark<.15*bright) & (x>.18)
        white = white_near & (x>.30)
        points = xy[white]
        lane = self.fit_lane(points, reference, require_known_side=True, inside_start=inside_start)
        result.white = result.current_white = points
        result.near_white = xy[white_near]
        result.lane, result.lane_support = lane, self.lane_support
        result.lane_observed = lane is not None
        if lane is not None:
            result.steering_heading = self.steering_heading
            if result.steering_heading is None:
                samples = np.linspace(*self.lane_support,12)
                result.steering_heading = self._steering_heading(np.column_stack([samples,np.polyval(lane,samples)]))
        route = lane if lane is not None else reference
        center = np.polyval(route, x) if route is not None else np.zeros_like(x)
        # Cover both the robot and the lane on approach, including a box on either edge.
        in_path = (y>np.minimum(center, 0.)-.30) & (y<np.maximum(center, 0.)+.30)
        # A turn sweeps the wrist beyond the straight-ahead body width.
        in_path |= abs(y)<.33
        # Local RT13 patch: 15 cm minimum (was 10 cm) so a rippled mat or a
        # tape seam is not reported as a raised box.
        raised = finite & (height>.15) & (height<.65) & (x>.22) & (x<1.45) & in_path
        # Draw what was recognised so the operator preview shows it: tape
        # candidates in magenta, raised cells in orange.
        result.overlay = rgb.copy()
        result.overlay[white_near] = (255, 80, 255)
        result.overlay[raised] = (255, 140, 0)
        supported = cv2.erode(raised.astype(np.uint8), np.ones((2, 2), np.uint8))
        if cv2.countNonZero(supported):
            # Keep an actual return per 4 cm cell, in gravity-aligned AHRS axes.
            points = level[supported!=0,:2]
            _, indices = np.unique(np.floor(points/.04).astype(int), axis=0, return_index=True)
            self.raised_sites = np.column_stack((points[indices],height[supported!=0][indices]))
            self.start_edge = None
            return result, 'raised_object'
        # Require measured floor across the approach, not merely absence of returns.
        # The first band starts in the visible area of the pitched head camera.
        cells, _, _ = np.histogram2d(x[floor], (y-center)[floor],
            bins=(np.linspace(.50,1.00,8), np.linspace(-.24,.24,7)))
        scene = 'clear'
        if np.any(cells<3):
            # The far end of a sharp bend can lie outside the camera. The
            # directly observed space in front can authorize yaw only.
            if result.steering_heading is not None and abs(result.steering_heading)>=np.deg2rad(16):
                front, _, _ = np.histogram2d(x[floor],y[floor],
                    bins=(np.linspace(.50,.85,6),np.linspace(-.33,.33,9)))
                if np.all(front>=3):
                    scene = 'turn_clear'
                else:
                    scene = 'unknown_depth'
            else:
                scene = 'unknown_depth'
            if scene=='unknown_depth':
                self.start_edge = None
                return result, scene
        if self.lane_basis in ('initial_left_edge','initial_right_edge'):
            previous = self.start_edge
            support = self.lane_support
            self.start_edge = (frame.time, self.lane_basis, yaw, lane.copy(), support)
            confirmed = False
            if previous is not None:
                stamp, edge, last_yaw, last_lane, last_support = previous
                start = max(support[0], last_support[0])
                end = min(support[1], last_support[1], start+.30)
                heading_change = np.arctan2(np.sin(yaw-last_yaw),np.cos(yaw-last_yaw))
                if (0<frame.time-stamp<=.65 and edge==self.lane_basis
                        and abs(heading_change)<.08 and end-start>=.10):
                    sample = np.linspace(start,end,5)
                    confirmed = np.max(abs(np.polyval(lane,sample)-np.polyval(last_lane,sample)))<.055
            if not confirmed:
                self.lane_basis = 'confirming_'+self.lane_basis
                result.lane = None
                result.lane_observed = False
                return result, scene
        else:
            self.start_edge = None
        if lane is not None:
            result.finish = self.finish_stripe(points, lane)
        return result, scene

    @staticmethod
    def finish_stripe(points, lane):
        if lane is None or not len(points):
            return np.empty((0, 2))
        rel = points[:,1]-np.polyval(lane,points[:,0])
        inside = points[(abs(rel)<.20)&(points[:,0]>.15)&(points[:,0]<1.5)]
        stripes = []
        for x in np.arange(.15, 1.5, .025):
            stripe = inside[abs(inside[:,0]-x)<.018]
            if len(stripe)>15 and np.ptp(stripe[:,1])>.32:
                stripes.append(np.median(stripe,axis=0))
        return np.asarray(stripes).reshape(-1,2)

    def update(self, frame, pose, attitude, ground_plane=None):
        self.map_translation = np.zeros(2)
        rgb, depth = frame.rgb, frame.depth
        body = (self.rays*depth[...,None])@self.rotation32.T+self.position32
        # Remove pitch/roll using measured attitude, retaining a heading-local frame.
        level = body@np.asarray(attitude,dtype=np.float32).T
        finite=np.isfinite(depth)&(depth>.15)&(depth<2.7)
        plane=ground_plane if ground_plane is not None else self.ground.estimate(level,finite&(body[...,0]>.30),timestamp=frame.time)
        if not plane.valid:
            return Perception(frame.time,None,np.empty((0,2)),np.empty((0,4)),np.empty((0,2)),float('nan'),rgb.copy(),lane_observed=False)
        ground=plane.z0
        height=plane.height(level)
        level[...,:2] = level[...,:2]@rotation2(pose[2])
        floor=finite&(abs(height)<.025)
        white=floor&(rgb.min(axis=2)>115)&((rgb.max(axis=2).astype(float)-rgb.min(axis=2))<.15*rgb.max(axis=2))
        pts=level[white,:2]
        keep=(pts[:,0]>.30)&(pts[:,0]<2.5)&(abs(pts[:,1])<1.8)
        pts=pts[keep]
        r=rotation2(pose[2])
        for tracked in self.obstacle_memory:
            tracked['size']=self.box_size(tracked,r)
        camera_xy=(self.camera_position@attitude.T)[:2]@r
        if len(pts):self.white_memory.append((frame.time,pts@r.T+pose[:2],pose[:2].copy()))
        self.white_memory=[(t,p,origin) for t,p,origin in self.white_memory
                           if frame.time-t<60 and np.linalg.norm(origin-pose[:2])<1.4]
        memory=np.vstack([p for _,p,_ in self.white_memory]) if self.white_memory else np.empty((0,2))
        # Bound the long-lived boundary point set: it grows with run time and
        # every later step (dedup, obstacle carve-out, lane fallback) is linear
        # in its size. A uniform cap keeps the same spatial spread.
        if len(memory)>20000:memory=memory[np.linspace(0,len(memory)-1,20000).astype(int)]
        local=(memory-pose[:2])@r
        recent=[p for t,p,_ in self.white_memory if frame.time-t<2.5]
        fresh=(np.vstack(recent)-pose[:2])@r if recent else np.empty((0,2))
        if len(local):
            quantized=np.round(local/.012).astype(np.int64)
            _,ids=np.unique((quantized[:,0]<<32)+quantized[:,1],return_index=True)
            local=local[ids]
        if len(fresh):
            quantized=np.round(fresh/.012).astype(np.int64)
            _,ids=np.unique((quantized[:,0]<<32)+quantized[:,1],return_index=True)
            fresh=fresh[ids]
        # Raised surfaces, independent of obstacle color. The robot is excluded
        # by the forward range gate; no segmentation or object IDs are read.
        raised=finite&(height>.06)&(height<.65)&(level[...,0]>.22)&(level[...,0]<2.7)&(abs(level[...,1])<1.2)
        op=level[raised,:2]
        oh=height[raised]
        adjacent_floor=cv2.dilate(floor.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        top_outline=(adjacent_floor&(height>.34)&(height<.46))[raised]
        def match_box(raw,lo,hi):
            candidates=[]
            for tracked in self.obstacle_memory:
                predicted=(tracked['center']-pose[:2])@r
                distance=np.linalg.norm(predicted-raw)
                contains=np.all(lo>=predicted-tracked['size']/2-.025)&np.all(hi<=predicted+tracked['size']/2+.025)
                if distance<.17 or contains:
                    candidates.append((min(distance,np.linalg.norm(predicted-(lo+hi)/2)) if contains else distance,tracked))
            return min(candidates,key=lambda item:item[0])[1] if candidates else None
        boxes=[]
        if len(op):
            cell=.025;grid=np.zeros((220,220),np.uint8)
            cells=np.floor((op+2.75)/cell).astype(int)
            grid[cells[:,1],cells[:,0]]=1
            grid=cv2.dilate(grid,np.ones((3,3),np.uint8))
            _,labels=cv2.connectedComponents(grid)
            ids=labels[cells[:,1],cells[:,0]]
            for label in np.unique(ids):
                cloud=op[ids==label]
                if len(cloud)<25:continue
                full_height=np.ptp(oh[ids==label])>=.20
                lo,hi=np.percentile(cloud,[1,99],axis=0)
                if np.any(hi-lo>.6) or np.max(hi-lo)<.08:continue
                # The 20 cm cube size is a task prior, not its hidden position.
                size=np.maximum(hi-lo,.20)
                center=(lo+hi)/2
                known=match_box(center,lo,hi)
                if not full_height:
                    if known is None:continue
                    predicted=(known['center']-pose[:2])@r
                    if (np.any(lo<predicted-known['size']/2-.035) or
                            np.any(hi>predicted+known['size']/2+.035)):
                        continue
                tops=cloud[oh[ids==label]>np.percentile(oh[ids==label],99)-.008]
                complete_top=False
                axes=None
                if len(tops)>=12:
                    rect=cv2.minAreaRect(tops.astype(np.float32))
                    sides=np.asarray(rect[1])
                    complete_top=bool(np.all(sides>.178)&np.all(sides<.23))
                    if complete_top:
                        center=np.asarray(rect[0]);corners=cv2.boxPoints(rect)
                        size=np.maximum(np.ptp(corners,axis=0),.20)
                        edge=corners[1]-corners[0]
                        normal=edge/np.linalg.norm(edge)
                        axes=np.array([normal,[-normal[1],normal[0]]])
                # Close boxes can be clipped by the camera's field of view.
                # A visible near face is a surface, not the cube's center.
                surfaces=None if complete_top else self.box_surfaces(cloud,oh[ids==label],camera_xy)
                if not full_height and not complete_top:
                    if np.median(oh[ids==label])>.34:
                        outline=cloud[top_outline[ids==label]]
                        surfaces=self.known_top_surfaces(outline,cloud,known,predicted,r)
                    if surfaces is None:continue
                if surfaces is not None:
                    center,surface_size,*_=surfaces
                    size=np.maximum(size,surface_size)
                    normal=surfaces[2][0]
                    axes=np.array([normal,[-normal[1],normal[0]]])
                if axes is None:continue
                boxes.append((center,size,lo,hi,complete_top,surfaces,axes))
        # Slatted boxes can expose an inner/back face through a gap. Associate
        # measured surfaces inside an existing footprint before interpreting
        # them as a new box center. Process near surfaces first when bootstrapping.
        boxes.sort(key=lambda b:np.linalg.norm((b[2]+b[3])/2-camera_xy))
        def face_offsets(surfaces,predicted,tracked):
            normals,offsets=surfaces[2:4]
            # A fragment inside a tracked footprint can expose diagonal slats.
            # Only its established outer face directions constrain this box.
            axes=tracked['axes']@r
            aligned=np.max(abs(normals@axes.T),axis=1)>np.cos(np.deg2rad(10))
            # An inner far face can be inset by unknown wall thickness. It is
            # evidence of this same box, but cannot locate its outer center.
            alternate=offsets-.20
            outer=abs(offsets-normals@predicted)<=abs(alternate-normals@predicted)
            return normals[outer&aligned],offsets[outer&aligned]
        # All boxes are stationary. A residual on an already tracked face also
        # measures common odometry drift. Move the whole local obstacle map in
        # observable directions, including temporarily hidden boxes, before
        # refining individual detections. Otherwise only the visible box moves
        # and a real gap gradually collapses in the map.
        # A matched surface may recover after an odometry jump. The common
        # map and individual box updates use the same capture distance.
        max_correction=.08
        constraints=[];residuals=[]
        for raw,size,lo,hi,complete_top,surfaces,axes in boxes:
            found=match_box(raw,lo,hi)
            if found is None:continue
            predicted=(found['center']-pose[:2])@r
            if complete_top:normals,offsets=np.eye(2),raw
            elif surfaces is not None:normals,offsets=face_offsets(surfaces,predicted,found)
            else:continue
            if not len(normals):continue
            error=offsets-normals@predicted
            if np.max(abs(error))<max_correction:
                constraints.extend(normals);residuals.extend(error)
        if constraints:
            a=np.asarray(constraints);b=np.asarray(residuals)
            u,s,vh=np.linalg.svd(a,full_matrices=False)
            observable=s>.25*s[0]
            correction=vh[observable].T@((u[:,observable].T@b)/s[observable])
            if np.linalg.norm(correction)<max_correction:
                self.map_translation=correction@r.T
                for tracked in self.obstacle_memory:tracked['center']+=self.map_translation
        for raw,size,lo,hi,complete_top,surfaces,axes in boxes:
            center=raw@r.T+pose[:2]
            found=match_box(raw,lo,hi)
            if found is not None:
                predicted=(found['center']-pose[:2])@r
                if surfaces is not None:
                    _,_,normals,offsets,tangent,lower,upper=surfaces
                    normals,offsets=face_offsets(surfaces,predicted,found)
                    if not len(normals):continue
                    raw=predicted+normals.T@(offsets-normals@predicted)
                    if len(normals)==1 and len(surfaces[2])==1:
                        raw+=tangent*(np.clip(raw@tangent,lower,upper)-raw@tangent)
                else:
                    for axis in range(2):
                        if not complete_top and hi[axis]-lo[axis]<.19:
                            raw[axis]=predicted[axis]
                center=raw@r.T+pose[:2]
                if np.linalg.norm(center-found['center'])>=max_correction:
                    continue
                found.update(center=.25*found['center']+.75*center,
                             size=self.box_size(found,r),time=frame.time)
            else:
                tracked={'center':center,'time':frame.time,'axes':axes@r.T}
                tracked['size']=self.box_size(tracked,r)
                self.obstacle_memory.append(tracked)
        # Occlusion does not mean free space. Keep an unseen box until the body
        # has actually passed it according to onboard odometry.
        self.obstacle_memory=[b for b in self.obstacle_memory if np.linalg.norm(b['center']-pose[:2])<3 and
                              (frame.time-b['time']<8 or ((b['center']-pose[:2])@r)[0]>-.4)]
        obstacles=np.array([[*((b['center']-pose[:2])@r),*b['size']] for b in self.obstacle_memory]).reshape(-1,4)
        clean=np.ones(len(local),bool)
        fresh_clean=np.ones(len(fresh),bool)
        for ox,oy,sx,sy in obstacles:
            clean&=(abs(local[:,0]-ox)>sx/2+.035)|(abs(local[:,1]-oy)>sy/2+.035)
            fresh_clean&=(abs(fresh[:,0]-ox)>sx/2+.035)|(abs(fresh[:,1]-oy)>sy/2+.035)
        local=local[clean]
        # Long-lived points bridge occlusion, but dead-reckoning drift makes
        # them weaker evidence than newly observed boundaries.
        reference=None
        if self.previous_lane is not None and self.last_pose is not None:
            x=np.linspace(-.1,1.5,40)
            previous=np.column_stack([x,np.polyval(self.previous_lane,x)])
            transformed=(previous@rotation2(self.last_pose[2]).T+self.last_pose[:2]-pose[:2])@r
            if np.ptp(transformed[:,0])>.3:reference=np.polyfit(transformed[:,0],transformed[:,1],2)
        lane=self.fit_lane(fresh[fresh_clean],reference)
        if lane is None:lane=self.fit_lane(local,reference)
        if lane is not None:self.previous_lane=lane.copy();self.last_pose=pose.copy()
        # A transverse white stripe must span most of the 50 cm lane.
        finish=self.finish_stripe(pts,lane)
        overlay=rgb.copy();overlay[white]=[255,80,255];overlay[raised]=[255,140,0]
        current_clean=np.ones(len(pts),bool)
        for ox,oy,sx,sy in obstacles:
            current_clean&=(abs(pts[:,0]-ox)>sx/2+.035)|(abs(pts[:,1]-oy)>sy/2+.035)
        current_white=pts[current_clean]
        lane_observed=False
        if lane is not None and len(current_white)>=12:
            current_x=current_white[:,0]
            slope=np.polyval(np.polyder(lane),current_x)
            offset=(current_white[:,1]-np.polyval(lane,current_x))/np.sqrt(1+slope*slope)
            for side in (-1,1):
                matching=abs(offset-side*.25)<.04
                if matching.sum()>=12 and np.ptp(current_x[matching])>.18:
                    lane_observed=True
                    break
        return Perception(frame.time,lane,local,obstacles,np.asarray(finish).reshape(-1,2),-ground,overlay,current_white,self.lane_support,lane_observed,None,current_white)


class VisualNavigator:
    def __init__(self, speed=.22, observation_lease=.35):
        self.speed=speed;self.command=np.zeros(3);self.perception=None
        # Freshness window for the last accepted image. It must match the runtime's
        # lease: perception on the Jetson produces a frame every 0.4--0.7 s, while
        # the simulator managed 0.07--0.15 s, so a fixed value would either stop the
        # real robot or accept stale images in simulation.
        self.observation_lease=float(observation_lease)
        self.path=np.empty((0,2));self.mode='initializing';self.finished=False
        self.finish_target=None;self.finish_normal=None;self.last_good=-1.;self.blocked_since=None
        self.effective_lane=None
        self.plan_stamp=None;self.cached_path=None;self.observed_pose=np.zeros(3)
        self.straight_headings=[];self.corridor_heading=None;self.corridor_point=None
        self.velocity_valid=False
        self.measured_velocity=np.zeros(2)
        self.velocity_pose=None
        self.velocity_time=None
        self.gap=GapController()
        # RT10: last translational command, held when no fresh plan exists.
        self.last_drive=np.zeros(2)

    def observe(self, perception, pose, turn, motion_observed=True):
        self.velocity_valid=False
        if motion_observed and self.velocity_time is not None and 0<perception.time-self.velocity_time<=self.observation_lease:
            dt=perception.time-self.velocity_time
            measured=(pose[:2]-self.velocity_pose)/dt
            if np.linalg.norm(measured)<1.:
                self.velocity_valid=True
                weight=min(1.,dt/.25)
                self.measured_velocity=(1-weight)*self.measured_velocity+weight*measured
        self.velocity_pose=pose[:2].copy();self.velocity_time=perception.time
        self.perception=perception
        self.observed_pose=pose.copy()
        alongside=any(-.4<o[0]<.85 and abs(o[1])<.8 for o in perception.obstacles)
        self.straight_headings=[v for v in self.straight_headings if perception.time-v[0]<1.5]
        if not alongside:
            self.corridor_heading=None;self.corridor_point=None
        if self.corridor_heading is None and perception.lane is not None:
            lane=perception.lane
            heading=np.arctan(np.polyval(np.polyder(lane),.1))
            if abs(lane[0])<.05 and abs(heading)<.15:
                value=pose[2]+heading
                point=pose[:2]+np.array([.1,np.polyval(lane,.1)])@rotation2(pose[2]).T
                self.straight_headings.append((perception.time,value,point))
                cluster=max(([v for v in self.straight_headings if abs((v[1]-candidate[1]+np.pi)%(2*np.pi)-np.pi)<.025] for candidate in self.straight_headings),key=len)
                # Do not discard all straight evidence after one occluded or
                # curved-looking frame. Require a time-spanning consensus.
                if alongside and len(cluster)>=3 and cluster[-1][0]-cluster[0][0]>=.12-1e-6:
                    self.corridor_heading=float(np.unwrap([v[1] for v in cluster]).mean())
                    self.corridor_point=np.mean([v[2] for v in cluster],axis=0)
        if self.corridor_point is not None and perception.current_white is not None:
            angle=self.corridor_heading-pose[2]
            normal=np.array([-np.sin(angle),np.cos(angle)])
            predicted=float(((self.corridor_point-pose[:2])@rotation2(pose[2]))@normal)
            cloud=perception.current_white
            values=cloud@normal
            tangent=np.array([normal[1],-normal[0]])
            along=cloud@tangent
            centers=[]
            for side in (-1,1):
                selected=abs(values-(predicted+side*.25))<.05
                if selected.sum()<12 or np.ptp(along[selected])<=.18:
                    continue
                line=np.polyfit(along[selected],values[selected],1)
                residual=values[selected]-np.polyval(line,along[selected])
                # A visible bend may intersect the expected white-line band.
                # Require a straight segment parallel to the established
                # corridor before interpreting it as a lateral displacement.
                if abs(line[0])>.025 or np.percentile(abs(residual),90)>.008:
                    continue
                centers.append(float(np.median(values[selected]))-side*.25)
            if centers and np.ptp(centers)<.025:
                world_normal=np.array([-np.sin(self.corridor_heading),np.cos(self.corridor_heading)])
                self.corridor_point+=.25*(np.median(centers)-predicted)*world_normal
        if self.corridor_point is not None and perception.lane is not None:
            slope=np.tan(self.corridor_heading-pose[2])
            point=(self.corridor_point-pose[:2])@rotation2(pose[2])
            predicted=point[1]-slope*point[0]
            lane=perception.lane
            if abs(lane[0])<.03 and abs(lane[1]-slope)<.02 and abs(lane[2]-predicted)<.025:
                normal=np.array([-np.sin(self.corridor_heading),np.cos(self.corridor_heading)])
                self.corridor_point+=.1*(lane[2]-predicted)*normal
        if perception.lane is not None:self.last_good=perception.time
        if turn>5.5 and len(perception.finish) and perception.lane is not None:
            f=np.median(perception.finish,axis=0)
            heading=pose[2]+np.arctan(np.polyval(np.polyder(perception.lane),f[0]))
            normal=np.array([np.cos(heading),np.sin(heading)])
            target=f@rotation2(pose[2]).T+pose[:2]
            if self.finish_target is None:
                if (target-pose[:2])@normal>0:
                    self.finish_target=target
                    self.finish_normal=normal
            elif (normal@self.finish_normal>.7 and
                  (target-pose[:2])@self.finish_normal>0 and
                  np.linalg.norm(target-self.finish_target)<.20):
                # Refresh the measured stripe while it is visible, so local
                # odometry corrections cannot leave a stale distant target.
                # Keep its approach normal fixed: turning in place must not
                # reverse the crossing test. After occlusion, only measured
                # motion relative to the last observed stripe can finish.
                self.finish_target=target

    def plan(self,p):
        step=.005
        edges=sorted(v for ox,oy,sx,sy in p.obstacles for v in (ox-sx/2-.135,ox+sx/2+.135))
        extra=[v for a,b in zip(edges,edges[1:]) for v in (a+1e-6,(a+b)/2,b-1e-6) if 0<v<1.6]
        x=np.unique(np.r_[np.arange(0,1.601,step),extra])
        center=np.polyval(p.lane,x)
        offsets=np.linspace(-.27,.27,109)
        y=center[:,None]+offsets[None,:]
        valid=np.ones(y.shape,bool)
        for ox,oy,sx,sy in p.obstacles:
            valid &= (abs(x[:,None]-ox)>=sx/2+.135)|(abs(y-oy)>=sy/2+.280)
        delta=offsets[:,None]-offsets[None,:]
        costs=np.zeros(len(offsets))
        parents=np.zeros(valid.shape,int)
        costs=2*offsets**2
        # Backward DP: the entire visible corridor informs the first maneuver.
        for i in range(len(x)-1,-1,-1):
            dx=x[i+1]-x[i] if i<len(x)-1 else step
            transition=8*(delta/dx)**2
            candidate=transition+costs[None,:]
            if i<len(x)-1:
                ya=center[i]+offsets[:,None]
                yb=center[i+1]+offsets[None,:]
                for ox,oy,sx,sy in p.obstacles:
                    enter=max(0.,(ox-sx/2-.135-x[i])/dx)
                    leave=min(1.,(ox+sx/2+.135-x[i])/dx)
                    if enter<leave:
                        a=ya+(yb-ya)*enter;b=ya+(yb-ya)*leave
                        blocked=(np.minimum(a,b)<oy+sy/2+.280)&(np.maximum(a,b)>oy-sy/2-.280)
                        candidate[blocked]=np.inf
            j=np.argmin(candidate,axis=1)
            costs=candidate[np.arange(len(offsets)),j]+2*offsets**2
            costs[~valid[i]]=np.inf
            parents[i]=j
        j=int(np.argmin(costs+100*(center[0]+offsets)**2))
        if not np.isfinite(costs[j]):return None
        choice=[]
        for i in range(len(x)):
            choice.append(j);j=parents[i,j]
        return np.column_stack([x,center+offsets[choice]])

    def update(self,time,pose,dt=.02):
        p=self.perception
        if p is not None and p.lane is not None and self.corridor_heading is not None:
            # After multiple images establish a straight narrow corridor, retain
            # its heading through box occlusion. A brief partial-line fit must
            # not rotate the arms into the boxes. Current images still correct
            # lateral position; missing images still stop the controller.
            angle=self.corridor_heading-self.observed_pose[2]
            point=(self.corridor_point-self.observed_pose[:2])@rotation2(self.observed_pose[2])
            slope=np.tan(angle)
            p=replace(p,lane=np.array([0.,slope,point[1]-slope*point[0]]))
        self.effective_lane=None if p is None or p.lane is None else p.lane.copy()
        desired=np.zeros(3)
        brake_forward=False
        gap_command=None
        if self.finished:
            self.mode='finished'
            # RT10: crossing the finish line is reported, not acted on.
            return self.command.copy()
        elif p is None or time-p.time>self.observation_lease or (p.lane is None and self.gap.phase=='idle'):
            self.mode='lost_vision'
            # RT10: keep the last request instead of standing still.
            return self.command.copy()
        else:
            if self.finish_target is not None:
                remaining=self.finish_target-pose[:2]
                tangent=np.array([-self.finish_normal[1],self.finish_normal[0]])
                if remaining@self.finish_normal<=0 and abs(remaining@tangent)<.25:
                    self.finished=True;self.mode='finished'
            if not self.finished:
                near=any(-.25<o[0]<.85 for o in p.obstacles)
                if self.gap.phase!='idle' or near:
                    gap_command=self.gap.update(time,p,self.observed_pose,pose,self.measured_velocity,self.corridor_heading,
                                                velocity_valid=self.velocity_valid)
                if gap_command is not None and np.any(gap_command):
                    self.mode='blocked' if self.gap.phase=='failed' else 'avoid'
                    yaw_error=(self.gap.heading-pose[2]+np.pi)%(2*np.pi)-np.pi
                    self.command[:2]=gap_command
                    self.command[2]=np.clip(2.5*yaw_error,-.35,.35)
                    return self.command.copy()
                # RT10: a gap hold/brake/failure no longer stops the walk; fall
                # through to the lateral planner or the lane servo instead.
                if self.plan_stamp!=p.time:
                    self.cached_path=self.plan(p) if near else np.column_stack([np.arange(0,1,.02),np.polyval(p.lane,np.arange(0,1,.02))])
                    self.plan_stamp=p.time
                path=self.cached_path
                if path is None:self.mode='blocked'
                else:
                    self.path=path;self.mode='avoid' if near else 'lane'
                    target_index=np.argmin(abs(path[:,0]-(.03 if near else .18)))
                    target=(path[target_index]@rotation2(self.observed_pose[2]).T+self.observed_pose[:2]-pose[:2])@rotation2(pose[2])
                    heading_x=.1 if near else .18
                    if p.lane_support is not None:heading_x=max(heading_x,p.lane_support[0])
                    support=p.current_white if p.current_white is not None and len(p.current_white) else p.white
                    observed_x=support[:,0]
                    observed_x=observed_x[np.isfinite(observed_x)&(observed_x>=0.)]
                    if len(observed_x):heading_x=max(heading_x,float(observed_x.min()))
                    # A curve fit can reverse its tangent when extrapolated
                    # into the camera's near blind region. Use observed support.
                    heading=np.arctan(np.polyval(np.polyder(p.lane),heading_x))+(self.observed_pose[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
                    desired=np.array([self.speed,np.clip(1.8*target[1],-.22,.22),np.clip(2.5*heading,-.8,.8)])
                    if near:
                        velocity=self.measured_velocity@rotation2(pose[2])
                        lateral_error=target[1]-.20*velocity[1]
                        lateral=np.clip(3*lateral_error,-.20,.20)
                        if abs(lateral_error)>.008:lateral=np.sign(lateral)*max(.10,abs(lateral))
                        else:lateral=0.
                        desired[:2]=[.20,lateral]
                        # Align with the open side before advancing into a single box.
                        if abs(target[1])>.06:
                            desired[0]=0.
                            brake_forward=True
        if np.any(desired[:2]):
            self.last_drive=desired[:2].copy()
        elif np.any(self.last_drive):
            # No plan (blocked/lost): keep translating instead of decaying to 0.
            desired[:2]=self.last_drive
        self.command+=np.clip(desired-self.command,-np.array([.4,.5,1.5])*dt,np.array([.4,.5,1.5])*dt)
        if brake_forward:self.command[0]=0.
        return self.command.copy()
