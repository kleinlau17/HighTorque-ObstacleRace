"""Camera-only local lane/obstacle perception and receding-horizon navigation.

No simulator, field configuration, saved route, or global map is accessible here.
Input: RGB-D, calibrated intrinsics/extrinsics, and estimated onboard odometry.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import cv2
import numpy as np
from navigation import params
from navigation.surface_odometry import SurfaceOdometry
from navigation.ground import GroundEstimator
from navigation.floor_odometry import FloorOdometry


def cluster_points(points, cell=None, min_pts=None):
    if cell is None: cell=params.get('box','cluster_cell')
    if min_pts is None: min_pts=params.get('box','cluster_min_pts')
    """Connected components on a 2D grid, so each obstacle is fitted alone."""
    xy=np.asarray(points,dtype=float)[:,:2]
    keys=np.floor(xy/cell).astype(np.int64); cells={}
    for i,(kx,ky) in enumerate(keys): cells.setdefault((int(kx),int(ky)),[]).append(i)
    seen=set(); out=[]; nb=[(a,b) for a in(-1,0,1) for b in(-1,0,1) if (a,b)!=(0,0)]
    for start in list(cells):
        if start in seen: continue
        stack=[start]; seen.add(start); mem=[]
        while stack:
            cur=stack.pop(); mem.extend(cells[cur])
            for a,b in nb:
                n=(cur[0]+a,cur[1]+b)
                if n in cells and n not in seen: seen.add(n); stack.append(n)
        if len(mem)>=min_pts: out.append(np.asarray(points,dtype=float)[mem])
    out.sort(key=len,reverse=True)
    return out


def fit_box(points, sx=None, sy=None, sz=None):
    if sx is None: sx=params.get('box','sx')
    if sy is None: sy=params.get('box','sy')
    if sz is None: sz=params.get('box','sz')
    """Fit a fixed 20x20x40 cm box: xy translation and z rotation only."""
    p=np.asarray(points,dtype=float)[:,:2]; keep=np.ones(len(p),bool)
    for ax in range(2):
        lo,hi=np.percentile(p[:,ax],[2,98]); keep&=(p[:,ax]>=lo)&(p[:,ax]<=hi)
    p=p[keep]
    if len(p)<8: return None
    best=None
    for deg in range(0,91,int(params.get('box','fit_step_deg'))):
        th=np.deg2rad(deg); c,s=np.cos(th),np.sin(th)
        xr=p[:,0]*c+p[:,1]*s; yr=-p[:,0]*s+p[:,1]*c
        ex=max(0.,xr.max()-xr.min()-sx)+max(0.,yr.max()-yr.min()-sy)
        if best is None or ex<best[0]-1e-9: best=(ex,deg,xr,yr)
    ex,deg,xr,yr=best; c,s=np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))
    cxr=.5*(xr.min()+xr.max()); cyr=.5*(yr.min()+yr.max())
    return dict(center=[float(cxr*c-cyr*s),float(cxr*s+cyr*c)],yaw=float(np.deg2rad(deg)),
                size=[sx,sy,sz],excess=float(ex),points=int(len(p)))


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
    raw_lane: np.ndarray | None = None
    edges: np.ndarray | None = None


class CameraPerception:
    def __init__(self, width, height, fovy, camera_position, camera_rotation, intrinsics=None):
        f = height/2/np.tan(np.deg2rad(fovy/2))
        fx,fy,cx,cy=intrinsics if intrinsics is not None else (f,f,(width-1)/2,(height-1)/2)
        u,v = np.meshgrid(np.arange(width), np.arange(height))
        self.rays = np.stack([(u-cx)/fx,(v-cy)/fy,np.ones_like(u)],-1).astype(np.float32)
        self.analysis = None
        self.raw_lane = None
        self.lane_edges = None
        self.lane_basis = 'no_fit'
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

    @staticmethod
    def _two_point(samples):
        (x0,y0),(x1,y1)=samples[0],samples[-1]
        slope=(y1-y0)/max(1e-6,float(x1-x0))
        return np.array([slope,y0-slope*x0])

    def _paired_no_fit(self, points):
        """Lane centre from binned medians of the two tape clusters, no RANSAC."""
        x,y=points[:,0],points[:,1]
        bins=np.linspace(.25,1.05,9)
        left=[];right=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(x>=lo)&(x<hi)
            if m.sum()<8: continue
            ys=np.sort(y[m])
            if ys[-1]-ys[0]<.30: continue
            gaps=np.diff(ys)
            k=int(gaps.argmax())
            if gaps[k]<params.get('vision','lane_gap'): continue
            cut=.5*(ys[k]+ys[k+1])
            l=ys[ys<cut]; r=ys[ys>=cut]
            if len(l)<3 or len(r)<3: continue
            c=.5*(lo+hi)
            left.append((c,float(np.median(l))))
            right.append((c,float(np.median(r))))
        if len(left)<3: return None
        l_line=self._two_point(left); r_line=self._two_point(right)
        xs=[p[0] for p in left]
        separation=[abs(np.polyval(l_line,c)-np.polyval(r_line,c)) for c in xs]
        if not all(.33<s<.85 for s in separation): return None
        centers=[(.5*(np.polyval(l_line,c)+np.polyval(r_line,c))) for c in xs]
        lane=self._two_point(list(zip(xs,centers)))
        self.lane_basis='paired_edges'
        self.lane_edges=np.vstack([l_line,r_line])
        self.lane_support=(float(min(xs)),float(max(xs)))
        self.steering_heading=float(np.arctan(np.polyval(np.polyder(lane),.4)))
        return lane

    def _binned_clusters(self, points):
        """x-binned medians, split into the two tape clusters. No RANSAC."""
        bins=np.linspace(params.get('vision','lane_x_lo'),params.get('vision','lane_x_hi'),int(params.get('vision','lane_bins'))); left=[]; right=[]; single=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(points[:,0]>=lo)&(points[:,0]<hi)
            if m.sum()<6: continue
            ys=np.sort(points[m,1]); c=.5*(lo+hi)
            gaps=np.diff(ys)
            if len(gaps) and gaps.max()>=.18 and ys[-1]-ys[0]>.30:
                k=int(gaps.argmax()); cut=.5*(ys[k]+ys[k+1])
                l=ys[ys<cut]; r=ys[ys>=cut]
                if len(l)>=3 and len(r)>=3:
                    left.append((c,float(np.median(l)))); right.append((c,float(np.median(r)))); continue
            single.append((c,float(np.median(ys))))
        return left,right,single

    def fit_lane(self, points, reference=None, require_known_side=False, inside_start=False):
        self.raw_lane = None
        self.lane_edges = None
        self.lane_basis = 'no_fit'
        self.steering_heading = None
        self.lane_support=None
        points = points[(points[:,0]>-.12)&(points[:,0]<1.4)&(abs(points[:,1])<1.4)]
        if len(points)<20:
            return None
        if len(points)>1024:
            points=points[np.linspace(0,len(points)-1,1024).astype(int)]
        left,right,single=self._binned_clusters(points)
        if len(left)>=2 and len(right)>=2:
            l_line=self._two_point(left); r_line=self._two_point(right)
            xs=[p[0] for p in left]
            separation=[abs(np.polyval(l_line,c)-np.polyval(r_line,c)) for c in xs]
            if all(params.get('vision','pair_min')<s<params.get('vision','pair_max') for s in separation):
                centers=[.5*(np.polyval(l_line,c)+np.polyval(r_line,c)) for c in xs]
                lane=self._two_point(list(zip(xs,centers)))
                self.lane_basis='paired_edges'
                self.lane_edges=np.vstack([l_line,r_line])
                self.lane_support=(float(min(xs)),float(max(xs)))
                self.steering_heading=float(np.arctan(np.polyval(np.polyder(lane),.5)))
                return lane
        if len(single)>=3:
            line=self._two_point(single)
            xs=[p[0] for p in single]
            identity_x=float(np.clip(.4,xs[0],xs[-1]))
            reference_y=0. if reference is None else float(np.polyval(reference,identity_x))
            slope=float(line[0])
            distance=(float(np.polyval(line,identity_x))-reference_y)/np.hypot(1.,slope)
            side=1 if distance>0 else -1
            self.raw_lane=line
            self.lane_basis='single_edge'
            if require_known_side and reference is not None:
                if abs(abs(distance)-.25)>.12:
                    self.lane_basis='edge_disagrees_with_history'
                    return None
                self.lane_basis='edge_with_history'
            elif require_known_side:
                self.lane_basis='edge_side_unknown'
                near=[p for p in single if p[0]<=xs[0]+.20]
                if not inside_start or len(near)<4 or np.ptp([p[0] for p in near])<.10:
                    return None
                tangent,intercept=np.polyfit([p[0] for p in near],[p[1] for p in near],1)
                body_distance=intercept/np.hypot(1.,tangent)
                if not .055<abs(body_distance)<.90 or abs(tangent)>1.65:
                    return None
                side=1 if body_distance>0 else -1
                self.lane_basis='initial_left_edge' if side>0 else 'initial_right_edge'
            nx=-slope/np.sqrt(1+slope*slope); ny=1/np.sqrt(1+slope*slope)
            center=[(x-side*.25*nx, y-side*.25*ny) for x,y in single]
            self.lane_support=(float(min(c[0] for c in center)),float(max(c[0] for c in center)))
            self.steering_heading=float(np.arctan(slope))
            return np.polyfit([c[0] for c in center],[c[1] for c in center],1)
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
                            -plane.z0, rgb, lane_observed=False, raw_lane=None)
        self.lane_basis = 'no_fit'
        self.raised_sites = self.floor_pixels = None
        if not plane.valid:
            self.start_edge = None
            return result, 'unknown_depth'
        height = plane.height(level)
        xy = level[..., :2]@rotation2(yaw).astype(np.float32)
        x, y = xy[..., 0], xy[..., 1]
        floor = finite & (abs(height)<params.get('vision','floor_band'))
        self.floor_pixels = floor
        self.clearance_depth = depth
        bright, dark = rgb.max(axis=2), rgb.min(axis=2)
        white = floor & (dark>params.get('vision','white_dark')) & (bright-dark<params.get('vision','white_ratio')*bright) & (x>params.get('vision','white_x_min'))
        points = xy[white]
        # One analysis per frame, shared by the lane and the avoidance modes.
        self.analysis=(level,finite,height,xy,points,floor,float(-plane.z0),float(yaw),frame.time,rgb)
        _fit_start = time.monotonic()
        lane = self.fit_lane(points, reference, require_known_side=True, inside_start=inside_start)
        self.last_fit_ms = (time.monotonic()-_fit_start)*1000.
        result.white = result.current_white = points
        result.lane, result.lane_support = lane, self.lane_support
        result.raw_lane = self.raw_lane
        result.edges = self.lane_edges
        result.lane_observed = lane is not None
        if lane is not None:
            result.steering_heading = self.steering_heading
            if result.steering_heading is None:
                samples = np.linspace(*self.lane_support,12)
                result.steering_heading = self._steering_heading(np.column_stack([samples,np.polyval(lane,samples)]))
        route = lane if lane is not None else reference
        center = np.polyval(route, x) if route is not None else np.zeros_like(x)
        # Cover both the robot and the lane on approach, including a box on either edge.
        in_path = (y>np.minimum(center, 0.)-params.get('vision','path_half')) & (y<np.maximum(center, 0.)+params.get('vision','path_half'))
        # A turn sweeps the wrist beyond the straight-ahead body width.
        in_path |= abs(y)<params.get('vision','path_abs')
        raised = finite & (height>params.get('vision','raised_h_min')) & (height<params.get('vision','raised_h_max')) & (x>params.get('vision','raised_x_min')) & (x<params.get('vision','raised_x_max')) & in_path
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

    def inspect_avoid(self, analysis):
        """Avoidance view of the same frame: its own line fit and raised test.

        It reuses the plane, level frame, white points and floor mask that the
        lane inspection already computed, so no second ground search runs.
        """
        level,finite,height,xy,points,floor,z0,yaw,t,rgb=analysis
        x,y=xy[...,0],xy[...,1]
        empty=np.empty((0,2))
        result=Perception(t,None,empty,np.empty((0,4)),empty,z0,rgb,lane_observed=False,raw_lane=None)
        self.lane_basis='no_fit'
        self.raised_sites=None
        self.floor_pixels=floor
        # Avoidance never builds a line model: it works on the white points.
        self.lane_basis='points'
        result.white=result.current_white=points
        result.lane=None; result.raw_lane=None; result.edges=None
        result.lane_observed=bool(len(points))
        route=None
        center=np.polyval(route,x) if route is not None else np.zeros_like(x)
        in_path=(y>np.minimum(center,0.)-params.get('vision','path_half'))&(y<np.maximum(center,0.)+params.get('vision','path_half'))
        in_path|=abs(y)<params.get('vision','path_abs')
        raised=finite&(height>params.get('vision','raised_h_min'))&(height<params.get('vision','raised_h_max'))&(x>params.get('vision','raised_x_min'))&(x<params.get('vision','raised_x_max'))&in_path
        supported=cv2.erode(raised.astype(np.uint8),np.ones((2,2),np.uint8))
        if cv2.countNonZero(supported):
            pts=level[supported!=0,:2]
            _,indices=np.unique(np.floor(pts/.04).astype(int),axis=0,return_index=True)
            self.raised_sites=np.column_stack((pts[indices],height[supported!=0][indices]))
            return result,'raised_object'
        return result,'clear'

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
        return Perception(frame.time,lane,local,obstacles,np.asarray(finish).reshape(-1,2),-ground,overlay,current_white,self.lane_support,lane_observed,None,self.raw_lane,self.lane_edges)
