"""Metric floor-tape observations and forward RGB-D clearance for the box section."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Edge:
    # y = slope*x + offset in the horizontal crate-heading frame, relative to the trunk.
    offset: float
    slope: float
    near: float
    far: float

    def at(self, x):
        return self.offset + self.slope*x


def _floor_tape(rgb, ground):
    bright, dark = rgb.max(axis=2), rgb.min(axis=2)
    neutral = (dark>115) & (bright-dark<.16*bright)
    # On the field the white tape can look pale blue against the green mat.
    # Require darker turf on BOTH sides of a tinted stripe in the same row.
    r, g, b = cv2.split(rgb)
    # The near tape is bluer than its far end under the head camera's auto
    # exposure. Retain it so clearance is based on paint close to the feet.
    tinted = ground & ~neutral & (dark>115) & (b==bright) & (r>=.70*g) & (dark>=.54*bright)
    if not tinted.any():
        return ground & neutral
    turf = ground & (r<.73*g) & (g>=.72*b)
    turf_green = np.where(turf, g, np.uint8(255))
    # At the working resolution a strip is a few pixels wide. A wide patch
    # or a carpet-to-box boundary lacks a close turf sample on one side.
    width = max(3, round(rgb.shape[1]*.04))
    window = np.ones((1,width),np.uint8)
    left = cv2.erode(turf_green,window,anchor=(width-1,0),borderType=cv2.BORDER_CONSTANT,borderValue=255)
    right = cv2.erode(turf_green,window,anchor=(0,0),borderType=cv2.BORDER_CONSTANT,borderValue=255)
    contrast = g.astype(np.int16)-np.maximum(left,right)
    return ground & (neutral | (tinted & (contrast>=32)))


def floor_edges(rgb, points, height, valid):
    x, y = points[..., 0], points[..., 1]
    ground = valid & (abs(height)<.024) & (x>.22) & (x<1.47) & (abs(y)<.85)
    paint = _floor_tape(rgb,ground)
    if np.count_nonzero(paint)<12:
        return []
    # Only join near-neighbour pixels; extrapolate a measured near straight, never a far bend.
    blobs = cv2.morphologyEx(paint.astype('uint8'), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)
    edges = []
    for k in range(1,count):
        if stats[k,cv2.CC_STAT_AREA]<12:
            continue
        sample = points[paint & (labels==k),:2]
        if len(sample)<12:
            continue
        near = np.percentile(sample[:,0],5)
        sample = sample[(sample[:,0]>=near)&(sample[:,0]<=near+.38)]
        if near>1.17 or len(sample)<12 or np.ptp(sample[:,0])<(.22 if near>.87 else .065):
            continue
        short=np.ptp(sample[:,0])<.17
        bins = np.floor((sample[:,0]-near)/(.014 if short else .035)).astype(int)
        sections = [np.median(sample[bins==j],axis=0) for j in np.unique(bins)
                    if np.count_nonzero(bins==j)>=2]
        if len(sections)<(4 if short else 5):
            continue
        xs, ys = np.asarray(sections).T
        if np.ptp(xs)<(.055 if short else .135) or np.max(np.diff(xs))>(.031 if short else .084):
            continue
        slope, intercept = np.polyfit(xs,ys,1)
        residual = abs(sample[:,1]-slope*sample[:,0]-intercept)
        if abs(slope)>.78 or np.percentile(residual,85)>(.013 if short else .016):
            continue
        edges.append(Edge(float(intercept),float(slope),float(xs[0]),float(xs[-1])))
    return edges


def _pairs(edges):
    for i, a in enumerate(edges):
        for b in edges[:i]:
            if max(a.near,b.near)>.87 or min(a.far-a.near,b.far-b.near)<.14:
                continue
            lo, hi = max(a.near, b.near), min(a.far, b.far)
            if lo-hi>.14 or abs(a.slope-b.slope)>.13:
                continue
            x=(lo+hi)*.5
            norm=np.hypot(1.,(a.slope+b.slope)*.5)
            width=abs(a.at(x)-b.at(x))/norm
            if .435<width<.565:
                yield abs(width-.50), tuple(sorted((a,b),key=lambda e:e.at(x)))


def _box_distance(edge, box_xy):
    x,y=box_xy
    if edge.far-edge.near<.14 or edge.near>.87 or x<edge.near-.65 or x>edge.far+.85:
        return None
    return (edge.at(x)-y)/np.hypot(1.,edge.slope)


def measured_pass_side(edges, box_xy):
    """An obstacle against one stripe is 10 cm from it and 40 cm from the other."""
    votes=set()
    for edge in edges:
        gap=_box_distance(edge,box_xy)
        if gap is None:
            continue
        if abs(abs(gap)-.40)<.078:
            votes.add(np.sign(gap))
        elif abs(abs(gap)-.10)<.052:
            votes.add(-np.sign(gap))
    return votes.pop() if len(votes)==1 else 0.


def opposite_edge(edges, side, box_xy):
    """Classify a presently observed far stripe using a pair or the current box."""
    if not side:
        return None
    candidates=sorted(_pairs(edges),key=lambda p:p[0])
    if candidates:
        score,pair=candidates[0]
        best=pair[int(side>0)]
        x=(best.near+best.far)*.5
        if all(value-score>.045 or abs(other[int(side>0)].at(x)-best.at(x))<.07
               for value,other in candidates[1:]):
            return best
    if box_xy is not None:
        scores=[]
        for edge in edges:
            distance=_box_distance(edge,box_xy)
            if distance is not None and abs(side*distance-.40)<=.085:
                scores.append((abs(side*distance-.40),edge))
        if scores:
            return min(scores,key=lambda item:item[0])[1]
    return None


def clipped_paint_pair(edges,anchors):
    """Identify current paint fragments using their measured width and crate faces."""
    matches=[]
    # A crate can mask the middle of one stripe and the near end of
    # the other. Independently test both current pieces at their own
    # measured longitudinal locations against a live crate position.
    for j,a in enumerate(edges):
        for b in edges[:j]:
            if min(a.far-a.near,b.far-b.near)>=.14:
                continue
            long,short=sorted((a,b),key=lambda e:e.far-e.near,reverse=True)
            if (long.far-long.near<.24 or abs(long.slope-short.slope)>.075 or
                    max(a.near,b.near)-min(a.far,b.far)>.36 or not anchors):
                continue
            slope=long.slope
            x=.5*(short.near+short.far)
            pair=sorted((a,b),key=lambda e:e.at(x))
            projected=[]
            for e in pair:
                sample_x=.5*(e.near+e.far)
                projected.append(Edge(e.at(sample_x)-slope*sample_x,slope,e.near,e.far))
            norm=np.hypot(1.,slope)
            width=(projected[1].offset-projected[0].offset)/norm
            if abs(width-.50)>.027:
                continue
            for pass_side,box in anchors:
                if box[0]<min(a.near,b.near)-.65 or box[0]>max(a.far,b.far)+.60:
                    continue
                offsets=[(e.at(box[0])-box[1])/norm for e in projected]
                expected=[pass_side*(.40 if pass_side==side else -.10) for side in (-1.,1.)]
                err=max(abs(v-w) for v,w in zip(offsets,expected))
                if err<=.048:
                    matches.append((err,tuple(projected)))
                    break
    return min(matches,key=lambda item:item[0])[1] if matches else (None,None)


class BoundaryTracks:
    """Associate fresh paint samples to right/left identities across camera frames."""
    def __init__(self):
        self.seen=[None,None]
        self.last_stamp=-np.inf
        self.near_seen=[None,None]

    def observe(self, edges, stamp, position, anchors):
        if stamp<=self.last_stamp:
            return (None,None)
        self.last_stamp=stamp
        options=[[],[]]
        priors=[]
        for old in self.seen:
            if old is None or not 0<stamp-old[0]<=1.6:
                priors.append(None)
                continue
            _,pos,e=old
            dx,dy=position-pos
            priors.append(Edge(e.offset+e.slope*dx-dy,e.slope,e.near-dx,e.far-dx))
        def continuity(k,edge):
            prev=priors[k]
            if prev is None:
                return None
            x=(edge.near+edge.far)*.5
            # Restrict projection along a straight segment into an unseen bend.
            if x<prev.near-.42 or x>prev.far+.42 or abs(prev.slope-edge.slope)>.17:
                return None
            return abs(prev.at(x)-edge.at(x))/np.hypot(1.,edge.slope)
        for width_error,pair in _pairs(edges):
            errors=[continuity(k,e) for k,e in enumerate(pair)]
            if any(v is not None and v>.105 for v in errors):
                continue
            for k,e in enumerate(pair):
                options[k].append((width_error+.12*sum(v or 0. for v in errors),e))
        for edge in edges:
            for k,side in enumerate((-1.,1.)):
                old_error=continuity(k,edge)
                if edge.far-edge.near<.14:
                    # The crate or image border can clip most of the tape.
                    # Every output here is centred on this capture's own
                    # RGB-D paint. Only the tangent/identity come from an
                    # earlier, independently established long stripe.
                    x=.5*(edge.near+edge.far)
                    for old_k,old in enumerate(self.near_seen):
                        if old is None:
                            continue
                        ts,base,near=old
                        if not 0.<stamp-ts<8. or np.linalg.norm(position-base)>.65:
                            continue
                        dx,dy=position-base
                        norm=np.hypot(1.,near.slope)
                        expected=near.at(x+dx)-dy+(side*.50*norm if old_k!=k else 0.)
                        residual=abs(edge.at(x)-expected)
                        if residual<.050 and abs(edge.slope-near.slope)<.19:
                            measured=Edge(edge.at(x)-near.slope*x,near.slope,edge.near,edge.far)
                            options[k].append((.05+residual,measured))
                    continue
                if edge.near>.87:
                    # A distant piece alone cannot provide the tangent at
                    # the trunk. Use its measured displacement only with a
                    # recent nearby observation of the very same straight.
                    anchor=self.near_seen[k]
                    if anchor is not None and 0.<stamp-anchor[0]<1.5:
                        _,base,near=anchor
                        dx,dy=position-base
                        x=(edge.near+edge.far)*.5
                        expected=near.at(x+dx)-dy
                        if abs(edge.slope-near.slope)<.055 and abs(edge.at(x)-expected)<.045:
                            measured=Edge(edge.at(x)-near.slope*x,near.slope,edge.near,edge.far)
                            options[k].append((.055+abs(edge.at(x)-expected),measured))
                    continue
                if old_error is not None and old_error<.085:
                    options[k].append((.025+old_error,edge))
                other=priors[1-k]
                if other is not None and abs(edge.slope-other.slope)<.08:
                    x=(edge.near+edge.far)*.5
                    distance=(edge.at(x)-other.at(x))/np.hypot(1.,edge.slope)
                    if other.near-.30<=x<=other.far+.30 and abs(distance-side*.50)<.055:
                        options[k].append((.040+abs(distance-side*.50),edge))
                for pass_side,box in anchors:
                    distance=_box_distance(edge,box)
                    if distance is None:
                        continue
                    expected=pass_side*(.40 if pass_side==side else -.10)
                    error=abs(distance-expected)
                    if error<(.085 if pass_side==side else .058):
                        if old_error is None or old_error<.135:
                            options[k].append((.038+error,edge))
        result=[]
        for choices in options:
            choices.sort(key=lambda p:p[0])
            edge=None
            if choices:
                score,best=choices[0]
                x=.5*(best.near+best.far)
                if all(abs(best.at(x)-other.at(x))<.07 or value-score>.048
                       for value,other in choices[1:]):
                    edge=best
            result.append(edge)
        if result[0] is not None and result[0] is result[1]:
            return (None,None)
        for k,edge in enumerate(result):
            if edge is not None:
                self.seen[k]=(stamp,position.copy(),edge)
                if edge.near<=.87 and edge.far-edge.near>=.14:self.near_seen[k]=self.seen[k]
        return tuple(result)


def _measured_fronts(x, y, height, valid, anchors):
    """Match current onboard IDs to front planes supported across both face axes."""
    fronts = []
    for side, (cx,cy) in anchors:
        expected = cx - .10
        face = (valid & (abs(x-expected)<.030) & (abs(y-cy)<.120) &
                (height>.035) & (height<.372))
        if np.count_nonzero(face)>=32:
            samples = x[face]
            bins = np.rint((samples-expected)/.004).astype(int)+8
            peak = expected + .004*(np.argmax(np.bincount(bins))-8)
            on = face & (abs(x-peak)<.006)
            if np.count_nonzero(on)>=32 and np.ptp(y[on])>=.052 and np.ptp(height[on])>=.075:
                lo,hi = np.percentile(y[face],[1,99])
                fronts.append((float(np.median(x[on])),float(cy),side,hi-.108,lo+.108))
                continue
        # At a steep viewing angle only the roof and the side facing the
        # camera remain visible. A near-complete 20 cm roof in x gives a
        # conservative near plane: even its farthest return is inside the box.
        roof = valid & (abs(x-cx)<.12) & (abs(y-cy)<.12) & (abs(height-.40)<.013)
        if np.count_nonzero(roof)>=48:
            near,far = np.percentile(x[roof],[1,99])
            if far-near>.164 and np.ptp(y[roof])>.040 and abs(far-.20-expected)<.036:
                lo,hi = np.percentile(y[roof],[1,99])
                fronts.append((float(far-.20),float(cy),side,hi-.108,lo+.108))
    return fronts


def _limb_front(limbs, radii, cy):
    # Nearest points of an expanded box side and each horizontal limb sphere.
    # Keep a central trunk envelope in addition to the encoder-derived extremities.
    dy = np.maximum(abs(limbs[:,1]-cy)-.115,0.)
    near = dy<radii
    reach = .145 if abs(cy)<=.17+.115 else -np.inf
    if np.any(near):
        reach = max(reach,np.max(limbs[near,0]+np.sqrt(radii[near]**2-dy[near]**2)))
    return reach


def forward_clearance(points, height, valid, body, depth, camera, attitude, basis, plane,
                      bounds=(), step=.063, anchors=(), diagnostics=None):
    """Confirm a short forward sweep with current depth and the 20 cm course crates."""
    def unobserved(reason):
        if diagnostics is not None:
            diagnostics['reason'] = reason
        return 0., 'depth_unobserved'
    if not plane.valid:
        return unobserved('missing_ground_plane')
    if body is None or not len(body):
        return unobserved('missing_body_geometry')
    limbs = body[:,:2] @ basis
    left = max(.17, np.max(limbs[:,1] + body[:,3])) + .018
    right = min(-.17, np.min(limbs[:,1] - body[:,3])) - .018
    front = max(.145, np.max(limbs[:,0] + body[:,3])) + .020
    x, y = points[...,0], points[...,1]
    inside = valid & (y >= right) & (y <= left)
    if not .01 <= step <= .12:
        return unobserved('invalid_step_length')
    raised = valid & (height>.045) & (height<.65)
    surfaces = _measured_fronts(x,y,height,valid,anchors)
    modeled = np.zeros_like(valid)
    limit = step
    for face,cy,*_ in surfaces:
        hull = ((x>=face-.008) & (x<=face+.214) & (abs(y-cy)<=.118) & (height<.425))
        modeled |= hull
        # The face measurement sets the real axial stop plane. Only limbs
        # capable of touching this box contribute their swept front extent.
        reach = _limb_front(limbs,body[:,3]+.012,cy)
        if face+.20 > .045 and face > .045:
            limit = min(limit,max(0.,face-.012-reach))

    # A raised patch without a verified front plane may be the side or back
    # of a 20 cm box. The visible point alone is not its nearest surface.
    unknown = inside & raised & ~modeled & (x>.045) & (x<front+step+.20)
    if np.any(unknown):
        adjacent = cv2.filter2D(unknown.astype('uint8'),cv2.CV_16U,np.ones((3,3),np.uint8))
        occupied = unknown & (adjacent>=3)
        if np.any(occupied):
            limit = min(limit,max(0.,float(x[occupied].min())-.20-front))
    if limit < min(step,.020):
        if diagnostics is not None:
            diagnostics['reason'] = 'obstacle_in_step'
        return 0., 'depth_blocked'

    # Far ground may be hidden behind a known, independently ranged crate.
    # Require floor in the other stripes and a measured ground plane now.
    floor = inside & (abs(height)<.032) & (x>max(.35,front+.12)) & (x<.80)
    borders = np.linspace(right,left,6)
    floor_bins = np.histogram(y[floor],bins=borders)[0]>=8
    if np.count_nonzero(floor_bins)<2:
        return unobserved('insufficient_floor')
    # Require observed range in each shadowed band as well. Project only
    # actual returns on verified safe box surfaces along their camera rays
    # to the floor plane, using its measured height and camera extrinsics.
    shadow = modeled & raised
    if np.any(shadow):
        lens = camera.position32 @ attitude.T
        lens_height = lens @ plane.normal + plane.offset
        lens_xy = lens[:2] @ basis
        ratio = lens_height/np.maximum(lens_height-height[shadow],.001)
        shadow_x = lens_xy[0]+ratio*(x[shadow]-lens_xy[0])
        shadow_y = lens_xy[1]+ratio*(y[shadow]-lens_xy[1])
        near_ground = (shadow_x>max(.35,front+.12)) & (shadow_x<.80)
        floor_bins |= np.histogram(shadow_y[near_ground],bins=borders)[0]>=8
    if not np.all(floor_bins):
        return unobserved('missing_floor_band')

    # A crate with near face anywhere from front to front+limit includes this
    # longitudinal slab. A single verified ray through its interior rules out
    # that placement. Use several x and height samples to accommodate the FOV.
    # Only a currently identified stripe can restrict possible crate centers.
    half = .10
    low, high = right-half, left+half
    # Two opposite, separately measured front planes pin the two rows of
    # course crates even while the nearest painted edge is occluded.
    row_intervals = None
    for _,ay,sa,*_ in surfaces:
        for _,by,sb,*_ in surfaces:
            if sa*sb<0 and .245<abs(ay-by)<.355:
                row_intervals = [(ay-.065,ay+.065),(by-.065,by+.065)]
                low = max(low,min(ay,by)-.065)
                high = min(high,max(ay,by)+.065)
    near_obstacle = unknown
    edges = list(bounds)
    if len(edges) == 2 and sum(edge is not None for edge in edges) == 1:
        # The lane has a known 50 cm width. This is an outside bound on the
        # unseen stripe (56.5 cm normal distance), not an observed alignment.
        missing = int(edges[1] is None)
        seen = edges[1-missing]
        width = .565*np.hypot(1.,seen.slope)
        outer = seen.offset + (width if missing else -width)
        edges[missing] = Edge(outer,seen.slope,seen.near,seen.far)
    for i, edge in enumerate(edges):
        if edge is None or edge.near > front+.60 or edge.far < front-.12:
            continue
        near, far = front+half, front+limit+half
        positions = [edge.at(near), edge.at(far)]
        # A course crate stands wholly on the lane side of the stripe.
        # Allow 4 cm for the measured stripe width and line extrapolation.
        # A visible patch within half a crate of the stripe may belong to an
        # object straddling it. Keep the whole swept width on that side.
        outside = y < edge.at(x)+.10 if i == 0 else y > edge.at(x)-.10
        contrary = near_obstacle & outside
        if np.any(contrary):
            supported = cv2.filter2D(contrary.astype('uint8'),cv2.CV_16U,np.ones((3,3),np.uint8))
            if np.any(contrary & (supported >= 3)):
                continue
        if bounds[i] is not None:
            sign = 1. if i==0 else -1.
            scale = np.hypot(1.,edge.slope)
            for face,cy,_,center_lo,center_hi in surfaces:
                distance = sign*(cy-edge.at(face+.10))/scale
                if min(abs(distance-.10),abs(distance-.40))<.043:
                    # Face or roof pixels constrain the center independently:
                    # center >= max(y)-10cm and <= min(y)+10cm. The opposite
                    # row is 30cm away in a 50cm lane with flush 20cm crates.
                    other = sign*(.30 if distance<.25 else -.30)*scale
                    drift = abs(edge.slope*(face-front))+.012
                    near_offset = .10 if distance<.25 else .40
                    tape_center = edge.at(front+.10+.5*limit)+sign*near_offset*scale
                    observed_lo = max(center_lo-drift,tape_center-.055)
                    observed_hi = min(center_hi+drift,tape_center+.055)
                    if observed_lo<=observed_hi:
                        row_intervals = [(observed_lo,observed_hi),
                                         (observed_lo+other,observed_hi+other)]
                    break
        if i == 0:
            low = max(low, min(positions)+half-.04)
        else:
            high = min(high, max(positions)-half+.04)
    if low >= high:
        return unobserved('inconsistent_corridor')
    columns = np.arange(low-half, high+half+.005, .010)
    stations = front + np.linspace(limit+.005, .196, 6)
    heights = np.arange(.025, .388, .020)
    xx, yy, hh = np.meshgrid(stations,columns,heights,indexing='ij')
    xy = np.column_stack((xx.ravel(),yy.ravel())) @ basis.T
    vertical = (hh.ravel() - xy @ plane.normal[:2] - plane.offset) / plane.normal[2]
    optical = (np.column_stack((xy,vertical)) @ attitude - camera.position32) @ camera.rotation32
    z = optical[:,2]
    fx,fy,cx,cy = camera.intrinsics
    u,v = np.rint(optical[:,:2] / np.maximum(z,.01)[:,None] * [fx,fy] + [cx,cy]).astype(int).T
    h,w = depth.shape
    visible = (z > .10) & (u >= 1) & (u < w-1) & (v >= 1) & (v < h-1)
    u,v = np.clip(u,1,w-2), np.clip(v,1,h-2)
    samples = depth[v[:,None]+[0,0,0,-1,1], u[:,None]+[0,-1,1,0,0]]
    beyond = np.isfinite(samples) & (samples > z[:,None]+.025) & (samples < 4.)
    # The center pixel itself must supply range; neighbors reject isolated
    # flying pixels. The 4 mm interior inset covers the pixel footprint.
    free = visible & beyond[:,0] & (beyond.sum(axis=1) >= 3)
    open_columns = columns[free.reshape(xx.shape).any(axis=(0,2))]
    centers = np.linspace(low,high,int(np.ceil((high-low)/.018))+1)
    # For a continuous interval of possible centers, a witness must be inside
    # EVERY crate in that interval, including placements at both endpoints.
    covered = ((open_columns[None,:] >= centers[1:,None]-half+.004) &
               (open_columns[None,:] <= centers[:-1,None]+half-.004))
    proved = np.any(covered,axis=1)
    # A second crate cannot occupy the interior of a confirmed crate. Its
    # measured front must lie inside every hypothetical footprint in the bin.
    for face,cy,*_ in surfaces:
        if max(front+limit+.008,face+.015)<min(front+.192,face+.184):
            proved |= (centers[1:]<cy+.183) & (centers[:-1]>cy-.183)
    if row_intervals is not None:
        possible = np.zeros_like(proved)
        for row_min,row_max in row_intervals:
            possible |= (centers[1:]>=row_min) & (centers[:-1]<=row_max)
        if np.any(possible):
            proved = proved[possible]
    if not np.all(proved):
        # A visible obstruction shortened this very step. Do not turn that
        # positive measurement into a mere field-of-view gap for line following.
        if limit<step-1.e-6:
            if diagnostics is not None:
                diagnostics['reason']='obstacle_in_step'
            return 0.,'depth_blocked'
        return unobserved('unseen_swept_path')
    return float(limit), 'depth_clear' if limit >= step else 'depth_limited'
