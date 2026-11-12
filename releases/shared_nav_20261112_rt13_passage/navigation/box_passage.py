"""Select a passing side for each observed crate using metric tape and floor measurements."""
from dataclasses import dataclass, field
import cv2
import numpy as np

from navigation.gap_controller import GapController
from navigation.box_view_recovery import ViewFloor


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
    tinted = ground & ~neutral & (dark>115) & (b==bright) & (r>=.73*g) & (dark>=.57*bright)
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


# The existing Pro gait leaves .285 m from the trunk to the box-side wrist.
# Track paint is a foot-support boundary (the outer arm may overhang paint).
BOX_HALF = .10
LIMB_MARGIN = .010
# The shipped path puts the trunk up to 2 cm over the stripe on this course.
# One foot must still have supported ground on the inside of that stripe.
TRUNK_OVER_PAINT = .020
MIN_OPEN_WIDTH = GapController.side_reach + LIMB_MARGIN - TRUNK_OVER_PAINT
# Allow measurements of the crate's own surface just beyond its nominal
# footprint. Charge that whole extension against the usable floor width.
FACE_TOLERANCE = .012
MAX_FACE_EXTENSION = .025


def measure_passages(edges, points, heights, center, previous_offset=0.):
    widths, offsets, blocked, _ = _measure_passages(edges, points, heights, center, previous_offset, None)
    return widths, offsets, blocked


def _measure_passages(edges, points, heights, center, previous_offset, boundary, view=None):
    """Right/left measured gap widths and usable target offsets, in metres."""
    cx, cy = center
    widths = [None, None]
    offsets = [None, None]
    blocked = [False, False]
    observed_edges = [None, None]
    x = points[:,0]-cx
    raised = (heights>.10)&(heights<.65)
    along_face = abs(x)<=BOX_HALF+FACE_TOLERANCE
    for side_index, sign in enumerate((-1, 1)):
        from_center = sign*(points[:,1]-cy)
        near_face = (raised & along_face & (from_center>BOX_HALF+FACE_TOLERANCE)
                     & (from_center<=BOX_HALF+MAX_FACE_EXTENSION))
        face = float(np.max(from_center[near_face])) if np.any(near_face) else BOX_HALF
        distances = []
        under_crate = []
        for edge in edges:
            if cx < edge.near-.35 or cx > edge.far+.65 or abs(edge.slope)>.35:
                continue
            # Measure at both ends of the box. Take the narrower clearance.
            ends = [sign*(edge.at(station)-cy) for station in (cx-BOX_HALF,cx+BOX_HALF)]
            bound = min(ends)
            if 0 < bound <= .55:
                if max(ends)<=face and sign*edge.at(cx-.22)<0 and sign*edge.at(cx-BOX_HALF)<0:
                    under_crate.append(bound-face)
                else:
                    distances.append((bound-face, edge))
        # A protrusion beside the observed face consumes clearance. A return
        # further into the passage, or ahead of that face, can revoke it even
        # when the paint is out of view.
        y = from_center-face
        arm_limit = abs(previous_offset) if sign*previous_offset>0 else .42
        swept = ((abs(x)<.16) & (from_center>BOX_HALF+FACE_TOLERANCE)
                 & (from_center<arm_limit+GapController.side_reach) & raised & ~near_face)
        blocked[side_index] = np.count_nonzero(swept)>=5
        if sign*previous_offset>0 and face+GapController.side_reach+LIMB_MARGIN>abs(previous_offset):
            blocked[side_index] = True
        # A nearby stripe extrapolated through a crate can switch apparent
        # sides as its last visible segment shortens. Only disambiguate it
        # if the real outer stripe is independently remeasured at the
        # previously confirmed clearance for this crate, with floor support.
        gap_volume=(abs(x)<.20)&(from_center>face+.012)&(from_center<arm_limit+GapController.side_reach)
        if np.count_nonzero(gap_volume&(heights<-.055))>=4:
            blocked[side_index]=True
        if blocked[side_index] and view is not None:
            view.clear_registered(side_index)
        if not distances and not under_crate:
            continue
        width, outer = min(distances, key=lambda item:item[0]) if distances else (0., None)
        matched_outer = False
        if outer is not None and boundary is not None and under_crate and cx<.75:
            old_sign, old_y, old_slope = boundary
            matched_outer = (old_sign==sign and outer.near<=cx+.035 and outer.far>=cx-.16
                             and abs(outer.at(cx)-cy-old_y)<.035 and abs(outer.slope-old_slope)<.12)
        if under_crate and not matched_outer:
            width = 0.
        width = max(0., width)
        widths[side_index] = width
        if width < MIN_OPEN_WIDTH or blocked[side_index]:
            blocked[side_index] = True
            continue
        # Confirm an area of measured floor across the gap, including the
        # entrance and the box itself. Missing/invalid depth cannot count.
        gap = (x>=-.22)&(x<=.10)&(y>.012)&(y<width)
        floor = gap & (abs(heights)<.032)
        samples = np.histogram2d(x[floor],y[floor],
            bins=([-.22,-.10,.10],np.linspace(.012,width,4)))[0]
        full=np.all(samples>=3)
        if not full and matched_outer and view is not None:
            # Current stripe and farther floor remain visible. Cover each
            # missing entrance cell across the measured pose uncertainty.
            full=view.registered_floor(side_index, width, face, center, samples, outer)
        if not full:
            if matched_outer:
                blocked[side_index] = True
            continue
        # Keep the established .42 m centerline for a nominal 30 cm gap,
        # allowing 5 mm stripe-fit uncertainty. Tuck inward for a narrower gap.
        offsets[side_index] = .42 if width>=.295 else face+width+TRUNK_OVER_PAINT
        observed_edges[side_index] = ((sign, outer.at(cx)-cy, outer.slope), matched_outer)
    return tuple(widths), tuple(offsets), tuple(blocked), observed_edges


@dataclass
class PassageChoice:
    view_active: bool = field(default=True,init=False,repr=False)
    relative: object = field(default=None,init=False)
    view_floor: object = field(default_factory=ViewFloor,init=False,repr=False)
    offset: float = 0.
    widths: tuple = (None, None)
    vote_side: int = 0
    vote_time: float = -np.inf
    seen_at: float = -np.inf
    boundary: tuple = ()
    boundary_stamp: float = -np.inf

    view_position: object = field(default=None,init=False)
    cells: dict = field(default_factory=dict,init=False,repr=False)
    current_center: object = field(default=None,init=False,repr=False)
    frame_interval: float = field(default=0.,init=False,repr=False)

    def clear_registered(self,side):
        self.cells.pop(side,None)

    def remember_floor(self,stamp,xy,heights,center):
        if not self.boundary or self.view_position is None:
            return
        sign,edge_y,slope=self.boundary
        side=int(sign>0)
        actual=np.asarray(self.view_position)+center
        survey=self.cells.get(side)
        if (survey is None or abs(survey[2][0]-actual[0])>.16 or abs(survey[2][1]-actual[1])>.035
                or abs(survey[0]-edge_y)>.030 or abs(survey[1]-slope)>.10):
            # Cells are in this measured crate's coordinate system. A relock
            # of the body/crate or a different outer stripe starts a new survey.
            survey=(edge_y,slope,actual,np.full((16,12),-np.inf))
            self.cells[side]=survey
        self.current_center=actual
        rel=(xy-center)+(actual-survey[2])
        valid=(abs(heights)<.032)&(rel[:,0]>=-.38)&(rel[:,0]<.26)
        x=rel[valid,0]
        y=sign*rel[valid,1]
        counts=np.histogram2d(x,y,bins=(np.linspace(-.38,.26,17),np.linspace(.06,.54,13)))[0]
        survey[3][counts>=4]=stamp

    def registered_floor(self,side,width,face,center,counts,outer):
        survey=self.cells.get(side)
        if survey is None or self.current_center is None:
            return False
        edge_y,slope,_,stamps=survey
        if abs(edge_y-(outer.at(center[0])-center[1]))>.030 or abs(slope-outer.slope)>.10:
            return False
        # Current RGB-D must still measure this stripe and supported floor at
        # the near face. Only the entrance, already covered by the approaching
        # camera before the lower rim obscured it, may age briefly.
        if not np.all(counts[1]>=5) or center[0]>.75:
            return False
        # A clipped front face makes the longitudinal center less precise.
        # Check ground throughout both possible registrations (crate based
        # and odometry based), including every intervening cell.
        dx,dy=self.current_center-survey[2]
        sign=1 if side else -1
        dy*=sign
        ys=.08+.04*np.arange(12)
        xs=-.36+.04*np.arange(16)
        inside=(ys>=face+.015+min(0.,dy))&(ys<face+width-.006+max(0.,dy))
        entrance=(xs>=-.22+min(0.,dx))&(xs<=-.10+max(0.,dx))
        forward=(xs>=-.10+min(0.,dx))&(xs<=.10+max(0.,dx))
        if inside.sum()<5:
            return False
        # Check a four-second survey plus at most one continuous camera
        # interval. A gap in captures does not extend the lifetime.
        grace=self.frame_interval if 0.<self.frame_interval<=.25 else 0.
        fresh=self.seen_at-stamps[:,inside]<4.+grace
        return bool(np.all(fresh[entrance]) and np.all(fresh[forward].any(axis=0)))

    def observe(self, stamp, edges, xy, heights, center, max_age, lane_offset=None):
        if stamp<=self.seen_at:
            return
        self.frame_interval=stamp-self.seen_at
        self.seen_at = stamp
        self.relative=np.array(center,dtype=float,copy=True)
        if self.view_active:
            self.remember_floor(stamp,xy,heights,self.relative)
            self.view_floor.observe(stamp,xy,heights,self.relative)
        fresh_boundary = stamp-self.boundary_stamp<=max_age
        self.widths, offsets, blocked, boundaries = _measure_passages(
            edges, xy, heights, center, self.offset, self.boundary or None, self if self.view_active else None)
        if self.offset and not blocked[int(self.offset>0)]:
            current = boundaries[int(self.offset>0)]
            if current is None:
                return
            if fresh_boundary or not current[1]:
                self.boundary, self.boundary_stamp = current[0], stamp
                return
        self.offset = 0.
        available = [i for i in (0,1) if offsets[i] is not None]
        if not available:
            self.vote_side = 0
            return
        best = max(available, key=lambda i:self.widths[i])
        sign = 1 if best else -1
        # Two boundaries enclosing the 50 cm track independently identify a
        # boxed-in side. Otherwise require agreement across separate captures.
        other = self.widths[1-best]
        paired = (other is not None and blocked[1-best]
                  and abs(self.widths[best]+other+2*BOX_HALF-.50)<.035)
        # A model of the lane fitted to current paint may also identify one
        # stripe; it cannot create an opening without the depth-supported gap.
        lane_confirmed = (lane_offset is not None and
                          abs(sign*lane_offset-(self.widths[best]+BOX_HALF-.25))<.045)
        boundary, disambiguated = boundaries[best]
        repeat = self.vote_side==sign and stamp-self.vote_time<=max_age
        # After a gap, reacquire an ambiguous boundary only on two distinct
        # floor-supported exposures. Its remembered geometry alone grants no motion.
        corroborated = (paired or lane_confirmed) and (fresh_boundary or not disambiguated)
        if repeat or corroborated:
            self.offset = sign*offsets[best]
            self.boundary, self.boundary_stamp = boundary, stamp
        self.vote_side, self.vote_time = sign, stamp
