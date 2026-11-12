"""Color-independent, gravity-constrained ground estimation from metric depth.

Input coordinates have measured gravity aligned with +Z; the robot body origin
must be above its support floor. Cold-start evidence must span more than one
20 cm obstacle top. The estimator never accepts a stale plane as a fresh
measurement and does not infer a floor hidden behind an occluder.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class GroundPlane:
    valid: bool
    normal: np.ndarray
    offset: float
    z0: float
    support: int
    area: float
    residual: float
    reason: str

    @classmethod
    def invalid(cls, reason):
        return cls(False,np.array([0.,0.,1.]),float('nan'),float('nan'),0,0.,float('nan'),reason)

    def height(self, points):
        """Signed perpendicular height above this measured plane, in metres."""
        return np.asarray(points)@self.normal+self.offset

    def diagnostic(self):
        return dict(valid=self.valid,reason=self.reason,z_at_body_origin_m=self.z0,
                    normal=self.normal.tolist(),support=self.support,
                    connected_support_area_m2=self.area,residual_rms_m=self.residual)


class GroundEstimator:
    """Select the lowest supported near-horizontal plane, then verify history.

    Box colors are deliberately absent from the interface. Spatial support is
    counted in metric cells, so a close 20 cm top cannot win by filling pixels.
    """
    def __init__(self):
        self.last=None
        self.last_time=None
        self.result=GroundPlane.invalid('initializing')

    @staticmethod
    def _horizontal_mask(points, valid):
        if points.ndim!=3 or min(points.shape[:2])<9:return valid.copy()
        step=3
        dx=points[step:-step,2*step:]-points[step:-step,:-2*step]
        dy=points[2*step:,step:-step]-points[:-2*step,step:-step]
        normals=np.cross(dx,dy)
        size=np.linalg.norm(normals,axis=2)
        # Noise near depth discontinuities is rejected before plane fitting.
        keep=(size>1e-8)&(abs(normals[...,2])>.90*size)
        keep&=valid[step:-step,2*step:]&valid[step:-step,:-2*step]&valid[2*step:,step:-step]&valid[:-2*step,step:-step]
        result=np.zeros(valid.shape,bool)
        result[step:-step,step:-step]=keep
        return result&valid

    @staticmethod
    def _support(points):
        if len(points)<30:return 0.,np.zeros(2)
        cell=.025
        lo=points[:,:2].min(axis=0)
        cells=np.floor((points[:,:2]-lo)/cell).astype(np.int32)
        dimensions=cells.max(axis=0)+1
        if np.any(dimensions>500):return 0.,np.zeros(2)
        grid=np.zeros((dimensions[1]+2,dimensions[0]+2),np.uint8)
        grid[cells[:,1]+1,cells[:,0]+1]=1
        connected=cv2.dilate(grid,np.ones((3,3),np.uint8))
        _,labels=cv2.connectedComponents(connected)
        # Count occupied measured cells, not the dilated cells used to bridge
        # isolated depth holes. Disconnected box tops remain separate islands.
        ids=labels[grid.astype(bool)]
        counts=np.bincount(ids)
        if len(counts)<=1:return 0.,np.zeros(2)
        counts[0]=0;largest=int(np.argmax(counts))
        members=labels[cells[:,1]+1,cells[:,0]+1]==largest
        span=np.ptp(points[members,:2],axis=0)
        return float(counts[largest]*cell*cell),np.sort(span)

    def estimate(self, level, valid=None, timestamp=None):
        points=np.asarray(level,dtype=float)
        if points.ndim not in (2,3) or points.shape[-1]!=3:
            raise ValueError('Expected N x 3 or H x W x 3 gravity-aligned depth points')
        finite=np.isfinite(points).all(axis=-1)
        if valid is not None:
            mask=np.asarray(valid,dtype=bool)
            if mask.shape!=finite.shape:raise ValueError('Ground validity mask shape differs')
            finite&=mask
        # This is an observation range bound, not a prescribed standing height.
        finite&=(np.linalg.norm(points[...,:2],axis=-1)<4.)&(abs(points[...,2])<2.)
        # A trusted recent plane lets consecutive frames skip both the horizontal
        # neighbour scan and the height-bin search: the floor cannot jump far
        # between frames, so only its neighbourhood needs to be refitted. The
        # fitted normal gate below still rejects anything but a near-horizontal
        # surface, and any failure falls through to the unchanged full search.
        if self.last is not None and timestamp is not None and self.last_time is not None and 0<=timestamp-self.last_time<=2.:
            near=points[finite&(abs(points@self.last.normal+self.last.offset)<.03)]
            if len(near)>12000:near=near[np.linspace(0,len(near)-1,12000).astype(int)]
            if len(near)>=50:
                matrix=np.column_stack([near[:,:2],np.ones(len(near))])
                coefficients,_,rank,_=np.linalg.lstsq(matrix,near[:,2],rcond=None)
                if rank==3:
                    normal=np.r_[-coefficients[:2],1.];normal/=np.linalg.norm(normal)
                    offset=-coefficients[2]*normal[2]
                    rms=float(np.sqrt(np.mean((near@normal+offset)**2)))
                    if (np.linalg.norm(coefficients[:2])<=np.tan(np.deg2rad(12)) and abs(coefficients[2]-self.last.z0)<.08
                            and coefficients[2]<-.03 and rms<=.005):
                        area,span=self._support(near)
                        if area>=.025 and span[1]>=.18 and span[0]>=.055:
                            result=GroundPlane(True,normal,float(offset),float(coefficients[2]),
                                               int(len(near)),area,rms,'measured_depth_plane')
                            self.last=result;self.last_time=timestamp;self.result=result
                            return result
        horizontal=self._horizontal_mask(points,finite)
        cloud=points[horizontal]
        if len(cloud)<60:return self._invalid('insufficient_horizontal_depth')
        if len(cloud)>12000:cloud=cloud[np.linspace(0,len(cloud)-1,12000).astype(int)]
        bins=np.floor(cloud[:,2]/.012).astype(np.int32)
        values,counts=np.unique(bins,return_counts=True)
        # Nearby occupied height bins bootstrap a fit; extend and refit against
        # the full near-horizontal cloud so modest IMU tilt does not bias z0.
        candidates=[]
        for index,count in zip(values,counts):
            if count<25:continue
            seed=abs(cloud[:,2]-(index+.5)*.012)<.024
            if seed.sum()<50:continue
            subset=cloud[seed]
            matrix=np.column_stack([subset[:,:2],np.ones(len(subset))])
            coefficients,_,rank,_=np.linalg.lstsq(matrix,subset[:,2],rcond=None)
            if rank<3:continue
            selected=seed
            for _ in range(3):
                if np.linalg.norm(coefficients[:2])>np.tan(np.deg2rad(12)):break
                residual=cloud[:,2]-cloud[:,:2]@coefficients[:2]-coefficients[2]
                selected=abs(residual)<.010
                if selected.sum()<50:break
                subset=cloud[selected]
                matrix=np.column_stack([subset[:,:2],np.ones(len(subset))])
                coefficients,_,rank,_=np.linalg.lstsq(matrix,subset[:,2],rcond=None)
            if rank<3 or selected.sum()<50 or np.linalg.norm(coefficients[:2])>np.tan(np.deg2rad(12)):continue
            # A support floor is below the body reference point. No known
            # simulator base height or carpet elevation is used here.
            if coefficients[2]>=-.03:continue
            subset=cloud[selected]
            area,span=self._support(subset)
            trusted=self.last is not None and (timestamp is None or self.last_time is None or 0<=timestamp-self.last_time<=2.)
            compatible=trusted and abs(coefficients[2]-self.last.z0)<.08
            min_area=.025 if compatible else .065
            min_span=.18 if compatible else .30
            if area<min_area or span[1]<min_span or span[0]<.055:continue
            if trusted and not compatible:continue
            normal=np.r_[-coefficients[:2],1.];normal/=np.linalg.norm(normal)
            offset=-coefficients[2]*normal[2]
            error=subset@normal+offset
            rms=float(np.sqrt(np.mean(error**2)))
            if rms>.005:continue
            candidates.append(GroundPlane(True,normal,float(offset),float(coefficients[2]),
                              int(len(subset)),area,rms,'measured_depth_plane'))
        if not candidates:return self._invalid('no_supported_ground_plane')
        # The lower support surface is the floor even when a raised top is
        # closer to the camera, larger in pixel area, or the same green color.
        result=min(candidates,key=lambda plane:plane.z0)
        self.last=result;self.last_time=timestamp;self.result=result
        return result

    def _invalid(self, reason):
        self.result=GroundPlane.invalid(reason)
        return self.result
