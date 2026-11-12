"""Metric point-to-line RGB-D registration with measured attitude.

White ground markings constrain lateral motion; vertical depth surfaces also
constrain forward motion. An unobservable axis is rejected, never filled using
commanded velocity. Only measured pixels, depth and attitude enter this module.
"""
import cv2
import numpy as np
from navigation.ground import GroundEstimator


class SurfaceOdometry:
    def __init__(self, rays, camera_position, camera_rotation):
        self.rays=np.asarray(rays,dtype=np.float32)
        self.position=camera_position
        self.rotation=camera_rotation
        # float32 copies keep the per-frame point cloud in single precision.
        self.position32=np.asarray(camera_position,dtype=np.float32)
        self.rotation32=np.asarray(camera_rotation,dtype=np.float32)
        self.support=0
        self.error=None
        self.ground=GroundEstimator()

    @staticmethod
    def sample(points):
        if not len(points):return np.empty((0,2),np.float32)
        # Pack the two quantized coordinates into one int64 key so the slow
        # 2-D np.unique(axis=0) row sort becomes a 1-D unique. Each coordinate
        # is shifted positive first, so negative values cannot collide, and
        # |coordinate| < 10 m keeps the shifted value inside 32 bits.
        q=np.round(points/.008).astype(np.int64)
        key=((q[:,0]+1_000_000)<<32)|(q[:,1]+1_000_000)
        _,indices=np.unique(key,return_index=True)
        points=points[indices]
        if len(points)>500:points=points[np.linspace(0,len(points)-1,500).astype(int)]
        return np.ascontiguousarray(points,dtype=np.float32)

    def project(self, frame, attitude):
        rgb,depth=frame.rgb,frame.depth
        body=(self.rays*depth[...,None])@self.rotation32.T+self.position32
        level=body@np.asarray(attitude,dtype=np.float32).T
        finite=np.isfinite(depth)&(depth>.15)&(depth<2.7)&(body[...,0]>.22)
        plane=self.ground.estimate(level,finite&(body[...,0]>.30),timestamp=frame.time)
        if not plane.valid:return [np.empty((0,2),np.float32)]*3
        height=plane.height(level)
        white=finite&(abs(height)<.025)&(rgb.min(axis=2)>115)&((rgb.max(axis=2).astype(float)-rgb.min(axis=2))<.15*rgb.max(axis=2))
        walls=finite&(height>.08)&(height<.32)
        # At close range the camera looks onto a box top and its vertical
        # front face leaves the image. The top/ground silhouette remains a
        # measured metric edge; image borders and missing-depth holes do not.
        floor=finite&(abs(height)<.025)
        neighboring_floor=cv2.dilate(floor.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        outline=finite&(height>.34)&(height<.46)&neighboring_floor
        # Keep near vertical faces and far top silhouettes separate: matching
        # one to the other would introduce a whole box-width localization jump.
        return [self.sample(level[mask,:2]) for mask in (white,walls,outline)]

    @staticmethod
    def surfels(points, neighbors=8):
        if len(points)<max(10,neighbors):return None
        distance,indices=cv2.batchDistance(points,points,cv2.CV_32F,normType=cv2.NORM_L2SQR,K=neighbors)
        neighborhood=points[indices]
        means=neighborhood.mean(axis=1)
        centered=neighborhood-means[:,None,:]
        covariance=np.einsum('nki,nkj->nij',centered,centered)/neighbors
        values,vectors=np.linalg.eigh(covariance)
        radius=.14 if neighbors==16 else .08
        good=(distance[:,-1]<radius**2)&(values[:,1]>2e-5)&(values[:,0]<.12*values[:,1])
        if good.sum()<8:return None
        return np.ascontiguousarray(means[good]),vectors[good,:,0]

    def register(self, reference, current, initial=None):
        self.support=0;self.error=None
        pairs=[]
        for index,(old,new) in enumerate(zip(reference,current)):
            # Smooth white-line normals over a longer metric segment. Pixel
            # edge noise on a straight marking must not look like curvature.
            neighbors=16 if index==0 else 8
            surfels=self.surfels(old,neighbors)
            measured=self.surfels(new,neighbors)
            if surfels is not None and measured is not None:pairs.append((*surfels,measured[0]))
        if not pairs:return None
        displacement=np.zeros(2) if initial is None else np.asarray(initial,dtype=float).copy()
        if displacement.shape!=(2,) or not np.isfinite(displacement).all() or np.linalg.norm(displacement)>.25:
            return None
        final=None
        for iteration in range(8):
            normals=[];errors=[]
            for points,directions,new in pairs:
                shifted=np.ascontiguousarray(new+displacement,dtype=np.float32)
                distance,indices=cv2.batchDistance(shifted,points,cv2.CV_32F,normType=cv2.NORM_L2SQR,K=1)
                indices=indices[:,0]
                n=directions[indices]
                residual=np.einsum('ij,ij->i',points[indices]-shifted,n)
                capture=.16 if iteration<2 else .10
                good=(distance[:,0]<capture**2)&(abs(residual)<capture*.8)
                normals.extend(n[good]);errors.extend(residual[good])
            if len(errors)<20:return None
            a=np.asarray(normals);b=np.asarray(errors)
            weight=np.minimum(1.,.008/np.maximum(abs(b),1e-9))
            h=a.T@(weight[:,None]*a)
            # Count geometric support before robust residual downweighting.
            # A recoverable 7 cm initial offset is not an unobservable axis.
            eigen=np.linalg.eigvalsh(a.T@a)
            if eigen[0]<.5 or eigen[0]<.0025*eigen[1]:return None
            weighted_eigen=np.linalg.eigvalsh(h)
            if weighted_eigen[0]<1e-6 or weighted_eigen[0]<1e-4*weighted_eigen[1]:return None
            delta=np.linalg.solve(h,a.T@(weight*b))
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>.10:return None
            displacement+=delta
            final=b
            if np.linalg.norm(delta)<.00015:break
        good=abs(final)<.012
        self.support=int(good.sum())
        self.error=float(np.sqrt(np.mean(final[good]**2))) if good.any() else None
        if good.mean()<.75 or self.error is None or self.error>.006 or np.linalg.norm(displacement)>.25:return None
        # Neighbouring surfels share samples. Account for that correlation
        # when accepting a weak but measured curved-line direction.
        uncertainty=max(.0015,self.error)*np.sqrt(16/eigen[0])
        if uncertainty>.012:return None
        return displacement

    def observable(self, clouds):
        normals=[]
        for index,cloud in enumerate(clouds):
            surfels=self.surfels(cloud,16 if index==0 else 8)
            if surfels is not None:normals.extend(surfels[1])
        if len(normals)<20:return False
        a=np.asarray(normals);eigen=np.linalg.eigvalsh(a.T@a)
        return bool(eigen[0]>=.5 and eigen[0]>=.0025*eigen[1])
