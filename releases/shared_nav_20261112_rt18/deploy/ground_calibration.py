"""Fit a green floor plane; this cannot identify full camera-to-body extrinsics."""
import numpy as np


def fit_ground(points, seed=21, iterations=180):
    if not isinstance(iterations,int) or not 20<=iterations<=1000:raise ValueError('Invalid RANSAC iteration count')
    p=np.asarray(points,dtype=float)
    p=p[np.isfinite(p).all(axis=1)]
    if len(p)<300: raise ValueError('Insufficient green-floor depth points')
    rng=np.random.default_rng(seed)
    if len(p)>4000:p=p[rng.choice(len(p),4000,replace=False)]
    best=None
    for _ in range(iterations):
        a,b,c=p[rng.choice(len(p),3,replace=False)]
        normal=np.cross(b-a,c-a)
        length=np.linalg.norm(normal)
        if length<1e-7:continue
        normal/=length
        mask=abs(np.einsum('ij,j->i',p,normal)-normal@a)<.03
        if best is None or mask.sum()>best.sum():best=mask
    if best is None or best.mean()<.70:raise ValueError('No dominant plane in green-floor samples')
    for _ in range(3):
        q=p[best];center=q.mean(axis=0)
        _,_,vh=np.linalg.svd(q-center,full_matrices=False)
        normal=vh[-1];d=-normal@center
        if d<0:normal=-normal;d=-d
        signed=np.einsum('ij,j->i',p,normal)+d
        residual=signed[best]
        sigma=1.4826*np.median(abs(residual-np.median(residual)))
        tolerance=float(np.clip(3*sigma,.008,.03))
        best=abs(signed)<tolerance
    if best.mean()<.70:raise ValueError('Floor plane refinement lost consensus')
    forward=np.array([0.,0.,1.]);forward-=normal*(normal@forward)
    if np.linalg.norm(forward)<.15:raise ValueError('Camera points nearly straight down; forward direction is poorly defined')
    forward/=np.linalg.norm(forward)
    left=np.cross(normal,forward)
    rotation=np.stack([forward,left,normal])
    inliers=p[best]
    spread=np.linalg.svd(inliers-inliers.mean(axis=0),compute_uv=False)/np.sqrt(len(inliers))
    if spread[1]<.07:raise ValueError('Floor observations are too narrow for a stable plane')
    if not .10<d<3.:raise ValueError('Implausible camera height')
    rms=float(np.sqrt(np.mean((np.einsum('ij,j->i',inliers,normal)+d)**2)))
    if rms>.015:raise ValueError('Floor depth residual exceeds 15 mm')
    return dict(normal_camera=normal.tolist(),camera_height_m=float(d),
        camera_to_ground_rotation=rotation.tolist(),camera_position_ground_m=[0.,0.,float(d)],
        downward_angle_deg=float(np.degrees(np.arcsin(np.clip(-normal[2],-1,1)))),
        inlier_ratio=float(best.mean()),residual_rms_m=rms,inlier_distance_m=tolerance,
        points=len(p),frame='ground_below_camera; X is projected camera optical forward',
        body_extrinsics_verified=False,
        unobservable=['camera/body longitudinal and lateral offsets','camera/body yaw','head joint kinematic chain','IMU/body alignment'])
