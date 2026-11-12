"""Geometry observability and fixed-input recovery regressions; no simulator."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.surface_odometry import SurfaceOdometry
from navigation.runtime import CameraFrame


class SurfaceTests(unittest.TestCase):
    def cloud(self):
        t=np.linspace(-.4,.4,120)
        return [np.c_[t+.7,np.full_like(t,.25)].astype('float32'),
                np.c_[np.full_like(t,.8),t].astype('float32')]

    def test_two_surface_directions_recover_metric_translation(self):
        ref=self.cloud();shift=np.array([.045,-.025])
        rng=np.random.default_rng(9)
        current=[np.asarray(points-shift+rng.normal(0,.0005,points.shape),dtype='float32') for points in ref]
        estimator=SurfaceOdometry(None,None,None)
        np.testing.assert_allclose(estimator.register(ref,current),shift,atol=.002)

    def test_parallel_lines_do_not_invent_forward_motion(self):
        ref=self.cloud()[:1];current=[ref[0]-np.array([.04,.02],dtype='float32')]
        self.assertIsNone(SurfaceOdometry(None,None,None).register(ref,current))

    def test_visible_top_silhouette_is_kept_when_no_vertical_face_is_visible(self):
        u,v=np.meshgrid(np.arange(64),np.arange(48))
        rays=np.stack([u/60,v/60,np.ones_like(u)],-1)
        estimator=SurfaceOdometry(rays,np.array([.4,0,0]),np.diag([1,1,-1.]))
        rgb=np.zeros((48,64,3),np.uint8);rgb[:]=[19,125,78]
        depth=np.ones((48,64));depth[12:32,18:42]=.6;rgb[12:32,18:42]=[210,205,180]
        frame=CameraFrame(0.,rgb,depth)
        clouds=estimator.project(frame,np.eye(3))
        self.assertGreater(len(clouds[2]),20)
        # The interior of a uniform top is not a horizontal position feature.
        self.assertFalse(np.any(np.all(abs(clouds[2]-[.7,.22])<.025,axis=1)))

    def test_fixed_keyframe_does_not_accumulate_stationary_edge_noise(self):
        ref=self.cloud();estimator=SurfaceOdometry(None,None,None)
        rng=np.random.default_rng(5);estimates=[]
        for _ in range(20):
            current=[np.asarray(points+rng.normal(0,.0005,points.shape),dtype='float32') for points in ref]
            estimates.append(estimator.register(ref,current))
        self.assertLess(np.max(np.linalg.norm(estimates,axis=1)),.002)


if __name__=='__main__':unittest.main()
