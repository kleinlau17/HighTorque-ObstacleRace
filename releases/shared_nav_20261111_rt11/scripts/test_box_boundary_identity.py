"""The box controller receives only currently measured, identified ground stripes."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation.box_corridor import BoundaryTracks, Edge, floor_edges, measured_pass_side


def stripe(offset,slope=0.,near=.30,far=.66):
    return Edge(offset,slope,near,far)


class BoundaryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tracker=BoundaryTracks()
        self.origin=np.zeros(2)

    def test_oblique_single_stripe_identifies_box_side_in_metric_normal(self):
        m=.55
        near=stripe(.34,m)
        bx=.79
        box=(bx,near.at(bx)-.10*np.hypot(1.,m))
        self.assertEqual(measured_pass_side([near],box),-1.)
        self.assertEqual(self.tracker.observe([near],1.,self.origin,[(-1,box)]),(None,near))
        opposite=stripe(near.offset-.50*np.hypot(1.,m),m)
        self.assertEqual(measured_pass_side([opposite],box),-1.)
        self.assertEqual(self.tracker.observe([opposite],1.2,self.origin,[]),(opposite,None))

    def test_one_side_across_occlusion_switch_does_not_publish_history(self):
        right,left=stripe(-.25),stripe(.25)
        self.assertEqual(self.tracker.observe([left,right],1.,self.origin,[]),(right,left))
        self.assertEqual(self.tracker.observe([],1.2,self.origin,[]),(None,None))
        # Robot has moved outside the right paint. Its positive intercept must
        # retain the right identity even though the next crate is on the left.
        at=np.array([.07,-.28])
        current=stripe(.03)
        self.assertEqual(self.tracker.observe([current],1.4,at,[]),(current,None))
        next_left=stripe(.53)
        self.assertEqual(self.tracker.observe([next_left],1.6,at,[]),(None,next_left))
        self.assertEqual(self.tracker.observe([current],3.4,at,[]),(None,None))

    def test_unrelated_or_ambiguous_paint_cannot_open_a_side(self):
        self.assertEqual(self.tracker.observe([stripe(-.24)],0.,self.origin,[]),(None,None))
        self.assertEqual(self.tracker.observe([stripe(-.74),stripe(-.24),stripe(.26)],
                                               1.,self.origin,[]),(None,None))
        self.assertEqual(self.tracker.observe([stripe(-.25),stripe(.25)],2.,self.origin,[]),
                         (stripe(-.25),stripe(.25)))
        self.assertEqual(self.tracker.observe([stripe(.02)],2.2,self.origin,[]),(None,None))

    def test_distant_fragment_needs_present_paint_and_recent_near_tangent(self):
        far=stripe(-.25,.015,.99,1.30)
        self.assertEqual(self.tracker.observe([far],.1,self.origin,[]),(None,None))
        self.tracker.observe([stripe(-.25),stripe(.25)],1.,self.origin,[])
        carried=self.tracker.observe([far],1.2,self.origin,[])
        self.assertAlmostEqual(carried[0].at(1.14),far.at(1.14),places=3)
        self.assertIsNone(carried[1])
        self.assertEqual(self.tracker.observe([stripe(-.4,.17,1.,1.32)],1.4,self.origin,[]),
                         (None,None))
        self.assertEqual(self.tracker.observe([far],3.5,self.origin,[]),(None,None))

    def test_short_current_paint_is_measured_with_a_registered_tangent(self):
        x,y=np.meshgrid(np.linspace(.24,1.,181),np.linspace(-.49,.49,160))
        xyz=np.stack([x,y,np.zeros_like(x)],axis=2)
        rgb=np.full((*x.shape,3),30,np.uint8)
        rgb[(x>.49)&(x<.61)&(abs(y-.22)<.009)]=225
        edges=floor_edges(rgb,xyz,np.zeros_like(x),np.ones_like(x,dtype=bool))
        self.assertEqual(len(edges),1)
        self.assertLess(edges[0].far-edges[0].near,.14)
        self.assertEqual(self.tracker.observe(edges,1.,self.origin,[]),(None,None))
        self.tracker.observe([stripe(-.25),stripe(.25)],1.2,self.origin,[])
        right,left=self.tracker.observe(edges,1.4,np.array([0.,.03]),[])
        self.assertIsNone(right)
        self.assertAlmostEqual(left.at(.55),.22,delta=.006)
        self.assertEqual(self.tracker.observe([],1.6,self.origin,[]),(None,None))
        self.assertEqual(self.tracker.observe([stripe(.22,.35,.50,.60)],1.8,self.origin,[]),
                         (None,None))
        self.assertEqual(self.tracker.observe(edges,10.,self.origin,[]),(None,None))

    def test_current_long_and_clipped_pair_needs_live_crate_geometry(self):
        from navigation.box_corridor import clipped_paint_pair
        near=stripe(.202,.010,.595,.652)
        far=stripe(-.296,.008,.920,1.272)
        # Long and short intervals do not overlap. Neither the far paint nor
        # the unregistered short piece alone verifies both identities.
        self.assertEqual(self.tracker.observe([far,near],1.,self.origin,[]),(None,None))
        pair=clipped_paint_pair([far,near],[(1.,(.50,-.19))])
        self.assertAlmostEqual(pair[0].at(1.),far.at(1.),delta=.006)
        self.assertAlmostEqual(pair[1].at(.623),near.at(.623),delta=.001)
        self.assertEqual(clipped_paint_pair([far,near],[(1.,(.50,-.05))]),(None,None))
        self.assertEqual(clipped_paint_pair([far,stripe(.202,.20,.595,.652)],
                                       [(1.,(.50,-.19))]),(None,None))

    def test_optimizer_ignores_white_obstacle_tops_and_transverse_tape(self):
        x,y=np.meshgrid(np.linspace(.25,1.30,128),np.linspace(-.50,.50,100))
        xyz=np.stack([x,y,np.zeros_like(x)],axis=2)
        valid=np.ones_like(x,dtype=bool)
        rgb=np.full((*x.shape,3),30,np.uint8)
        rgb[(abs(y+.25)<.014)|(abs(x-.85)<.012)]=225
        height=np.zeros_like(x)
        height[abs(y+.25)<.014]=.22
        self.assertEqual(floor_edges(rgb,xyz,height,valid),[])

    def test_blue_cast_tape_requires_floor_contrast_on_both_sides(self):
        # RGB values sampled from the Jetson's right stripe and the adjacent mat.
        y,x=np.meshgrid(np.linspace(.51,-.51,256),np.linspace(.95,.35,192))
        xyz=np.stack([x,y,np.zeros_like(x)],axis=2)
        height=np.zeros_like(x)
        valid=np.ones_like(x,dtype=bool)
        tape=(abs(y+.22)<.009)&(x<.78)
        mat=(23,114,124)
        pale_blue=(170,191,240)
        rgb=np.empty((*x.shape,3),np.uint8)
        rgb[:]=mat
        rgb[tape]=pale_blue
        # The same stripe in the real aligned pose retained its far color
        # while the near segment lost red. The forward check needs that near
        # segment to locate the edge alongside the swept body.
        rgb[tape & (x<.64)]=(131,182,240)
        lines=floor_edges(rgb,xyz,height,valid)
        right,left=self.tracker.observe(lines,1.,self.origin,[(-1.,(.66,.18))])
        self.assertIsNone(left)
        self.assertAlmostEqual(right.offset,-.22,delta=.006)
        self.assertLess(right.near,.50)
        for name,mask,color in [
            ('wide bright patch',abs(y+.22)<.08,pale_blue),
            ('colored floor transition',y<-.22,pale_blue),
            ('saturated blue marking',tape,(62,126,246)),
            ('round reflection',((y+.22)**2+(x-.65)**2)<.025**2,pale_blue),
        ]:
            with self.subTest(name=name):
                rgb[:]=mat
                rgb[mask]=color
                self.assertEqual(floor_edges(rgb,xyz,height,valid),[])
        rgb[:]=mat
        rgb[tape]=pale_blue
        height[tape]=.20
        self.assertEqual(floor_edges(rgb,xyz,height,valid),[])
        height[:]=0.
        valid[y<-.229]=False
        self.assertEqual(floor_edges(rgb,xyz,height,valid),[])


if __name__=='__main__':
    unittest.main()
