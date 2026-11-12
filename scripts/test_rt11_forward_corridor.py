"""Regression for RT11's optical clearance using calibrated noisy MuJoCo depth and archived RGB-D.

Give a boxes_center or boxes_oblique run directory. The default frame tests the clear route at 4 s.
"""
import json
import sys
import zipfile
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'releases/shared_nav_20261111_rt11'
sys.path[:0] = [str(RELEASE), str(ROOT)]

from deploy.geometry import quaternion_matrix_xyzw
from navigation.box_body import limb_points
from navigation.box_corridor import BoundaryTracks, floor_edges, forward_clearance
from navigation.box_servo import rot
from navigation.runtime import CameraFrame
from navigation.visual import CameraPerception
from sim.sensors import SensorRig


def measure(frame, camera, attitude, basis, body, anchors=()):
    camera.inspect_lane(frame, attitude)
    plane = camera.ground.result
    level = ((camera.rays * frame.depth[...,None]) @ camera.rotation32.T + camera.position32) @ attitude.T
    height = plane.height(level)
    points = level.copy()
    points[...,:2] = level[...,:2] @ basis
    valid = np.isfinite(frame.depth) & (frame.depth > .12) & (frame.depth < 4.)
    args = (points, height, valid, body, frame.depth, camera, attitude, basis, plane)
    lines = floor_edges(frame.rgb, points, height, valid)
    bounds = BoundaryTracks().observe(lines, frame.time, np.zeros(2), anchors)
    return args, bounds


def load_rows(folder):
    return [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]


class Fixture:
    def __init__(self, folder, stamp):
        self.folder = folder
        cfg = json.loads((RELEASE/'deploy/pro_model_config.json').read_text())
        profile = json.loads((folder/'setup.json').read_text())['camera']
        self.rows = load_rows(folder)
        self.model = mj.MjModel.from_xml_path(str(folder/'scene.xml'))
        self.data = mj.MjData(self.model)
        state = np.load(folder/'states.npz')
        j = np.argmin(abs(state['t'] - stamp))
        self.data.qpos[:] = state['qpos'][j]
        self.data.time = state['t'][j]
        mj.mj_forward(self.model, self.data)
        self.rig = SensorRig(self.model, seed=1, noise=True, camera_profile=profile)
        self.camera = CameraPerception(self.rig.width, self.rig.height, 45., cfg['camera_position_body_m'],
                                       cfg['camera_rotation_optical_to_body'], self.rig.intrinsics)
        q = self.rig.proprioception(self.data).orientation_wxyz
        self.attitude = quaternion_matrix_xyzw(q[[1,2,3,0]]) @ np.asarray(cfg['body_from_imu']).T
        row = min((r for r in self.rows if r.get('crate_heading') is not None),key=lambda r:abs(r['t']-stamp))
        self.basis = rot(row['crate_heading'])
        positions = {self.model.joint(i).name:self.data.qpos[self.model.jnt_qposadr[i]]
                     for i in range(1,self.model.njnt)}
        self.body = limb_points(positions,self.attitude)
        self.ids = [self.model.body(f'hb_obstacle_0{i}').id for i in (1,2,3)]
        self.trunk = self.data.xpos[self.model.body('base_link').id,:2].copy()

    def render(self, box=None):
        # Geometry is used to inject a fault into the renderer only.
        for i,b in enumerate(self.ids):
            self.model.body_pos[b,:2] = (20.+i,20.)
        if box is not None:
            self.model.body_pos[self.ids[0],:2] = self.trunk + np.array(box) @ self.basis.T
        mj.mj_forward(self.model,self.data)
        frame = self.rig.camera(self.data)
        return measure(frame,self.camera,self.attitude,self.basis,self.body)

    def recorded(self):
        with zipfile.ZipFile(self.folder/'camera_rgbd.zip') as archive:
            frames = np.asarray(json.loads(archive.read('index.json'))['frames'])
            n = int(np.argmin(abs(frames[:,0] - self.data.time)))
            def read(path):
                return cv2.imdecode(np.frombuffer(archive.read(path),np.uint8),cv2.IMREAD_UNCHANGED)
            rgb = read(f'rgb/{n:05}.png')[:,:,::-1]
            depth = read(f'depth_mm/{n:05}.png').astype(np.float32)*.001
            depth[depth <= 0] = np.nan
        t = frames[n,1]
        row = min((r for r in self.rows if r.get('crate_heading') is not None),
                  key=lambda r:abs(r['tracking_frame_time']-t))
        # These are onboard camera associations, also used by the real caller.
        anchors = [(row['crate_side']*(-1 if i%2 else 1),np.array(row['crate_centers'][i])-row['crate_pose'])
                   for i in row['crate_visible']]
        self.anchors = anchors
        return measure(CameraFrame(t,rgb,depth),self.camera,self.attitude,rot(row['crate_heading']),self.body,anchors)


def trials(folder, stamp):
    fixture = Fixture(folder,stamp)
    rows = []
    def check(name, args, bounds, clear, anchors=()):
        room, status = forward_clearance(*args,bounds,.063,anchors)
        rows.append(dict(case=folder.name, scene=name, space_m=room, depth=status))
        assert (room >= .05) == clear, rows[-1]
    try:
        args, bounds = fixture.recorded()
        check('recorded_original',args,bounds,True,fixture.anchors)
        args, bounds = fixture.render()
        check('empty',args,bounds,True)
        assert forward_clearance(*args,bounds,.010)==(.010,'depth_clear')
        # A live boundary does not replace range. Keep the original camera and
        # floor fit, but remove the central 70% of current depth returns.
        depth = args[4].copy()
        w = depth.shape[1]
        depth[:,int(w*.15):int(w*.85)] = np.nan
        changed = list(args)
        changed[2] = np.isfinite(depth)
        changed[4] = depth
        check('center_depth_masked',changed,bounds,False)
        changed[2] = np.zeros_like(changed[2])
        changed[4] = np.full_like(depth,np.nan)
        check('depth_missing',changed,bounds,False)
        check('encoder_missing',args[:3]+(None,)+args[4:],bounds,False)
        for name, position in [('frontal_crate',(.24,0.)), ('left_fringe_crate',(.30,.21)),
                               ('right_fringe_crate',(.25,-.22)), ('straddling_stripe',(.26,-.27))]:
            scene_args, scene_bounds = fixture.render(position)
            check(name,scene_args,scene_bounds,False)
            if name=='frontal_crate':
                check('foreground_with_false_id',scene_args,scene_bounds,False,[(1.,(.67,.34))])
    finally:
        fixture.rig.close()
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='A completed RGB-D box simulation')
    parser.add_argument('--time', type=float, default=4., help='Simulated seconds at the clear starting pose')
    options = parser.parse_args()
    print(json.dumps(trials(options.run.resolve(), options.time), indent=2))
