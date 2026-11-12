"""Run a closed-loop MuJoCo box-entry test with timed depth and pose faults.

Times are absolute seconds on the simulation clock, including the 2 s settle.
Each invocation takes a new output directory. The retained RT11 harness records
all actual RGB-D inputs, plans, guard outputs, policy inputs, and MuJoCo states.
"""
import argparse
import csv
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARNESS = ROOT/'output/deployment/rt11_verified_20261111/run_validation.py'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='candidate root containing navigation/ and deploy/')
    parser.add_argument('--out', type=Path, required=True, help='new run directory')
    parser.add_argument('--mode', choices=('both', 'depth', 'freeze', 'none'), default='both')
    parser.add_argument('--start', type=float, default=3.80, help='absolute simulation seconds')
    parser.add_argument('--restore', default='never', help='absolute seconds, or never for a permanent fault')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timing', type=Path, help='recorded rt10 navigation directory')
    parser.add_argument('--case', default='boxes_center')
    parser.add_argument('--limit', type=float, help='maximum simulation seconds after the 2 s settle')
    parser.add_argument('--mirror', action='store_true', help='reflect boxes and box-area spawn across the lane center')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--harness', type=Path, default=DEFAULT_HARNESS)
    args = parser.parse_args()
    if args.restore in ('never','permanent','inf'):
        args.restore = None
    else:
        try:
            args.restore = float(args.restore)
        except ValueError:
            parser.error('--restore requires an absolute time or never')
        if not math.isfinite(args.restore) or args.restore <= args.start:
            parser.error('--restore must be later than --start')
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('--start must be finite and nonnegative')
    if args.limit is not None and (not math.isfinite(args.limit) or args.limit <= 0):
        parser.error('--limit must be finite and positive')
    for required in ('navigation/box_servo.py','deploy/shared_navigation.py'):
        if not (args.source/required).is_file():
            parser.error(f'candidate source is missing {required}')
    return args


def load_harness(args):
    file = args.harness.resolve()
    sys.path.insert(0, str(file.parent))
    sys.argv = [str(file), args.case, '--source', str(args.source.resolve()), '--out', str(args.out.resolve()),
                '--seed', str(args.seed)]
    if args.timing is not None:
        sys.argv.extend(('--timing', str(args.timing.resolve())))
    if args.limit is not None:
        sys.argv.extend(('--limit', str(args.limit)))
    if args.gui:
        sys.argv.append('--gui')
    spec = importlib.util.spec_from_file_location('rt_entry_harness', file)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    if args.case not in harness.cases:
        raise ValueError(f'Unknown case {args.case}. Available: {", ".join(harness.cases)}')
    return harness


def mirror_scene(harness, case, overlay):
    # The archived harness reads the immutable device audit and two MJCF files
    # under BASE. Overlay them in a temporary directory, retaining its own
    # scene.xml copy in the output as usual.
    base = harness.BASE
    view = Path(overlay)
    (view/'device_import_audit.json').symlink_to(base/'device_import_audit.json')
    (view/'results/full_01').mkdir(parents=True)
    (view/'results/small_entry').symlink_to(base/'results/small_entry', target_is_directory=True)
    track = harness.Track('corner')
    lane_center = track.config['small_radius'] + track.config['track_width']/2
    scene = ElementTree.parse(base/'results/full_01/scene.xml')
    mirrored = {}
    for box in track.config['obstacles']:
        body = scene.find(f".//body[@name='hb_obstacle_{box['id']}']")
        x,y,z = map(float,body.get('pos').split())
        assert np.allclose((x,y),box['position'])
        y = round(2*lane_center-y,9)
        body.set('pos',f'{x} {y} {z}')
        mirrored[box['id']] = [x,y]
    scene.write(view/'results/full_01/scene.xml',encoding='unicode')

    class MirroredTrack(harness.Track):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for box in self.config['obstacles']:
                box['position'] = mirrored[box['id']].copy()
                box['edge'] = {'outer':'inner','inner':'outer'}[box['edge']]
            self.obstacles = np.array([box['position'] for box in self.config['obstacles']])

    if case.startswith('boxes'):
        x,y,angle = harness.cases[case]
        harness.cases[case] = [x,2*lane_center-y,2*np.pi-angle]
    harness.Track = MirroredTrack
    harness.BASE = view


class Fault:
    def __init__(self, args, log):
        self.start = args.start
        self.restore = args.restore
        self.log = csv.writer(log)
        self.log.writerow(['t','phase','capture_stamp','raw_box_y_m','input_box_y_m','input_box_vy_m_s','entry_hold'])
        self.stats = dict(masked_frames=0,masked_pixels=0,frozen_command_calls=0,
                          frozen_calls_by_phase={},frozen_calls_during_hold=0,first_frozen_command_s=None,last_frozen_command_s=None,
                          frozen_y_m=None)

    def active(self, stamp):
        return stamp >= self.start and (self.restore is None or stamp < self.restore)

    def install_depth_loss(self, harness):
        fault = self
        class CrateDepthLoss(harness.SensorRig):
            def __init__(self, model, *args, **kwargs):
                super().__init__(model,*args,**kwargs)
                # Some regression runs intentionally start on a box-free track.
                target = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hb_obstacle_03')
                self.target_geoms = np.flatnonzero(model.geom_bodyid == target) if target >= 0 else []

            def camera(self, data):
                frame = super().camera(data)
                if not fault.active(frame.time) or not len(self.target_geoms):
                    return frame
                self.renderer.enable_segmentation_rendering()
                try:
                    segmentation = self.renderer.render()
                finally:
                    self.renderer.disable_segmentation_rendering()
                mask = ((segmentation[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                        np.isin(segmentation[...,0],self.target_geoms)).astype(np.uint8)
                mask = cv2.remap(mask,*self.rectify_maps,cv2.INTER_NEAREST).astype(bool)
                count = int(np.count_nonzero(mask))
                if not count:
                    return frame
                fault.stats['masked_frames'] += 1
                fault.stats['masked_pixels'] += count
                depth = frame.depth.copy()
                depth[mask] = np.nan
                return replace(frame,depth=depth)

        harness.SensorRig = CrateDepthLoss

    def install_pose_freeze(self):
        # This module was loaded from --out/source by the archived harness.
        from navigation.box_servo import BoxServo, rot
        original_command = BoxServo._command
        fault = self
        def frozen_command(servo, t, yaw, frame):
            if frame is None or not fault.active(t):
                return original_command(servo,t,yaw,frame)
            if fault.stats['frozen_y_m'] is None and (not frame.centers or not frame.side):
                return original_command(servo,t,yaw,frame)
            basis = rot(frame.heading)
            leg = servo.latest_leg
            anchor = frame.leg_anchor
            fresh_leg = anchor is not None and leg is not None and 0 <= t-leg[0] <= .10
            if fresh_leg:
                delta = (leg[1]-anchor) @ basis
                vel = leg[2] @ basis
                raw_y = frame.position[1]+delta[1]+vel[1]*(t-leg[0])
            else:
                raw_y = frame.position[1]+frame.velocity[1]*min(.30,max(0.,t-frame.stamp))
            stats = fault.stats
            if stats['frozen_y_m'] is None:
                stats['frozen_y_m'] = float(raw_y)
                stats['first_frozen_command_s'] = t
            stats['last_frozen_command_s'] = t
            stats['frozen_command_calls'] += 1
            phases = stats['frozen_calls_by_phase']
            phases[servo.phase] = phases.get(servo.phase,0)+1
            pos,vel = frame.position.copy(),frame.velocity.copy()
            pos[1],vel[1] = stats['frozen_y_m'],0.
            if fresh_leg:
                # Retain the source timestamp, sample freshness and longitudinal
                # interpolation; suppress only the lateral anchor delta and speed.
                anchor = anchor+delta[1]*basis[:,1]
                servo.latest_leg = (leg[0],leg[1],leg[2]-np.dot(leg[2],basis[:,1])*basis[:,1])
            manipulated = replace(frame,position=pos,velocity=vel,leg_anchor=anchor)
            hold = getattr(servo,'entry_hold',None)
            if hold or 'hold' in servo.phase:
                stats['frozen_calls_during_hold'] += 1
            fault.log.writerow((t,servo.phase,frame.stamp,raw_y,pos[1],vel[1],hold))
            try:
                return original_command(servo,t,yaw,manipulated)
            finally:
                if fresh_leg:
                    servo.latest_leg = leg
        BoxServo._command = frozen_command


def main():
    args = parse_args()
    harness = load_harness(args)
    target = args.out.resolve()
    source = args.source.resolve()
    meta = dict(mode=args.mode,start_sim_s=args.start,restore_sim_s=args.restore,
                mirror=args.mirror,case=args.case,seed=args.seed,source=str(source),
                timing=str(harness.RUN),harness=str(args.harness.resolve()),
                harness_sha256=harness.HASH(args.harness),injector_sha256=harness.HASH(Path(__file__)),
                box_servo_sha256=harness.HASH(target/'source/navigation/box_servo.py'))
    with (target/'fault_input.csv').open('w',newline='') as log, tempfile.TemporaryDirectory() as overlay:
        fault = Fault(args,log)
        if args.mirror:
            mirror_scene(harness,args.case,overlay)
        if args.mode in ('depth','both'):
            fault.install_depth_loss(harness)
        if args.mode in ('freeze','both'):
            fault.install_pose_freeze()
        try:
            harness.run(args.case,args.gui,False)
        finally:
            (target/'injection.json').write_text(json.dumps({**meta,**fault.stats},indent=2)+'\n')


if __name__ == '__main__':
    main()
