"""Measure the painted crate in camera-relative 3D from one RGB-D view.

Detection reuses the RT13 onboard crate algorithm (``BoxServo.observe``):
depth cells above the measured ground plane inside the crate height band, the
same 2.5 cm cell grid, 3x3 dilation and connected components.  Added here: the
colour label that picks the red crate, and the change of frame into the camera's
optical frame.

    .venv-shared/bin/python scripts/box_camera_position.py \
        --run output/sim/rt13_boxes_center_165736 --frame last

Frames reported
    camera optical  X right, Y down, Z forward  (metres, lens origin)
    robot body      X forward, Y left, Z up     (metres, body origin)

The run directory must contain ``scene.xml`` (camera mount + intrinsics),
``setup.json`` (calibration actually used), ``camera_rgbd.zip`` (``rgb/``,
``depth_mm/``), ``frames.jsonl`` (per-frame ground plane) and ``states.npz``
(MuJoCo qpos, used for the attitude).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# BoxServo.observe gates, verbatim.
HEIGHT_LOW, HEIGHT_HIGH = .065, .48
FORWARD_MIN, DEPTH_MAX = .15, 2.6
CELL, MIN_POINTS = .025, 25
CRATE_SIDE_M = .20


def camera_extrinsics(run: Path):
    """Body-relative camera position and optical->body rotation, as SensorRig."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body('base_link').id
    camera = model.camera('nav_camera').id
    body_from_world = data.xmat[base].reshape(3, 3).T
    position = body_from_world @ (data.cam_xpos[camera] - data.xpos[base])
    rotation = body_from_world @ data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    return position, rotation


def attitude_at(run: Path, time_s: float, initial_time: float) -> np.ndarray:
    """Heading-local attitude, exactly as ``OnboardNavigation`` builds it.

    The measured body->world rotation is premultiplied by the inverse of the
    initial heading, so the perception frame starts level with the robot's first
    heading and only roll, pitch and yaw drift remain.
    """
    import mujoco
    from navigation.visual import rotation2

    model = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    states = np.load(run / 'states.npz')
    base = model.body('base_link').id

    def rotation_at(stamp):
        data = mujoco.MjData(model)
        data.qpos[:] = states['qpos'][int(np.argmin(abs(states['t'] - stamp)))]
        mujoco.mj_forward(model, data)
        return data.xmat[base].reshape(3, 3).copy()

    start = rotation_at(initial_time)
    initial = np.eye(3)
    initial[:2, :2] = rotation2(-float(np.arctan2(start[1, 0], start[0, 0])))
    return initial @ rotation_at(time_s)


def load_frame(run: Path, index: int):
    archive = zipfile.ZipFile(run / 'camera_rgbd.zip')
    rgb = cv2.imdecode(np.frombuffer(archive.read(f'rgb/{index:05d}.png'), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imdecode(np.frombuffer(archive.read(f'depth_mm/{index:05d}.png'), np.uint8), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.
    depth[(depth <= 0) | ~np.isfinite(depth)] = np.nan
    return rgb, depth


def intrinsics(run: Path, shape):
    """Prefer the profile the run recorded; fall back to the current sim profile."""
    setup = run / 'setup.json'
    profile = json.loads(setup.read_text())['camera'] if setup.exists() else \
        json.loads((ROOT / 'sim/d435i_camera.json').read_text())
    scale = profile['processing_scale']
    if (int(profile['height'] * scale), int(profile['width'] * scale)) != shape:
        raise SystemExit(f'frame {shape} does not match the calibrated processing size '
                         f"{profile['width']}x{profile['height']} x {scale}")
    return (profile['fx'] * scale, profile['fy'] * scale,
            (profile['cx'] + .5) * scale - .5, (profile['cy'] + .5) * scale - .5)


def clusters(points, pixels):
    """Connected components on the 2.5 cm cell grid; returns point/pixel groups."""
    cells = np.clip(np.floor(points[:, :2] / CELL + 120).astype(np.int32), 0, 239)
    grid = np.zeros((240, 240), np.uint8)
    grid[cells[:, 1], cells[:, 0]] = 1
    count, labels = cv2.connectedComponents(cv2.dilate(grid, np.ones((3, 3), np.uint8)))
    groups = labels[cells[:, 1], cells[:, 0]]
    result = []
    for label in range(1, count):
        selected = groups == label
        if selected.sum() < MIN_POINTS:
            continue
        cloud, uv = points[selected], pixels[selected]
        if np.ptp(cloud[:, :2], axis=0).max() > .39 or np.ptp(cloud[:, 2]) < .055:
            continue
        result.append((cloud, uv))
    return result


def measure(run: Path, index: int, release: Path, red_ratio: float = 1.6, output: Path | None = None):
    """Detect crates in one recorded frame and report their camera-frame centres."""
    rows = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines() if line.strip()]
    row = rows[index]
    output = output or run / 'box_camera'
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(release))
    from navigation.ground import GroundPlane

    rgb, depth = load_frame(run, index)
    fx, fy, cx, cy = intrinsics(run, depth.shape)
    position, rotation = camera_extrinsics(run)
    attitude = attitude_at(run, float(row['t']), float(rows[0]['t']))

    ground = row['ground']
    plane = GroundPlane(True, np.asarray(ground['normal'], float), -float(ground['z_at_body_origin_m']),
                        float(ground['z_at_body_origin_m']), int(ground['support']),
                        float(ground['connected_support_area_m2']), float(ground['residual_rms_m']),
                        ground['reason'])

    height_px, width_px = depth.shape
    u, v = np.meshgrid(np.arange(width_px), np.arange(height_px))
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, np.float32)], -1)
    body = (rays * depth[..., None]) @ rotation.T + position
    level = body @ attitude.T
    height = level @ plane.normal + plane.offset
    keep = np.isfinite(height) & (height > HEIGHT_LOW) & (height < HEIGHT_HIGH) & (body[..., 0] > FORWARD_MIN)
    keep &= depth < DEPTH_MAX
    if not keep.any():
        raise SystemExit(f'frame {index}: no crate cells above the measured floor')

    points, pixels = level[keep], np.argwhere(keep)          # (row, col)
    overlay = rgb.copy()
    legend = []
    report = {'run': str(run), 'frame': index, 'time_s': float(row['t']),
              'ground_z_m': float(plane.z0), 'camera_position_body_m': position.tolist(),
              'gate': {'height_m': [HEIGHT_LOW, HEIGHT_HIGH], 'forward_min_m': FORWARD_MIN,
                       'depth_max_m': DEPTH_MAX, 'cell_m': CELL},
              'objects': []}

    for cloud, uv in clusters(points, pixels):
        colours = rgb[uv[:, 0], uv[:, 1]].astype(float)
        colour = np.median(colours, axis=0)
        red = bool(colour[0] > red_ratio * max(colour[1], colour[2], 1.))

        low, high = np.percentile(cloud, [2, 98], axis=0)
        footprint = np.ptp(cloud[:, :2], axis=0)
        centre_level = np.array([(low[0] + high[0]) / 2., (low[1] + high[1]) / 2.,
                                 plane.z0 + CRATE_SIDE_M])       # crate stands on the floor
        if max(footprint) < .18:
            # Only one face is visible: RT13 offsets that face by half the cube.
            face = cloud[:, :2].mean(axis=0)
            away = face - (position @ attitude.T)[:2]
            away /= max(np.linalg.norm(away), 1e-6)
            centre_level[:2] += away * CRATE_SIDE_M / 2.
        centre_body = centre_level @ attitude
        centre_camera = (centre_body - position) @ rotation
        camera_points = (cloud @ attitude - position) @ rotation
        nearest = float(np.percentile(camera_points[:, 2], 3))

        label = 'red' if red else 'other'
        report['objects'].append({
            'label': label, 'cells': int(len(cloud)),
            'median_rgb': [round(float(c), 1) for c in colour],
            'centre_camera_m': [round(float(c), 3) for c in centre_camera],
            'centre_body_m': [round(float(c), 3) for c in centre_body],
            'nearest_camera_z_m': round(nearest, 3),
            'distance_m': round(float(np.linalg.norm(centre_camera)), 3),
            'bearing_deg': round(float(np.degrees(np.arctan2(centre_camera[0], centre_camera[2]))), 1),
            'footprint_m': [round(float(c), 3) for c in np.ptp(cloud, axis=0)],
        })

        shade = (255, 40, 40) if red else (200, 200, 200)
        projected = np.column_stack([fx * camera_points[:, 0] / camera_points[:, 2] + cx,
                                     fy * camera_points[:, 1] / camera_points[:, 2] + cy])
        if red:
            for row_px, col_px in uv:                       # validated cells on the crate
                overlay[int(row_px), int(col_px)] = shade
            u0, v0 = np.clip(np.floor(projected.min(axis=0)).astype(int), [0, 0], [width_px - 1, height_px - 1])
            u1, v1 = np.clip(np.ceil(projected.max(axis=0)).astype(int), [0, 0], [width_px - 1, height_px - 1])
            cv2.rectangle(overlay, (u0, v0), (u1, v1), shade, 1)
            legend.append((f"RED  X {centre_camera[0]:+.2f}  Y {centre_camera[1]:+.2f}  Z {centre_camera[2]:+.2f} m",
                           f"     range {np.linalg.norm(centre_camera):.2f} m  bearing {np.degrees(np.arctan2(centre_camera[0], centre_camera[2])):+.1f} deg"))

    for line, (text, extra) in enumerate(legend):
        for row, body_text in enumerate((text, extra)):
            y = height_px - 16 + 12 * (line * 2 + row)
            cv2.putText(overlay, body_text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, body_text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 40, 40), 1, cv2.LINE_AA)
    reds = [item for item in report['objects'] if item['label'] == 'red']
    report['red'] = max(reds, key=lambda item: item['cells']) if reds else None
    (output / 'box_camera.json').write_text(json.dumps(report, indent=2))
    cv2.imwrite(str(output / 'box_camera.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frame', default='last', help="'last' or a frame index")
    parser.add_argument('--release', type=Path, default=ROOT / 'releases/shared_nav_20261112_rt13_passage')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--red-ratio', type=float, default=1.6, help='R/G and R/B ratio for the red label')
    args = parser.parse_args()
    run = args.run.resolve()
    rows = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines() if line.strip()]
    index = len(rows) - 1 if args.frame == 'last' else int(args.frame)
    report = measure(run, index, args.release, args.red_ratio, args.output)
    row = rows[index]

    print(f"frame {index} t={row['t']:.2f}s floor_z={report['ground_z_m']:+.3f}m objects={len(report['objects'])}")
    for item in report['objects']:
        print(f"  {item['label']:5s} rgb={item['median_rgb']} cells={item['cells']:3d} "
              f"camera=({item['centre_camera_m'][0]:+.3f},{item['centre_camera_m'][1]:+.3f},"
              f"{item['centre_camera_m'][2]:+.3f})m d={item['distance_m']:.3f}m "
              f"body=({item['centre_body_m'][0]:+.3f},{item['centre_body_m'][1]:+.3f},"
              f"{item['centre_body_m'][2]:+.3f})m")
    if report['red']:
        red = report['red']
        print(f"RED crate: camera X={red['centre_camera_m'][0]:+.3f} Y={red['centre_camera_m'][1]:+.3f} "
              f"Z={red['centre_camera_m'][2]:+.3f} m, distance {red['distance_m']:.3f} m, "
              f"bearing {red['bearing_deg']:+.1f} deg, nearest surface at {red['nearest_camera_z_m']:.3f} m")
    else:
        print('RED crate: not detected in this frame')
    print(f"wrote {(args.output or run/'box_camera')/'box_camera.json'} and box_camera.png")


if __name__ == '__main__':
    main()
