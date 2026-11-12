"""Calibrate the box-centre lateral estimate against a tape measure.

The box-zone thresholds (``avoid.body_half``, the ``abs(by)`` gate) are written
in the controller's own body frame. If that estimate is biased, a threshold that
looks right on paper clears the box on paper only. This tool compares what the
perception reports with what a tape says, at several known placements, and turns
the result into a corrected threshold.

It reads the newest ``scene.json`` written by ``deploy/live_scene.py`` on the
robot. Field run (one box in view, robot standing still, per station):

    .venv-shared/bin/python deploy/box_offset_calibration.py \
        --host nvidia@10.149.47.34 --password nvidia \
        --remote /home/nvidia/competition_nav/releases/shared_nav_20261112_rt16/runs/live \
        --wrist-reach .30 --margin .05 --box-half .10 \
        --output runs/box_offset_calibration.csv

Offline fit from a recorded CSV (true_x,true_y,rep_x,rep_y):

    .venv-shared/bin/python deploy/box_offset_calibration.py --points points.csv

At each station the tape values are typed in: lateral offset of the box CENTRE
from the robot centreline (left positive, metres) and forward distance. The
robot centreline is a plumb line through the midpoint of the two hip joints, not
the base-link origin.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REMOTE_SPOOL = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted(root.rglob('scene.json'), key=lambda p: p.stat().st_mtime_ns)
if not paths:
    print(json.dumps({'error': 'no scene.json under ' + str(root)})); raise SystemExit(3)
print(paths[-1].read_text())
'''


def newest_scene_local(path):
    p = Path(path)
    if p.is_file():
        return json.loads(p.read_text())
    paths = sorted(p.rglob('scene.json'), key=lambda x: x.stat().st_mtime_ns)
    if not paths:
        raise SystemExit(f'no scene.json under {p}')
    return json.loads(paths[-1].read_text())


def newest_scene_remote(host, remote, password):
    cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10', host,
           'python3 -c ' + subprocess.list2cmdline([REMOTE_SPOOL]) + ' ' + remote]
    if password:
        cmd = ['sshpass', '-p', password] + cmd
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise SystemExit((done.stderr or done.stdout).strip() or 'ssh failed')
    return json.loads(done.stdout)


def nearest_box(scene):
    boxes = [b for b in scene.get('boxes', []) if b and 'center' in b]
    if not boxes:
        return None
    return min(boxes, key=lambda b: float(np.hypot(*b['center'])))


def ask(prompt, default=None):
    shown = f' [{default}]' if default is not None else ''
    while True:
        raw = input(f'{prompt}{shown}: ').strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print('  请输入数字（米）')


def fit(points):
    """points: list of (true_x, true_y, rep_x, rep_y). Returns fits and residuals."""
    a = np.asarray(points, dtype=float)
    tx, ty, rx, ry = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    result = {}
    for name, true, rep in (('y', ty, ry), ('x', tx, rx)):
        if len(set(np.round(true, 4))) < 2:
            gain, bias = 1.0, float(np.mean(rep - true))
        else:
            gain, bias = np.polyfit(true, rep, 1)
        pred = gain*true + bias
        result[name] = dict(gain=float(gain), bias=float(bias),
                            rmse=float(np.sqrt(np.mean((rep-pred)**2))),
                            max_err=float(np.max(np.abs(rep-pred))))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', help='scene.json file or directory (local)')
    ap.add_argument('--host', default='nvidia@10.149.47.34')
    ap.add_argument('--remote', default='/home/nvidia/competition_nav/runs/live')
    ap.add_argument('--password')
    ap.add_argument('--points', help='offline CSV: true_x,true_y,rep_x,rep_y')
    ap.add_argument('--wrist-reach', type=float, default=.30,
                    help='half of the measured maximum body width (metres)')
    ap.add_argument('--margin', type=float, default=.05, help='wanted clearance to the box face')
    ap.add_argument('--box-half', type=float, default=.10)
    ap.add_argument('--output', default='runs/box_offset_calibration.csv')
    args = ap.parse_args()

    points = []
    if args.points:
        with open(args.points, newline='') as f:
            for row in csv.DictReader(f):
                points.append((float(row['true_x']), float(row['true_y']),
                               float(row['rep_x']), float(row['rep_y'])))
        if not points:
            raise SystemExit('no rows in ' + args.points)
    else:
        if not args.scene and not args.host:
            raise SystemExit('需要 --scene 或 --host 读取 scene.json')
        print('每站：机器人站定、只放一只箱子、量好卷尺再输入。空行结束。')
        while True:
            raw = input('横向偏移（米，左正）: ').strip()
            if not raw:
                break
            true_y = float(raw)
            true_x = ask('前向距离（米）')
            if args.scene:
                scene = newest_scene_local(args.scene)
            else:
                scene = newest_scene_remote(args.host, args.remote, args.password)
            box = nearest_box(scene) if scene else None
            if box is None:
                print('   scene.json 里没有箱子，检查 live_scene 是否在跑、箱子是否在画面里')
                continue
            rep_x, rep_y = float(box['center'][0]), float(box['center'][1])
            print(f'   估计: x={rep_x:.3f} y={rep_y:.3f}  (点数 {box.get("points","?")})')
            points.append((true_x, true_y, rep_x, rep_y))

    fits = fit(points)
    rows = []
    print('\n 真值 x,y        估计 x,y        残差 y')
    for tx, ty, rx, ry in points:
        pred = fits['y']['gain']*ty + fits['y']['bias']
        print(f' {tx:6.3f},{ty:6.3f}   {rx:6.3f},{ry:6.3f}   {ry-pred:+.3f}')
        rows.append(dict(true_x=tx, true_y=ty, rep_x=rx, rep_y=ry, residual_y=ry-pred))

    need_true = args.wrist_reach + args.margin + args.box_half
    gain, bias = fits['y']['gain'], fits['y']['bias']
    need_rep = abs(gain)*need_true + bias
    print(f'\n横向估计: 增益 {gain:.3f} 偏差 {bias:+.3f} m, RMSE {fits["y"]["rmse"]:.3f} m, 最大残差 {fits["y"]["max_err"]:.3f} m')
    print(f'前向估计: 增益 {fits["x"]["gain"]:.3f} 偏差 {fits["x"]["bias"]:+.3f} m, RMSE {fits["x"]["rmse"]:.3f} m')
    print(f'腕展 {args.wrist_reach:.2f} m + 余量 {args.margin:.2f} m + 箱半宽 {args.box_half:.2f} m')
    print(f'=> 真值要求 |车心-箱心| >= {need_true:.2f} m; 按标定折算控制器应看到 |by| >= {need_rep:.2f} m')
    if fits['y']['max_err'] > .02:
        print('警告: 残差超过 2 cm, 先用它修外参/深度, 再定阈值。')
    if abs(gain-1) > .15:
        print('警告: 增益偏离 1 超过 15%, 检查相机外参与深度尺度。')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\n已写入 {out}')


if __name__ == '__main__':
    main()
