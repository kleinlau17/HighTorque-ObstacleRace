"""Watch a saved RT11 run with its original RGB and depth frames."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np

from rgbd_view import FrameReader, show_panel


def replay(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    frames = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
    frame_times = [row['t'] for row in frames]
    with np.load(directory / 'states.npz') as states:
        times, poses = states['t'], states['qpos']
    model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
    data = mujoco.MjData(model)
    camera = FrameReader(directory)
    first_box = next((t for t, reason in camera.plans if reason.startswith('box_')), 0.)
    keys = SimpleQueue()
    paused = False
    elapsed = 0.
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                          show_left_ui=False, show_right_ui=False) as view:
            with view.lock():
                view.cam.distance, view.cam.elevation = 2.05, -28.
            print('RT11 replay open', flush=True)
            previous = time.monotonic()
            while view.is_running():
                now = time.monotonic()
                if not paused:
                    elapsed += now - previous
                previous = now
                try:
                    while True:
                        key = keys.get_nowait()
                        if key == ord('P'):
                            paused = not paused
                        elif key == ord('R'):
                            elapsed = 0.
                        elif key == ord('B'):
                            elapsed = first_box
                        elif key in (ord('J'), ord('L')):
                            elapsed = np.clip(elapsed + (2. if key == ord('L') else -2.), 0., times[-1])
                except Empty:
                    pass
                if elapsed > times[-1] + 2.5:
                    elapsed = 0.
                index = max(0, bisect_right(times, elapsed) - 1)
                with view.lock():
                    data.qpos[:] = poses[index]
                    mujoco.mj_forward(model, data)
                    trunk = data.body('base_link')
                    view.cam.lookat[:] = trunk.xpos
                    view.cam.lookat[2] = .23
                    view.cam.azimuth = 83. + np.degrees(np.arctan2(trunk.xmat[3], trunk.xmat[0]))
                record = bisect_right(frame_times, elapsed) - 1
                gate = frames[record].get('crate_gate') if record >= 0 else None
                label = summary['result'] if elapsed >= times[-1] else gate or camera.reason_at(elapsed)
                view.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, 'RT11 | ' + summary['case'],
                                f'{times[index]:.2f}s   {label}\nP pause    B boxes    R restart    J / L skip 2s'))
                show_panel(view, camera.at(times[index]))
                view.sync()
                time.sleep(.018)
    finally:
        camera.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path, help='Run directory containing summary.json and states.npz')
    replay(cli.parse_args().directory)
