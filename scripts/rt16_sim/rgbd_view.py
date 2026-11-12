"""Record and display the simulated D435i inputs without resampling the sensors."""
from bisect import bisect_right
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARCHIVE = 'camera_rgbd.zip'
NEAR_MM, FAR_MM = 150, 4000
INVALID = (93, 103, 111)
PANEL_WIDTH, PANEL_HEIGHT = 280, 532
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
try:
    LABEL = ImageFont.truetype(FONT, 16)
    SMALL = ImageFont.truetype(FONT, 13)
except OSError:
    LABEL = SMALL = ImageFont.load_default()
PHASE = {'box_enter': '进入箱区', 'box_along': '沿箱前进', 'box_shift': '箱间横移',
         'box_lane_entry': '回到车道中心', 'ready': '巡线', 'initializing': '等待起步',
         'return_to_observed_lane': '根据边线纠偏', 'unknown_depth': '深度待确认',
         'lost_vision': '视觉跟踪丢失', 'stale_camera': '等待新相机帧',
         'waiting_for_trusted_visual_motion_and_lane': '等待运动与边线观测',
         'turn_reobserving': '转向后重新观测', 'checking_raised_object': '确认凸起物'}
# Near is yellow, far is purple. Keep valid distant returns distinct from invalid gray.
levels = np.linspace(244, 36, 256).round().astype(np.uint8).reshape(1, -1)
COLORS = cv2.cvtColor(cv2.applyColorMap(levels, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)[0]


def millimeters(depth_m):
    result = np.zeros(depth_m.shape, np.uint16)
    valid = np.isfinite(depth_m)
    result[valid] = np.rint(depth_m[valid]*1000).astype(np.uint16)
    return result


class FrameWriter:
    def __init__(self, directory):
        self.path = Path(directory)/ARCHIVE
        self.zip = zipfile.ZipFile(self.path.with_suffix('.partial'), 'w')
        self.frames = []
        self.plans = []

    def capture(self, stamp, available, rgb, depth):
        number = len(self.frames)
        for name, array in (('rgb', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                            ('depth_mm', millimeters(depth))):
            ok, png = cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise RuntimeError('Unable to encode camera frame')
            self.zip.writestr(f'{name}/{number:05d}.png', png.tobytes())
        self.frames.append((available, stamp))

    def plan(self, stamp, reason):
        if not self.plans or self.plans[-1][1] != reason:
            self.plans.append((stamp, reason))

    def close(self):
        self.zip.writestr('index.json', json.dumps(dict(frames=self.frames, plans=self.plans,
            frame_columns=['available_at_sim_s','capture_stamp_s'], depth_unit='mm', invalid=0)))
        self.zip.close()
        self.path.with_suffix('.partial').replace(self.path)


class FrameReader:
    def __init__(self, directory):
        self.zip = zipfile.ZipFile(Path(directory)/ARCHIVE)
        index = json.loads(self.zip.read('index.json'))
        self.frames = index['frames']
        self.times = [item[0] for item in self.frames]
        self.plans = index['plans']
        self.plan_times = [item[0] for item in self.plans]
        self.previous = None
        self.panel = None

    def at(self, t):
        i = bisect_right(self.times, t) - 1
        p = bisect_right(self.plan_times, t) - 1
        if self.previous != (i, p):
            stamp = rgb = depth_mm = None
            if i >= 0:
                stamp = self.frames[i][1]
                rgb = cv2.cvtColor(self.decode('rgb', i), cv2.COLOR_BGR2RGB)
                depth_mm = self.decode('depth_mm', i)
            reason = self.plans[p][1] if p >= 0 else 'initializing'
            self.panel = sensor_panel(rgb, depth_mm, stamp, reason)
            self.previous = (i, p)
        return self.panel

    def reason_at(self, t):
        i = bisect_right(self.plan_times, t) - 1
        return self.plans[i][1] if i >= 0 else 'initializing'

    def decode(self, name, i):
        buf = self.zip.read(f'{name}/{i:05d}.png')
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)

    def close(self):
        self.zip.close()


def body_lateral_speed(trace):
    """Simulated base velocity projected leftward, averaged over a 0.20 s window."""
    t = trace[:, 0]
    before = np.searchsorted(t, t - .10).clip(0, len(t)-1)
    after = np.searchsorted(t, t + .10).clip(0, len(t)-1)
    delta = (trace[after, 1:3] - trace[before, 1:3]) / (t[after] - t[before])[:, None]
    yaw = trace[:, 3]
    return -delta[:, 0]*np.sin(yaw) + delta[:, 1]*np.cos(yaw)


def sensor_panel(rgb, depth_mm, stamp, reason):
    panel = Image.new('RGB', (PANEL_WIDTH, PANEL_HEIGHT), '#15262e')
    pen = ImageDraw.Draw(panel)
    pen.text((12, 6), '机载彩色图  RGB', font=LABEL, fill='white')
    pen.text((12, 230), '配准深度图', font=LABEL, fill='white')
    if rgb is None:
        for y in (32, 256):
            pen.rectangle((12, y, 267, y+191), fill='#263740')
            pen.text((91, y+84), '等待相机帧', font=LABEL, fill='#c5d1d5')
    else:
        idx = np.clip((depth_mm.astype(float)-NEAR_MM)*255/(FAR_MM-NEAR_MM), 0, 255).astype(np.uint8)
        colors = COLORS[idx]
        colors[depth_mm == 0] = INVALID
        panel.paste(Image.fromarray(rgb).resize((256,192), Image.Resampling.NEAREST), (12,32))
        panel.paste(Image.fromarray(colors).resize((256,192), Image.Resampling.NEAREST), (12,256))
    gradient = np.repeat(COLORS[None, :], 8, axis=0)
    panel.paste(Image.fromarray(gradient).resize((176,8)), (12,459))
    pen.text((12,469), '近 0.15 m', font=SMALL, fill='white')
    pen.text((134,469), '远 4.0 m', font=SMALL, fill='white')
    pen.rectangle((210,457,221,469), fill=INVALID)
    pen.text((227,453), '无效', font=SMALL, fill='white')
    stamp_text = '—' if stamp is None else f'{stamp:.2f} s'
    pen.text((12,494), f'采集 {stamp_text}', font=SMALL, fill='#c2d4dd')
    pen.text((120,494), PHASE.get(reason, reason), font=SMALL, fill='white')
    return panel


def show_panel(view, panel):
    import mujoco
    viewport = view.viewport
    if viewport is None or viewport.width < 6 or viewport.height < 24:
        return
    scale = min(viewport.width*.27 / PANEL_WIDTH, (viewport.height-12) / PANEL_HEIGHT, 1.4)
    w, h = max(1,int(PANEL_WIDTH*scale)), max(1,int(PANEL_HEIGHT*scale))
    pixels = np.asarray(panel.resize((w,h), Image.Resampling.BILINEAR))
    view.set_images((mujoco.MjrRect(viewport.left+viewport.width-w-6, viewport.bottom+6, w,h), pixels))
