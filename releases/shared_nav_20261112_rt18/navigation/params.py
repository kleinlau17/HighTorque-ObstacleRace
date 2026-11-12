"""Single place for the tuned constants. Defaults live in deploy/tuning.json."""
import json
from pathlib import Path

DEFAULTS = {
  "lane": {
    "exit_frames": 3,
    "recall_frames": 3,
    "recall_timeout_s": 3.0,
    "recall_gain": 1.5,
    "recall_limit": 0.25
  },
  "avoid": {
    "stop_frames": 5,
    "strafe_speed": 0.16,
    "strafe_s": 1.0,
    "back_speed": -0.30,
    "back_s": 1.0,
    "body_half": 0.38,
    "follow_vx": 0.07,
    "follow_ky": 1.8,
    "follow_vy_max": 0.12,
    "follow_kw": 2.5,
    "follow_w_max": 0.30,
    "strafe_offset": 0.22,
    "approach_near": 0.80,
    "approach_far": 1.00,
    "follow_align_tol": 0.08,
    "follow_side": 0.25
  },
  "box": {
    "sx": 0.20, "sy": 0.20, "sz": 0.40,
    "cluster_cell": 0.12,
    "cluster_min_pts": 12,
    "fit_step_deg": 3
  },
  "vision": {
    "white_dark": 115,
    "white_ratio": 0.25,
    "white_x_min": 0.30,
    "floor_band": 0.025,
    "raised_h_min": 0.06,
    "raised_h_max": 0.65,
    "raised_x_min": 0.10,
    "raised_x_max": 1.15,
    "path_half": 0.30,
    "path_abs": 0.33,
    "lane_x_lo": 0.15,
    "lane_x_hi": 1.25,
    "lane_bins": 9,
    "lane_gap": 0.18,
    "pair_min": 0.26,
    "pair_max": 1.00
  },
  "ground": {
    "refit_band": 0.03,
    "refit_rms": 0.005,
    "recover_failures": 5,
    "search_bins": 60,
    "search_ms": 80
  },
  "perception": {"scale": 0.4, "lease_s": 1.0}
}

_P = {k: dict(v) for k, v in DEFAULTS.items()}


def load(path=None):
    global _P
    merged = {k: dict(v) for k, v in DEFAULTS.items()}
    if path is not None:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            for section, values in data.items():
                if isinstance(values, dict) and section in merged:
                    merged[section].update(values)
                elif not isinstance(values, dict):
                    merged.setdefault(section, {})
    _P = merged
    return _P


def get(section, key):
    return _P[section][key]


def as_dict():
    return {k: dict(v) for k, v in _P.items()}
