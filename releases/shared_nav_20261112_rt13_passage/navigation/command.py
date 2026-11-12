"""Navigation-side adaptation to the unchanged factory AMP input filter.

The factory still applies its own filter/deadband. This module produces the
raw velocity request; it never changes policy weights or controller settings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_contract():
    return json.loads(Path(__file__).with_name('amp_contract.json').read_text())


def verify_device_files(amp_directory, controller_binary=None, pd_path=None):
    """Read-only identity check. A different factory revision is not accepted."""
    contract = load_contract()
    paths = {name: Path(amp_directory)/name for name in ('amp.yaml', 'amp.trt')}
    expected = dict(contract['sha256'])
    if pd_path is not None:
        paths['pd.yaml'] = Path(pd_path)
    if controller_binary is not None:
        paths['controller_binary'] = Path(controller_binary)
        expected['controller_binary'] = contract['controller_binary_sha256']
    actual = {}
    for name, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[name]:
            raise ValueError('Factory AMP contract changed: ' + name)
        actual[name] = digest
    return actual


class AMPCommandAdapter:
    """Quantized low-speed targets and bounded filter feed-forward at 50 Hz.

    Feedback, if provided, is the actual policy's observed vx/vy/wz, not body
    velocity. Stop never requests reverse motion to cancel a filter.
    """
    def __init__(self):
        self.contract = load_contract()
        self.filtered = np.zeros(3)
        self.active = np.zeros(3, dtype=bool)
        self.start_threshold = np.array([.04, .04, .08])
        self.stop_threshold = np.array([.02, .02, .04])
        self.minimum_target = np.array([.10, .10, .20])
        self.raw_limit = np.array([.4, .4, 1.])
        self.target = np.zeros(3)

    def update(self, desired, feedback=None, enabled=True):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError('Invalid navigation velocity')
        if feedback is not None:
            feedback = np.asarray(feedback, dtype=float)
            if feedback.shape != (3,) or not np.isfinite(feedback).all():
                raise ValueError('Invalid policy command feedback')
            self.filtered = feedback.copy()
        self.active = ((abs(desired) >= self.start_threshold) |
                       (self.active & (abs(desired) > self.stop_threshold))) if enabled else np.zeros(3, bool)
        self.target = np.where(self.active, np.sign(desired)*np.maximum(abs(desired), self.minimum_target), 0.)
        # With this exact Pro network, 0.10--0.16 forward-only inputs reach
        # the policy but settle back into standing in the simulation probe.
        # 0.20 sustains walking. Lateral motion has a different response and
        # must retain small longitudinal corrections for narrow box gaps.
        if self.target[0] > 0 and self.target[1] == 0:
            self.target[0] = max(.20, self.target[0])
        c = self.contract
        previous = self.filtered.copy()
        raw = (self.target/np.asarray(c['scale'])-(1-c['alpha'])*previous)/c['alpha']
        raw = np.where(self.target > 0, np.maximum(raw, 0), np.minimum(raw, 0))
        raw = np.where(self.active, raw, 0.)
        raw = np.clip(raw, np.maximum(c['lower'], -self.raw_limit), np.minimum(c['upper'], self.raw_limit))
        self.filtered = (c['alpha']*raw+(1-c['alpha'])*previous)*c['scale']
        self.filtered[abs(self.filtered) < np.asarray(c['deadband'])] = 0.
        return raw
