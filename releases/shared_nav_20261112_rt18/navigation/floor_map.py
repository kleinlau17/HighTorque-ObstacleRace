"""Bounded online map of metric floor patches in the initial AHRS frame."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Keyframe:
    time: float
    heading: float
    odom_xy: np.ndarray
    map_xy: np.ndarray
    patch: np.ndarray
    distance: float
    segment: int
    camera_position: np.ndarray
    camera_rotation: np.ndarray


def basis(heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])


def patch_displacement(reference, current, floor):
    """Match disjoint observed floor tiles, returning a translation in metres."""
    shifts = []
    # Distributed subwindows retain overlap after coasting. Matching the two
    # axes independently would accept a straight stripe as a place identity.
    for row in (6, 75, 144):
        for col in (5, 57):
            tile = current[row:row+48, col:col+38]
            std, ratio = floor.texture_quality(tile)
            if std < .10 or ratio < .035:
                continue
            scores = cv2.matchTemplate(reference, tile, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (x, y) = cv2.minMaxLoc(scores)
            if peak < .45:
                continue
            scores[max(y-4, 0):y+5, max(x-4, 0):x+5] = -1.
            if peak - scores.max() >= .12:
                shifts.append(np.array([y-row, x-col]) * floor.resolution)
    if len(shifts) < 2:
        return None
    center = np.median(shifts, axis=0)
    if np.max(np.linalg.norm(shifts-center, axis=1)) > .005:
        return None
    return center


class FloorMap:
    """Keep map corrections out of the continuous, local obstacle servo."""
    def __init__(self, floor):
        self.floor = floor
        self.keys = []
        self.segment = 0
        self.attached = True
        self.offset = np.zeros(2)
        self.pending = None
        self.fast_tracking_until = -np.inf
        self.last_check = -np.inf
        self.local_updates = 0
        self.revisits = 0
        self.recoveries = 0

    def start_segment(self):
        """A blind odometry restart leaves the transform to old keys unknown."""
        self.segment += 1
        self.pending = None
        self.attached = not self.keys

    @staticmethod
    def _confirmed(mode, timestamp, correction, observed, pending):
        if pending is None or pending[0] != mode:
            return False
        elapsed = timestamp - pending[1]
        if not 0 < elapsed < .6:
            return False
        if mode == 'track':
            # Degenerate lane geometry can shift odometry between samples.
            # Compare the two measured map positions in their common frame.
            return np.linalg.norm(observed - pending[3]) < .5 * elapsed + .01
        return np.linalg.norm(correction - pending[2]) <= .025

    def update(self, frame, attitude, plane, xy, distance, recovering=False):
        interval = .09 if recovering else .24
        if not recovering and frame.time <= self.fast_tracking_until:
            interval = .05
        if frame.time-self.last_check < interval or not plane.valid:
            return None
        self.last_check = frame.time
        yaw = np.arctan2(attitude[1, 0], attitude[0, 0])
        heading = round(yaw / .06) * .06
        patches = {}
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)

        def project(angle):
            if angle not in patches:
                patches[angle] = self.floor.project(gray, frame.depth, attitude, plane, angle)
            return patches[angle]

        # The odometry prior bounds the search only. Each accepted correction
        # requires matched sensor pixels in two successive frames.
        predicted = xy + self.offset
        recent, revisit = [], []
        for key in self.keys:
            if np.cos(key.heading-yaw) < np.cos(.24):
                continue
            travel = distance-key.distance
            separation = np.linalg.norm(key.odom_xy-xy) if recovering else np.linalg.norm(key.map_xy-predicted)
            if key.segment == self.segment and travel <= .6 and separation <= .35:
                recent.append((separation, key))
            elif not recovering and (key.segment != self.segment or travel >= 3.) and separation <= 1.:
                revisit.append((separation, key))

        def match_keys(candidates, limit):
            matches = []
            for _, key in sorted(candidates, key=lambda entry: entry[0])[:limit]:
                patch = project(key.heading)
                if patch is None:
                    continue
                delta = patch_displacement(key.patch, patch, self.floor)
                if delta is not None:
                    origin = key.odom_xy if recovering else key.map_xy
                    matches.append(origin + basis(key.heading) @ delta - xy)
            if matches:
                center = np.median(matches, axis=0)
                if np.max(np.linalg.norm(matches-center, axis=1)) <= .025:
                    return center
            return None

        correction = match_keys(revisit, 12)
        mode = 'revisit'
        if correction is None:
            mode = 'recover' if recovering else 'track'
            correction = match_keys(recent, 3)
            prior = np.zeros(2) if recovering else self.offset
            if correction is not None and np.linalg.norm(correction-prior) > (.18 if recovering else .20):
                correction = None
        pending = self.pending
        self.pending = None
        if correction is not None:
            observed = xy + correction
            self.pending = (mode, frame.time, correction, observed)
            if self._confirmed(mode, frame.time, correction, observed, pending):
                if recovering:
                    self.recoveries += 1
                    return xy + correction
                if mode == 'track' and np.linalg.norm(correction-self.offset) > .012:
                    self.fast_tracking_until = frame.time + .6
                self.attached = True
                self.offset = correction
                predicted = xy + correction
                if mode == 'revisit':
                    self.revisits += 1
                else:
                    self.local_updates += 1
            if mode == 'revisit':
                return None

        if recovering or not self.attached:
            return None
        if self.keys and self.keys[-1].segment == self.segment:
            last = self.keys[-1]
            if np.linalg.norm(predicted-last.map_xy) < .10 and np.cos(last.heading-yaw) > np.cos(.12):
                return None
        patch = project(heading)
        if patch is not None:
            std, ratio = self.floor.texture_quality(patch)
            if std >= .10 and ratio >= .035:
                self.keys.append(Keyframe(frame.time, heading, xy.copy(), predicted, patch, distance,
                                          self.segment, self.floor.position.copy(), self.floor.rotation.copy()))
                if len(self.keys) > 256:
                    del self.keys[0]
        return None

    def save(self, path):
        np.savez_compressed(path, map_xy=np.asarray([k.map_xy for k in self.keys]).reshape(-1, 2),
                            time=np.array([k.time for k in self.keys]),
                            segment=np.array([k.segment for k in self.keys], dtype=int),
                            heading=np.array([k.heading for k in self.keys]),
                            patch=np.array([k.patch for k in self.keys], dtype=np.float32),
                            intrinsics=np.asarray(self.floor.intrinsics),
                            camera_position_body_m=np.asarray([k.camera_position for k in self.keys]).reshape(-1, 3),
                            camera_rotation_optical_to_body=np.asarray([k.camera_rotation for k in self.keys]).reshape(-1, 3, 3),
                            resolution_m=self.floor.resolution,
                            forward_left_origin_m=[self.floor.x[0, 0], self.floor.y[0, 0]])
