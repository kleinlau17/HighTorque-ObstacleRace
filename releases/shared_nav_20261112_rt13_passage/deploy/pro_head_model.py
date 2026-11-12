"""Nominal Pro head kinematics from the user's URDF; no ROS or motor access.

camera_link is a mechanical frame. This module intentionally does not invent a
camera_link -> RGB optical-frame transform or an encoder-zero calibration.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def _vector(element, key, default):
    value = default if element is None else element.get(key, default)
    result = np.fromstring(value, sep=' ')
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Invalid URDF vector: ' + key)
    return result


def _rotation(axis, angle):
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or not np.isfinite(angle):
        raise ValueError('Invalid joint axis/angle')
    x, y, z = axis / norm
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*cross + (1.-np.cos(angle))*(cross@cross)


class ProHeadModel:
    def __init__(self, path):
        root = ET.parse(Path(path)).getroot()
        parents = {}
        for joint in root.findall('joint'):
            child = joint.find('child').get('link')
            if child in parents:
                raise ValueError('Multiple parents for link: ' + child)
            parents[child] = joint
        chain, link, visited = [], 'camera_link', set()
        while link != 'base_link':
            if link in visited or link not in parents:
                raise ValueError('No unique base_link -> camera_link chain')
            visited.add(link)
            joint = parents[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain = list(reversed(chain))
        self.limits = {}
        for joint in self.chain:
            kind, name = joint.get('type'), joint.get('name')
            if kind == 'revolute':
                limit = joint.find('limit')
                lo, hi = float(limit.get('lower')), float(limit.get('upper'))
                if not np.isfinite([lo, hi]).all() or lo >= hi:
                    raise ValueError('Invalid limits: ' + name)
                self.limits[name] = (lo, hi)
            elif kind != 'fixed':
                raise ValueError('Unsupported head joint type: ' + kind)
        if set(self.limits) != {'head_yaw_joint', 'head_pitch_joint'}:
            raise ValueError('Unexpected movable head chain')

    def check_angles(self, angles, margin=0.):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('Invalid limit margin')
        for name, (lo, hi) in self.limits.items():
            q = float(angles[name])
            if not np.isfinite(q) or not lo+margin <= q <= hi-margin:
                raise ValueError(name + ' is outside the supplied URDF limits')

    def camera_link_transform(self, angles):
        """Return base_from_camera_link, for URDF angles inside model limits."""
        self.check_angles(angles)
        transform = np.eye(4)
        for joint in self.chain:
            origin = joint.find('origin')
            xyz = _vector(origin, 'xyz', '0 0 0')
            roll, pitch, yaw = _vector(origin, 'rpy', '0 0 0')
            step = np.eye(4)
            step[:3, :3] = (_rotation(np.array([0., 0., 1.]), yaw)
                            @ _rotation(np.array([0., 1., 0.]), pitch)
                            @ _rotation(np.array([1., 0., 0.]), roll))
            step[:3, 3] = xyz
            if joint.get('type') == 'revolute':
                step[:3, :3] = step[:3, :3] @ _rotation(
                    _vector(joint.find('axis'), 'xyz', '1 0 0'),
                    float(angles[joint.get('name')]))
            transform = transform @ step
        return transform
