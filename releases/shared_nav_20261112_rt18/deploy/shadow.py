"""Real sensor diagnostic. Intentionally contains NO ROS publisher or motor API."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import select
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Imu, JointState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.geometry import quaternion_matrix_xyzw, yaw_matrix


def read_exact(stream, size, timeout=5.):
    deadline = time.monotonic()+timeout
    chunks = []
    remaining = size
    while remaining:
        wait = deadline-time.monotonic()
        if wait <= 0 or not select.select([stream], [], [], max(0., wait))[0]:
            raise TimeoutError('RGB-D stream timed out')
        data = os.read(stream.fileno(), remaining)
        if not data:
            raise EOFError('RGB-D stream closed')
        chunks.append(data)
        remaining -= len(data)
    return b''.join(chunks)


@dataclass
class Frame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    # Host monotonic time at which the packet was fully read. Kept last with a
    # default so existing Frame(time, rgb, depth) construction stays valid;
    # capture_worker always passes the real read-completion time explicitly.
    received: float = 0.0


class Sensors:
    """Live factory state for the perception loop.

    The per-motor MotorState topics publish at roughly 700 Hz each, about 15k
    messages per second for the 22 joints. That is more Python callback work
    than one core can absorb, and it starved this process's main thread: the
    same perception frame measured 34 ms alone but ~1 s while those
    subscriptions were active. The controller republishes the identical 22
    joints at 18 Hz on /sim2real_master_node/rbt_state, which the guard and the
    terminal preflight already read, so one JointState subscription replaces
    the 22 topics.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.imu = deque(maxlen=500)
        self.joints = {}
        self.head = {name: deque(maxlen=200) for name in ('head_pitch_joint', 'head_yaw_joint')}
        self.subs = [
            rospy.Subscriber('/imu/data', Imu, self.on_imu, queue_size=20),
            rospy.Subscriber('/sim2real_master_node/rbt_state', JointState, self.on_state, queue_size=1)]

    def on_imu(self, msg):
        q = msg.orientation
        values = np.array([q.x,q.y,q.z,q.w])
        if msg.orientation_covariance[0] < 0 or not np.isfinite(values).all() or abs(np.linalg.norm(values)-1) > .05:
            return
        with self.lock:
            self.imu.append((time.monotonic(), values, float(msg.header.stamp.to_sec())))

    def on_state(self, msg):
        stamp = time.monotonic()
        with self.lock:
            for joint, position, velocity in zip(msg.name, msg.position, msg.velocity):
                if joint in self.head:
                    self.head[joint].append((stamp, float(position), float(velocity)))
                else:
                    self.joints[joint] = (stamp, float(position), float(velocity))

    def snapshot(self, timestamp):
        with self.lock:
            nearest = min(self.imu, key=lambda v:abs(v[0]-timestamp)) if self.imu else None
            joints = dict(self.joints)
            head = {name: min(history, key=lambda v: abs(v[0]-timestamp))
                    for name, history in self.head.items() if history}
        if nearest is None or abs(nearest[0]-timestamp) > .15:
            raise ValueError('No fresh IMU orientation near camera host receipt')
        return nearest, joints, head


def capture_worker(stream, width, height, latest, status=None):
    def offer(value):
        # Consumer may empty the queue between full() and get(); do not turn
        # that ordinary race into a camera failure.
        while True:
            try:
                latest.put_nowait(value)
                return
            except queue.Full:
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
    try:
        while True:
            packet = read_exact(stream, 8+width*height*7, timeout=18.)
            received = time.monotonic()
            timestamp = struct.unpack('<d', packet[:8])[0]
            n = width*height
            frame = Frame(timestamp,
                np.frombuffer(packet, np.uint8, n*3, 8).reshape(height,width,3).copy(),
                np.frombuffer(packet, '<f4', n, 8+n*3).reshape(height,width).copy(),
                received)
            if status is not None:
                status['last_received'] = received
                status['last_frame_time'] = timestamp
                status['frames'] = status.get('frames', 0)+1
            offer(frame)
    except Exception as exc:
        if status is not None:
            status['error'] = exc
        offer(exc)


def main():
    # The legacy name now enters the same shared pipeline as run_visual.py.
    from deploy.shared_navigation import main as shared_main
    return shared_main()


if __name__ == '__main__':
    raise SystemExit(main())
