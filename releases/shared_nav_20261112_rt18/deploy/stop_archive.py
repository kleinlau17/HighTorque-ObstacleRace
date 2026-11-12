"""Archive camera evidence for guarded command transitions, off the control thread."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time



def moving(vector):
    return any(float(v) != 0. for v in vector)


class GuardStops:
    """Count commanded stops at the sole velocity publisher, including startup."""
    def __init__(self):
        self.stop_id = 0
        self.moving = None
        self.stopped_at = None

    def update(self, stamp, command):
        active = moving(command)
        if not active and self.moving is not False:
            self.stop_id += 1
            self.stopped_at = stamp
        self.moving = active
        return self.stop_id, self.stopped_at


def append_row(out, row):
    with (out/'stops.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def stop_row(stop_id, stamp, reason, guard_reason, scene=None):
    return dict(event='stop', stop_id=stop_id, time=time.time()-(time.monotonic()-stamp),
                monotonic_s=stamp, reason=reason, guard_reason=guard_reason, scene=scene,
                image=None, depth=None, frame_timestamp_s=None, frame_age_s=None, new_frame=False,
                frame_relation='unavailable', imu=None, source='guard_command_zero')


def save_frames(out, basename, packet):
    import cv2
    import numpy as np
    meta, rgb, depth = packet
    image = basename+'.jpg'
    depth_name = basename+'_depth_mm.png'
    # Write complete files before publishing their names in the ledger.
    rgb_target, depth_target = out/image, out/depth_name
    tmp_rgb = rgb_target.with_suffix('.part.jpg')
    if not cv2.imwrite(str(tmp_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError('JPEG encoder returned false')
    os.replace(str(tmp_rgb), str(rgb_target))
    metric = np.zeros(depth.shape, np.uint16)
    valid = np.isfinite(depth) & (depth > 0.) & (depth <= 65.535)
    metric[valid] = np.maximum(1., np.rint(depth[valid]*1000.)).astype(np.uint16)
    tmp_depth = depth_target.with_suffix('.part.png')
    if not cv2.imwrite(str(tmp_depth), metric, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise IOError('Depth PNG encoder returned false')
    os.replace(str(tmp_depth), str(depth_target))
    return dict(meta, image=image, depth=depth_name, depth_unit='mm', invalid_depth=0,
                image_space='native_aligned_rgbd')


class StopArchive:
    def __init__(self, out, max_image_age=.35):
        self.out = out
        self.max_image_age = max_image_age
        self.closed = False
        (out/'stops').mkdir(parents=True, exist_ok=True)
        (out/'stops.jsonl').touch()
        self.frames = deque(maxlen=8)
        self.decisions = deque(maxlen=96)
        self.lock = threading.Lock()
        self.current = None
        self.recorded = False
        self.last_reported = set()
        self.previous_stop_frame = None
        # Only the diagnostics subscriber can wait for disk. The world-state
        # callbacks and the 50 Hz request publisher never take this queue's lock.
        self.jobs = queue.Queue(maxsize=16)
        self.encoded = {}
        self.worker = threading.Thread(target=self.write_loop, daemon=True)
        self.worker.start()
        self.previews = queue.Queue(maxsize=1)
        self.preview_worker = threading.Thread(target=self.write_previews, daemon=True)
        self.preview_worker.start()

    def preview(self, stamp, rgb, overlay):
        try:
            self.previews.get_nowait()
        except queue.Empty:
            pass
        self.previews.put_nowait((stamp,rgb,overlay))

    def write_previews(self):
        import cv2
        previous = None
        while True:
            job = self.previews.get()
            if job is None:
                return
            stamp, rgb, overlay = job
            try:
                name = '.stop_preview_%d.jpg'%int(stamp*1e9)
                path = self.out/name
                if not cv2.imwrite(str(path), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                    raise IOError('Preview JPEG encoder returned false')
                link = self.out/'.next_rgb.jpg'
                os.link(str(path),str(link))
                os.replace(str(link),str(self.out/'latest_rgb.jpg'))
                tmp = self.out/'.next_stop_rgb.json'
                tmp.write_text(json.dumps(dict(frame_timestamp_s=stamp,image=name,depth=None,
                    imu=None,image_space='rectified_perception_preview')))
                os.replace(str(tmp),str(self.out/'.latest_stop_rgb.json'))
                if previous:
                    previous.unlink()
                previous = path
                if cv2.imwrite(str(self.out/'.next_overlay.jpg'), cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)):
                    os.replace(str(self.out/'.next_overlay.jpg'),str(self.out/'latest_overlay.jpg'))
            except Exception as exc:
                print('Stop preview write failed: '+str(exc),flush=True)

    def frame(self, raw, imu, head):
        meta = dict(frame_timestamp_s=float(raw.time), imu=None)
        if imu is not None:
            meta['imu'] = dict(receipt_s=float(imu[0]), quaternion_xyzw=imu[1].tolist(),
                               message_time_s=float(imu[2]))
        if head:
            meta['head'] = {k:list(v) for k,v in head.items()}
        # The camera reader allocates a new pair per capture. Retaining these
        # arrays needs no copy or encode on the perception thread.
        self.frames.append((meta, raw.rgb, raw.depth))

    def decision(self, request_stamp, reason, scene, frame_time, new_frame):
        self.decisions.append((request_stamp, reason, scene, frame_time, new_frame))

    def enqueue(self, row, packet=None):
        self.jobs.put((row, packet))

    def guard(self, guard):
        if 'stop_id' not in guard:
            return
        with self.lock:
            if self.closed:
                return
            if moving(guard['command']):
                if self.current and not self.recorded:
                    self.emit(self.current)
                self.current = None
                self.recorded = False
                return
            new_stop = self.current is None or self.current['stop_id'] != guard['stop_id']
            if new_stop:
                if self.current and not self.recorded:
                    self.emit(self.current)
                self.current = dict(guard, **(guard.get('stop_origin') or {}))
                self.recorded = False
                self.last_reported.clear()
            if not self.recorded:
                # A startup zero may predate the first camera packet. Attach
                # that first packet only while startup still owns this zero.
                if guard['stop_id'] != 1 or (self.frames and self.frames[0][0]['frame_timestamp_s']<=guard['t']):
                    self.emit(self.current, guard['t'])
            else:
                stamp = guard.get('navigation_stamp_ns')
                context = next((d for d in reversed(self.decisions.copy()) if d[0]==stamp), None)
                reason = context[1] if context else None
                if reason in ('motion_stalled','turn_stalled') and reason not in self.last_reported:
                    self.last_reported.add(reason)
                    self.enqueue(dict(event='stop_reason',stop_id=guard['stop_id'],reason=reason,
                                      scene=context[2],monotonic_s=guard['t'],time=time.time(),
                                      image=self.image_for_stop,context_image=self.context_image_for_stop,guard_reason=guard['reason']))

    def emit(self, guard, through=None):
        stamp = guard['stopped_at']
        request_stamp = guard.get('navigation_stamp_ns')
        context = next((d for d in reversed(self.decisions.copy()) if d[0]==request_stamp), None)
        reason = guard.get('stop_reason') or guard['reason']
        if reason in ('navigation_zero','ok') and context:
            reason = context[1]
        row = stop_row(guard['stop_id'], stamp, reason, guard['reason'], context[2] if context else None)
        row['guard_source'] = guard['source']
        eligible = [p for p in self.frames.copy() if p[0]['frame_timestamp_s'] <= stamp]
        target = context[3] if context else None
        packet = next((p for p in reversed(eligible) if p[0]['frame_timestamp_s']==target),None)
        relation = 'decision_frame' if packet is not None else 'latest_preceding'
        row['decision_frame_timestamp_s'] = target
        if eligible and (packet is None or stamp-packet[0]['frame_timestamp_s']>self.max_image_age):
            packet = eligible[-1]
            relation = 'latest_preceding'
        if packet is None and guard['stop_id']==1 and through is not None:
            packet = next((p for p in self.frames.copy() if p[0]['frame_timestamp_s']<=through),None)
            relation = 'first_frame_while_stopped'
        self.image_for_stop = None
        self.context_image_for_stop = None
        if packet is not None:
            frame_time = packet[0]['frame_timestamp_s']
            age = stamp-frame_time
            if age < 0.:
                relation = 'subsequent_startup_reference'
            elif age > self.max_image_age:
                relation = 'stale_preceding_reference'
            row.update(frame_relation=relation,frame_age_s=stamp-frame_time,
                       frame_timestamp_s=frame_time,new_frame=bool(
                           (relation=='decision_frame' and context[4]) or relation=='first_frame_while_stopped'))
            if frame_time == self.previous_stop_frame:
                row['new_frame'] = False
            self.previous_stop_frame = frame_time
            if relation.endswith('_reference'):
                self.context_image_for_stop = 'stops/%05d_reference.jpg'%guard['stop_id']
            else:
                self.image_for_stop = 'stops/%05d.jpg'%guard['stop_id']
        self.enqueue(row,packet)
        self.recorded = True
        self.last_reported.add(reason)

    def save_event(self, row, packet):
        frame_time = packet[0]['frame_timestamp_s']
        reference = row['frame_relation'].endswith('_reference')
        basename = 'stops/%05d'%row['stop_id']+('_reference' if reference else '')
        saved = self.encoded.get(frame_time)
        if saved is None:
            saved = save_frames(self.out,basename,packet)
            self.encoded[frame_time] = saved
            if len(self.encoded)>16:
                del self.encoded[next(iter(self.encoded))]
        else:
            saved = dict(saved)
            for key, suffix in [('image','.jpg'),('depth','_depth_mm.png')]:
                target = basename+suffix
                os.link(str(self.out/saved[key]),str(self.out/target))
                saved[key] = target
        if reference:
            saved = {'context_'+k:v for k,v in saved.items()}
        row.update(saved)

    def write_loop(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            row, packet = job
            try:
                if packet is not None:
                    try:
                        self.save_event(row,packet)
                    except Exception as exc:
                        row['archive_error'] = str(exc)
                for key in ('image','context_image'):
                    if row.get(key) and not (self.out/row[key]).is_file():
                        row[key] = None
                append_row(self.out,row)
            except Exception as exc:
                print('Stop archive write failed: '+str(exc), flush=True)

    def close(self):
        with self.lock:
            if self.current and not self.recorded:
                self.emit(self.current,time.monotonic())
            latest = self.frames[-1] if self.frames else None
            self.closed = True
        self.jobs.put(None)
        self.worker.join()
        self.previews.put(None)
        self.preview_worker.join()
        if latest is not None:
            # The supervisor can finish a stop after the camera process exits.
            # This single temporary pair is consumed and removed at reconciliation.
            saved = save_frames(self.out,'.stop_last',latest)
            (self.out/'.stop_last.json').write_text(json.dumps(saved))


def reconcile(out, final_zero, terminal_reason):
    """After ownership is restored, account for missing guard edges and final zero."""
    vision = out/'vision'
    vision.mkdir(exist_ok=True)
    (vision/'stops').mkdir(exist_ok=True)
    ledger = vision/'stops.jsonl'
    ledger.touch()
    existing = set()
    with ledger.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if row['event']=='stop':
                    existing.add(row['stop_id'])
            except (KeyError,ValueError):
                continue
    saved = []
    for meta in (vision/'.stop_last.json',vision/'.latest_stop_rgb.json'):
        try:
            packet = json.loads(meta.read_text())
            if (vision/packet['image']).is_file():
                saved.append(packet)
        except (OSError,ValueError,KeyError):
            pass
    def missing(stop_id, stamp, reason, source):
        if stop_id in existing:
            return
        row = stop_row(stop_id,stamp,reason,reason)
        row['source'] = source
        eligible = [p for p in saved if p['frame_timestamp_s']<=stamp]
        if eligible:
            packet = max(eligible,key=lambda p:p['frame_timestamp_s'])
            age = stamp-packet['frame_timestamp_s']
            reference = age > .35
            copied = dict(packet)
            for key, suffix in [('image','.jpg'),('depth','_depth_mm.png')]:
                if packet.get(key):
                    target = 'stops/%05d%s%s'%(stop_id,'_reference' if reference else '',suffix)
                    shutil.copyfile(str(vision/packet[key]),str(vision/target))
                    copied[key] = target
            if reference:
                copied = {'context_'+k:v for k,v in copied.items()}
            row.update(copied,frame_timestamp_s=packet['frame_timestamp_s'],frame_age_s=age,
                       frame_relation='stale_preceding_reference' if reference else 'latest_preceding_on_cleanup')
        append_row(vision,row)
        existing.add(stop_id)
    guard_path = out/'guard.jsonl'
    last = None
    if guard_path.is_file():
        with guard_path.open() as stream:
            for line in stream:
                try:
                    guard = json.loads(line)
                except ValueError:
                    continue
                if 'stop_id' not in guard:
                    continue
                last = guard
                if guard['stop_id'] and guard['stop_id'] not in existing:
                    origin = guard.get('stop_origin')
                    if not origin and not moving(guard['command']):
                        origin = guard
                    reason = (origin.get('stop_reason') or origin['reason']) if origin else 'guard_zero'
                    missing(guard['stop_id'],guard['stopped_at'],reason,'guard_command_zero_recovered')
    if final_zero is not None and (last is None or moving(last['command'])):
        stop_id = 1 if last is None else last['stop_id']+1
        missing(stop_id,final_zero,terminal_reason,'supervisor_command_zero')
    for pattern in ('.stop_last*','.stop_preview_*.jpg','.latest_stop_rgb.json'):
        for path in vision.glob(pattern):
            path.unlink()
