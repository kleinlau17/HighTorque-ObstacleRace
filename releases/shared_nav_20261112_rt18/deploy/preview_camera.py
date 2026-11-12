"""Local, read-only camera preview over SSH; no robot control interfaces.

Use --mode snapshots while another diagnostic owns the camera, then restart
with --mode live for continuous video. SSH authentication must already work.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time


REMOTE = r'''
import json, os, select, signal, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
proc = None

def shutdown(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGHUP, shutdown)

def send(jpeg, age, source):
    header = json.dumps(dict(size=len(jpeg), age_s=age, source=source)).encode()
    sys.stdout.buffer.write(header + b'\n' + jpeg)
    sys.stdout.buffer.flush()

def read_exact(stream, size, timeout=20):
    result = bytearray()
    deadline = time.monotonic() + timeout
    while len(result) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise TimeoutError('Camera frame timed out')
        chunk = os.read(stream.fileno(), size-len(result))
        if not chunk:
            raise EOFError('Camera stream closed')
        result.extend(chunk)
    return result

try:
    if mode == 'snapshots':
        previous = None
        while True:
            paths = list((root/'runs').glob('*/latest_rgb.jpg'))
            if paths:
                path = max(paths, key=lambda p: p.stat().st_mtime_ns)
                stat = path.stat()
                key = (str(path), stat.st_mtime_ns)
                if key != previous:
                    jpeg = path.read_bytes()
                    if jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9'):
                        send(jpeg, max(0, time.time()-stat.st_mtime), str(path.parent.name))
                        previous = key
            time.sleep(.2)
    else:
        import cv2
        import numpy as np
        proc = subprocess.Popen([str(root/'deploy/realsense_stream')],
                                stdout=subprocess.PIPE, bufsize=0)
        header = bytearray()
        while not header.endswith(b'\n'):
            header.extend(read_exact(proc.stdout, 1))
            if len(header) > 4096:
                raise ValueError('Invalid camera header')
        camera = json.loads(header)
        width, height = camera['width'], camera['height']
        frame_size = 8 + width*height*7
        last_sent = 0
        while True:
            raw = read_exact(proc.stdout, frame_size)
            now = time.monotonic()
            if now-last_sent < .12:
                continue
            rgb = np.frombuffer(raw, dtype=np.uint8, offset=8,
                                count=width*height*3).reshape(height, width, 3)
            ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            send(jpeg.tobytes(), 0, 'D435i RGB 640x480')
            last_sent = now
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
'''


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mini Pi Plus Pro · 真机相机</title><style>
body{margin:0;background:#11151c;color:#e8edf5;font:16px system-ui}
main{max-width:1040px;margin:28px auto;padding:0 24px}
h1{font-size:24px;margin:0 0 10px}p{color:#aebbd0;line-height:1.6}
.bar{display:flex;justify-content:space-between;gap:20px;margin:18px 0}
#status{color:#ffce78}#detail{color:#aebbd0;font-variant-numeric:tabular-nums}
.view{background:#080b10;border:1px solid #303b4c;border-radius:12px;
overflow:hidden;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}
img{display:block;width:100%;height:100%;object-fit:contain}
</style><main><h1>Mini Pi Plus Pro · 真机相机</h1>
<p>D435i 彩色视角 · 仅预览</p><div class="bar"><b id="status">正在连接相机…</b>
<span id="detail"></span></div><div class="view"><img id="view" alt="等待真机相机画面"></div>
<p id="note">正在读取机器人相机。</p></main><script>
const view=document.querySelector('#view'), status=document.querySelector('#status');
let sequence=-1;
async function refresh(){try{
 const s=await (await fetch('/status',{cache:'no-store'})).json();
 const live=s.mode==='live' && s.age_s!==null && s.age_s<2 && !s.error;
 status.textContent=s.error?'连接中断，正在重连':live?'● 实时画面':s.mode==='snapshots'?'● 最新采集画面（等待视角调整结束）':'等待新画面';
 status.style.color=live?'#63e1ae':'#ffce78';
 document.querySelector('#detail').textContent=s.age_s===null?'尚未收到图像':
  `画面已过去 ${s.age_s.toFixed(1)} 秒 · 接收 ${s.frames} 帧`;
 document.querySelector('#note').textContent=s.error||
  (s.mode==='snapshots'?'当前显示另一采集程序保存的最新图像；有新图像时自动更新。':'实时接收 640 × 480 彩色图像。');
 if(sequence!==s.frames && s.frames>0){view.src='/frame.jpg?n='+s.frames;sequence=s.frames;}
}catch(e){status.textContent='本机预览服务已停止';status.style.color='#ffce78';}
setTimeout(refresh,150);}refresh();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='nvidia@10.149.47.34')
    parser.add_argument('--remote-root', default='/home/nvidia/competition_nav/releases/pro_shadow_20261110_03')
    parser.add_argument('--control-path', default='/tmp/competition-preview-%C')
    parser.add_argument('--mode', choices=['live', 'snapshots'], default='snapshots')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    lock = threading.Lock()
    stopping = threading.Event()
    state = dict(jpeg=None, received=0, age=0, frames=0, error=None, source=None)
    active = [None]

    def ingest():
        command = 'python3 -u - '+shlex.quote(args.remote_root)+' '+shlex.quote(args.mode)
        while not stopping.is_set():
            proc = None
            try:
                proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o',
                    'ConnectTimeout=8', '-o', 'ServerAliveInterval=5', '-o',
                    'ServerAliveCountMax=2', '-o', 'ControlPath='+args.control_path,
                    args.host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                active[0] = proc
                proc.stdin.write(REMOTE.encode())
                proc.stdin.close()
                while not stopping.is_set():
                    line = proc.stdout.readline(4097)
                    if not line:
                        raise EOFError('相机连接断开；正在重新连接。')
                    meta = json.loads(line)
                    size = meta['size']
                    if not 0 < size < 5000000:
                        raise ValueError('Invalid JPEG size')
                    jpeg = proc.stdout.read(size)
                    if len(jpeg) != size:
                        raise EOFError('Incomplete camera frame')
                    with lock:
                        state.update(jpeg=jpeg, received=time.monotonic(), age=meta['age_s'],
                                     frames=state['frames']+1, error=None, source=meta['source'])
            except Exception as exc:
                with lock:
                    state['error'] = str(exc)
                print('Preview:', exc, flush=True)
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                active[0] = None
            stopping.wait(3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            with lock:
                if path == '/status':
                    age = state['age']+time.monotonic()-state['received'] if state['jpeg'] else None
                    body = json.dumps(dict(mode=args.mode, age_s=age, frames=state['frames'],
                                           error=state['error'], source=state['source'])).encode()
                    content_type = 'application/json'
                elif path == '/frame.jpg' and state['jpeg']:
                    body, content_type = state['jpeg'], 'image/jpeg'
                elif path == '/':
                    body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
                else:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    worker = threading.Thread(target=ingest, daemon=True)
    worker.start()
    print('Preview URL: http://127.0.0.1:'+str(args.port), flush=True)

    def shutdown(signum, frame):
        stopping.set()
        if active[0] is not None:
            active[0].terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        if active[0] is not None:
            active[0].terminate()
        worker.join(timeout=5)
        server.server_close()


if __name__ == '__main__':
    main()
