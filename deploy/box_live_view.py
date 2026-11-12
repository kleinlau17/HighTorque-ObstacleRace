"""Read-only live view: robot camera + box-zone measurements, in one window.

Runs on the operator machine. A single SSH session reads the newest
``latest_rgb.jpg`` and the last telemetry row written by the navigation
release, so it can run while the robot is driving (no camera contention).

    .venv-shared/bin/python deploy/box_live_view.py --password nvidia
    open http://127.0.0.1:8767

The page shows the camera image with an overlay of the measurements the
box-zone controller uses: raised-cell count, its lateral centre, the chosen
side and how many white tape points were found on each side.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
import threading
import time

REMOTE = r'''
import json, os, sys, time
from pathlib import Path

root = Path(sys.argv[1])

def newest(glob):
    paths = list(root.glob(glob))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime_ns)

def tail_json(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size-65536))
            lines = stream.read().split(b'\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                return row
    except OSError:
        pass
    return {}

previous = None
while True:
    image = newest('runs/**/latest_rgb.jpg')
    overlay = None
    if image is not None:
        candidate = image.parent/'latest_overlay.jpg'
        if candidate.exists():
            overlay = candidate
    if image is None:
        time.sleep(.3)
        continue
    stat = image.stat()
    key = (str(image), stat.st_mtime_ns)
    if key == previous:
        time.sleep(.1)
        continue
    previous = key
    row = tail_json(image.parent/'telemetry.jsonl')
    try:
        data = image.read_bytes()
    except OSError:
        continue
    if not (data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9')):
        continue
    okey = None
    if overlay is not None:
        try:
            okey = overlay.read_bytes()
        except OSError:
            okey = None
    cloud = b''
    points_path = image.parent/'latest_points.json'
    if points_path.exists():
        try:
            cloud = points_path.read_bytes()
        except OSError:
            cloud = b''
    extra_bytes = []
    for name in ('latest_depth.jpg', 'latest_3d.jpg'):
        candidate = image.parent/name
        try:
            extra_bytes.append(candidate.read_bytes() if candidate.exists() else b'')
        except OSError:
            extra_bytes.append(b'')
    header = json.dumps(dict(
        size=len(data), overlay_size=len(okey or b''),
        depth_size=len(extra_bytes[0]), view3d_size=len(extra_bytes[1]),
        cloud_size=len(cloud),
        age_s=max(0., time.time()-stat.st_mtime),
        release=image.parts[-5] if len(image.parts) >= 5 else '',
        t=row.get('t'), reason=row.get('reason'), scene=row.get('scene'),
        cells=row.get('ride_cells'), y=row.get('ride_y'),
        side=row.get('ride_side'), tape=row.get('tape_points'),
        lane=row.get('lane_basis'), boxes=len(row.get('boxes') or []),
        moving=(row.get('measured_velocity') or [0, 0])[0])).encode()
    sys.stdout.buffer.write(header + b'\n' + data)
    if okey:
        sys.stdout.buffer.write(okey)
    for extra in extra_bytes:
        if extra:
            sys.stdout.buffer.write(extra)
    if cloud:
        sys.stdout.buffer.write(cloud)
    sys.stdout.buffer.flush()
'''

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>箱区识别 · 实时</title><style>
body{margin:0;background:#0d1117;color:#e8edf5;font:15px system-ui}
main{max-width:1180px;margin:22px auto;padding:0 20px}
h1{font-size:22px;margin:0 0 6px}p{color:#9fb0c7}
.bar{display:flex;justify-content:space-between;gap:16px;margin:12px 0}
#state{font-weight:700}#detail{color:#9fb0c7;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
figure{margin:0;background:#05070a;border:1px solid #2b3546;border-radius:12px;overflow:hidden}
figcaption{color:#9fb0c7;font-size:13px;padding:6px 8px;border-bottom:1px solid #1d2531}
figure img{display:block;width:100%;height:auto}
canvas{display:block;width:100%;height:360px;cursor:grab;touch-action:none}
</style><main><h1>箱区识别 · 实时</h1>
<p>彩色带检测标记 · 深度 · 可拖动 3D 点云（橙=箱子 洋红=白线）</p>
<div class="bar"><span id="state">连接中…</span><span id="detail"></span></div>
<div class="bar"><label><input type="checkbox" id="raw"> 彩色显示原始画面（不叠加标记）</label>
<label><input type="checkbox" id="spin"> 自动旋转</label></div>
<div class="grid">
 <figure><figcaption>彩色（橙=箱子 洋红=白线）</figcaption><img id="view" alt="等待画面"></figure>
 <figure><figcaption>深度（近亮远暗）</figcaption><img id="depth" alt="等待深度"></figure>
 <figure><figcaption>3D 点云：拖动旋转，滚轮缩放</figcaption><canvas id="cloud"></canvas></figure>
</div></main><script>
const img=document.querySelector('#view'),state=document.querySelector('#state'),
      detail=document.querySelector('#detail'),depthImg=document.querySelector('#depth');
const cv=document.querySelector('#cloud'),ctx=cv.getContext('2d');
let seq=-1,yaw=-0.7,pitch=0.55,dist=3.4,cloud=null,drag=null,z0=0;

function fit(){const r=cv.getBoundingClientRect();cv.width=r.width;cv.height=r.height;draw();}
window.addEventListener('resize',fit);

function project(x,y,z){
  const cy=Math.cos(yaw),sy=Math.sin(yaw);
  let X=x*cy-y*sy, Y=x*sy+y*cy, Z=z;
  const cp=Math.cos(pitch),sp=Math.sin(pitch);
  const Y2=Y*cp-Z*sp, Z2=Y*sp+Z*cp;
  const d=dist-X;
  if(d<0.15) return null;
  const s=(cv.height*0.62)/d;
  return [cv.width/2+Y2*s, cv.height*0.72-Z2*s];
}
function medianZ(pts){if(!pts||!pts.length)return 0;const a=pts.map(p=>p[2]).sort((x,y)=>x-y);return a[a.length>>1];}
function axis(p0,p1,color,label){
  const a=project(p0[0],p0[1],p0[2]-z0), b=project(p1[0],p1[1],p1[2]-z0);
  if(!a||!b)return;
  ctx.strokeStyle=color;ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  ctx.fillStyle=color;ctx.font='12px system-ui';
  ctx.fillText(label,b[0]+3,b[1]-3);
}
function draw(){
  ctx.fillStyle='#05070a';ctx.fillRect(0,0,cv.width,cv.height);
  if(!cloud) return;
  const put=(pts,color)=>{ctx.fillStyle=color;
    for(const p of pts){const q=project(p[0],p[1],p[2]-z0);
      if(!q)continue; ctx.fillRect(q[0],q[1],1.6,1.6);}};
  // ground grid at z=0
  ctx.strokeStyle='#182431';ctx.lineWidth=1;
  for(let g=-1;g<=2;g+=0.5){
    const a=project(-0.5,g,0), b=project(2.5,g,0);
    if(a&&b){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
    const c=project(g,-1.2,0), d=project(g,1.2,0);
    if(c&&d){ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(d[0],d[1]);ctx.stroke();}
  }
  put(cloud.floor,'#2f9dd6');
  put(cloud.tape,'#ff5ce1');
  put(cloud.raised,'#ff8c00');
  axis([0,0,0],[1.2,0,0],'#ff5b5b','X 前');
  axis([0,0,0],[0,1.0,0],'#5bff8c','Y 左');
  axis([0,0,0],[0,0,0.8],'#5b9bff','Z 上');
  const o=project(0,0,0);
  if(o){ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(o[0],o[1],4,0,6.283);ctx.fill();}
}
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!drag)return;
  yaw=drag.yaw+(e.clientX-drag.x)*0.01;
  pitch=Math.max(-0.2,Math.min(1.4,drag.pitch+(e.clientY-drag.y)*0.01));draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(1.4,Math.min(6,dist+e.deltaY*0.002));draw();},{passive:false});

async function tick(){try{
 const s=await (await fetch('/status',{cache:'no-store'})).json();
 const seen=(s.cells||0)>=8;
 state.textContent=seen?'● 已识别到箱子':'○ 前方未识别到箱子';
 state.style.color=seen?'#63e1ae':'#ffce78';
 detail.textContent=`t=${s.t? s.t.toFixed(1):'-'}s · 格子 ${s.cells??'-'} · 横向 ${s.y==null?'-':s.y.toFixed(2)}m · 选边 ${s.side||0} · 白点 ${(s.tape||[0,0]).join('/')} · 车道 ${s.lane||'-'}`;
 if(seq!==s.frames&&s.frames>0){
   img.src=(document.querySelector('#raw').checked?'/frame.jpg':'/overlay.jpg')+'?n='+s.frames;
   depthImg.src='/depth.jpg?n='+s.frames;
   try{const r=await fetch('/points?n='+s.frames);
     if(r.ok){cloud=await r.json();z0=medianZ(cloud.floor);draw();}}catch(e){}
   seq=s.frames;
 }
}catch(e){state.textContent='预览服务已停止';state.style.color='#ff9d9d';}
setTimeout(tick,250);}tick();fit();
setInterval(()=>{if(document.querySelector('#spin').checked){yaw+=0.06;draw();}},80);
</script></main></html>'''



class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.overlay = None
        self.depth = None
        self.view3d = None
        self.cloud = None
        self.fields = {}
        self.frames = 0
        self.error = None


S = State()


def pull(host, root, password):
    cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10', host,
           'python3 - ' + root]
    while True:
        try:
            env = None
            if password:
                cmd = ['sshpass', '-p', password] + cmd
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, env=env)
            proc.stdin.write(REMOTE.encode())
            proc.stdin.close()
            stream = proc.stdout
            while True:
                line = stream.readline()
                if not line:
                    raise EOFError('stream closed')
                header = json.loads(line)
                data = b''
                while len(data) < header['size']:
                    chunk = stream.read(header['size']-len(data))
                    if not chunk:
                        raise EOFError('truncated frame')
                    data += chunk
                def take(count):
                    blob = b''
                    while len(blob) < count:
                        chunk = stream.read(count-len(blob))
                        if not chunk:
                            raise EOFError('truncated image')
                        blob += chunk
                    return blob
                overlay = take(int(header.get('overlay_size') or 0))
                depth = take(int(header.get('depth_size') or 0))
                view3d = take(int(header.get('view3d_size') or 0))
                cloud = take(int(header.get('cloud_size') or 0))
                with S.lock:
                    S.frame = data
                    S.overlay = overlay or S.overlay
                    S.depth = depth or S.depth
                    S.view3d = view3d or S.view3d
                    if cloud:
                        S.cloud = cloud
                    S.fields = header
                    S.frames += 1
                    S.error = None
        except Exception as exc:  # reconnect forever
            with S.lock:
                S.error = str(exc)
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/frame.jpg'):
            with S.lock:
                data = S.frame
            if not data:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith('/overlay.jpg'):
            with S.lock:
                data = S.overlay
            if not data:
                data = b''
            self.send_response(200 if data else 503)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)
        elif self.path.startswith('/depth.jpg') or self.path.startswith('/view3d.jpg'):
            with S.lock:
                data = S.depth if self.path.startswith('/depth') else S.view3d
            self.send_response(200 if data else 503)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data or b'')))
            self.end_headers()
            if data:
                self.wfile.write(data)
        elif self.path.startswith('/points'):
            with S.lock:
                data = S.cloud
            self.send_response(200 if data else 503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data or b'')))
            self.end_headers()
            if data:
                self.wfile.write(data)
        elif self.path.startswith('/status'):
            with S.lock:
                payload = dict(S.fields, frames=S.frames, error=S.error)
            body = json.dumps(payload, default=float).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='nvidia@10.149.47.34')
    parser.add_argument('--remote-root',
                        default='/home/nvidia/competition_nav/releases/shared_nav_20261112_rt13_passage')
    parser.add_argument('--password', default=None)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    threading.Thread(target=pull, args=(args.host, args.remote_root, args.password),
                     daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('http://127.0.0.1:%d' % args.port, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
