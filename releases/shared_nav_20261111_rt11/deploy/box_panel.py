"""Per-lap box tuning panel and its small file mailbox for the vision worker."""
from dataclasses import asdict
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import sys
from threading import Lock, Thread
import time
from urllib.parse import parse_qs, urlsplit

from navigation.box_tuning import BoxTuning, DEPTH_REASON_TEXT, FIELDS

PORT=18761
PAIR_FILE=Path.home()/'.config/competition_nav/rt11-panel.key'


def signed(key, message):
    return hmac.new(key,message.encode(),hashlib.sha256).hexdigest()


def write_json(path, data):
    # A file has a single writer (HTTP submissions are serialized).
    temp=path.with_suffix('.new')
    with os.fdopen(os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as stream:
        json.dump(data,stream,ensure_ascii=False,allow_nan=False)
    temp.replace(path)


class VisionMailbox:
    def __init__(self, folder):
        self.folder=folder
        self.last_poll=-1.
        self.seen=0

    def update(self, servo, now):
        if now-self.last_poll<.15:
            return
        self.last_poll=now
        try:
            item=json.loads((self.folder/'box-request.json').read_text())
            if item['revision']>self.seen:
                servo.queue_tuning(item['revision'],item['values'])
                self.seen=item['revision']
        except FileNotFoundError:
            pass
        except (OSError,ValueError,TypeError,KeyError) as exc:
            print('RT11 panel request:',exc,file=sys.stderr)
        try:
            write_json(self.folder/'box-state.json',dict(servo.panel_status(now),timestamp=time.time()))
        except (OSError,ValueError) as exc:
            print('RT11 panel state:',exc,file=sys.stderr)


class BoxPanel:
    def __init__(self, folder, host='0.0.0.0', port=PORT, key_path=PAIR_FILE):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.vision=self.folder/'vision'
        self.vision.mkdir(exist_ok=True)
        self.session=secrets.token_hex(16)
        try:
            self.key=key_path.read_bytes()
            if len(self.key)!=32:
                raise ValueError('Invalid RT11 panel pairing key')
        except FileNotFoundError:
            self.key=secrets.token_bytes(32)
        self.token=signed(self.key,'panel/'+self.session)
        self.defaults=asdict(BoxTuning())
        try:
            request=json.loads((self.vision/'box-request.json').read_text())
        except FileNotFoundError:
            self.values=self.defaults.copy()
            self.revision=0
        else:
            self.values=asdict(BoxTuning.read(request['values']))
            self.revision=request['revision']
        self.lock=Lock()
        self.html=Path(__file__).with_name('box_panel.html').read_bytes()
        self.server=ThreadingHTTPServer((host,port),PanelHandler)
        self.server.panel=self
        self.thread=Thread(target=self.server.serve_forever,daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def state(self):
        try:
            control=json.loads((self.vision/'box-state.json').read_text())
        except (FileNotFoundError,ValueError):
            control={}
        if control.get('gate') in ('depth_unobserved','depth_blocked'):
            control['depth_explanation']=DEPTH_REASON_TEXT.get(control.get('depth_detail'))
        elif control.get('gate') in DEPTH_REASON_TEXT:
            control['depth_explanation']=DEPTH_REASON_TEXT[control['gate']]
        with self.lock:
            return dict(session=self.session,run=self.folder.parent.name,fields=FIELDS,defaults=self.defaults,
                        desired=self.values,revision=self.revision,control=control,
                        stale=time.time()-control.get('timestamp',0)>1.5)

    def submit(self,item):
        if not isinstance(item,dict) or type(item.get('base')) is not int:
            raise ValueError('缺少设置版本')
        values=asdict(BoxTuning.read(item.get('values')))
        with self.lock:
            if self.revision!=item['base']:
                return 409,dict(revision=self.revision,desired=self.values,error='另一窗口刚刚更新了参数。请核对当前值后重新应用。')
            revision=self.revision+1
            write_json(self.vision/'box-request.json',dict(revision=revision,values=values))
            self.values=values
            self.revision=revision
            return 200,dict(revision=revision)


class PanelHandler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def respond(self,code,data,kind='application/json; charset=utf-8',location=None):
        body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('X-Frame-Options','DENY')
        self.send_header('X-Content-Type-Options','nosniff')
        if location:self.send_header('Location',location)
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization','').encode(),
                                   ('Bearer '+self.server.panel.token).encode())

    def do_GET(self):
        panel=self.server.panel
        url=urlsplit(self.path)
        if url.path=='/':
            self.respond(200,panel.html,'text/html; charset=utf-8')
        elif url.path=='/local' and self.client_address[0]=='127.0.0.1' and self.headers.get('Host','').split(':')[0] in ('127.0.0.1','localhost'):
            self.respond(302,b'',location='/#'+panel.token)
        elif url.path=='/session':
            nonce=parse_qs(url.query).get('nonce',[''])[0]
            if len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce):
                return self.respond(400,{})
            proof=signed(panel.key,'open/'+panel.session+'/'+nonce)
            self.respond(200,dict(session=panel.session,proof=proof))
        elif url.path=='/api/state' and self.authorized():
            self.respond(200,panel.state())
        else:self.respond(403 if url.path=='/api/state' else 404,{})

    def do_POST(self):
        if self.path!='/api/settings' or not self.authorized():
            return self.respond(403,{})
        if self.headers.get_content_type()!='application/json':
            return self.respond(415,{})
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 1<=size<=2048:return self.respond(413,{})
            code,result=self.server.panel.submit(json.loads(self.rfile.read(size)))
        except (ValueError,TypeError,KeyError,OverflowError):
            return self.respond(400,dict(error='请输入各项允许范围内的数值'))
        except OSError:
            return self.respond(503,dict(error='机器人暂时无法保存设置'))
        self.respond(code,result)


def panel_address(port):
    client=os.environ.get('SSH_CONNECTION','').split()
    if len(client)==4:
        address=client[2]
    else:
        try:
            address=socket.gethostbyname(socket.gethostname())
            # Linux often maps the host name to 127.0.1.1.
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                sock.connect(('192.0.2.1',9))
                address=sock.getsockname()[0]
        except OSError:
            address='127.0.0.1'
    return 'http://'+('['+address+']' if ':' in address else address)+':'+str(port)


def start_panel(folder):
    panel=BoxPanel(folder)
    try:
        write_json(Path(folder)/'box-panel.json',dict(url=panel_address(panel.server.server_port)+'/#'+panel.token))
        panel.start()
        peer=os.environ.get('SSH_CONNECTION','').split()
        if len(peer)==4:
            # The paired Mac can find the new address after a Wi-Fi change.
            try:
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                    message=dict(session=panel.session,proof=signed(panel.key,'announce/'+panel.session))
                    sock.sendto(json.dumps(message).encode(),(peer[0],PORT+1))
            except OSError:
                pass
        return panel
    except Exception:
        panel.server.server_close()
        raise
