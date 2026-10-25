# Start persistent ComfyUI and print the Cloudflare browser link.
import re, socket, platform, urllib.request, subprocess, time
from IPython.display import display, HTML

assert STACK_READY and PY.is_file() and COMFY.is_dir(), 'Installation must finish before opening ComfyUI.'
assert platform.machine().lower() in ('x86_64', 'amd64'), 'This cloudflared binary requires x86_64 Linux.'

TUNNEL_BIN = ROOT / 'cloudflared'
SERVER_LOG = ROOT / 'comfyui-ui.log'
TUNNEL_LOG = ROOT / 'cloudflared.log'
cloudflared_sha='053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0'
if not TUNNEL_BIN.is_file() or sha256_file(TUNNEL_BIN)!=cloudflared_sha:
    # Official Cloudflare binary; downloaded in the running Kaggle environment.
    url = 'https://github.com/cloudflare/cloudflared/releases/download/2026.10.3/cloudflared-linux-amd64'
    temp = TUNNEL_BIN.with_suffix('.download')
    try:
        urllib.request.urlretrieve(url, temp)
        assert sha256_file(temp)==cloudflared_sha, 'cloudflared checksum mismatch.'
        temp.chmod(0o755)
        temp.replace(TUNNEL_BIN)
    finally:
        temp.unlink(missing_ok=True)

# Stop this notebook's previous tunnel before starting a new one.
old_tunnel = globals().get('_h3_ui_tunnel')
if old_tunnel is not None and old_tunnel.poll() is None:
    old_tunnel.terminate()
    try: old_tunnel.wait(timeout=10)
    except subprocess.TimeoutExpired:
        old_tunnel.kill();old_tunnel.wait(timeout=10)

# Reuse an already-listening ComfyUI, or launch it using the same GPU flags as the renderer.
def local_server_ready():
    try:
        api_request('/system_stats', timeout=3)
        return True
    except Exception:
        return False

if not local_server_ready():
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError(f'Port {PORT} is occupied by another program.')
    ui_flags = ['--cuda-device','all','--vram-headroom',str(VRAM_HEADROOM_GB),
                '--preview-method','none','--disable-auto-launch']
    ui_flags += ['--use-ck-attention'] if USE_KITCHEN_ATTENTION else ['--use-pytorch-cross-attention']
    with SERVER_LOG.open('w') as log:
        _h3_ui_server = subprocess.Popen(
            [str(PY),'-u','main.py','--listen','127.0.0.1','--port',str(PORT),*ui_flags],
            cwd=COMFY, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if _h3_ui_server.poll() is not None:
            raise RuntimeError('ComfyUI exited during startup:\n'+tail_log(SERVER_LOG))
        if local_server_ready(): break
        time.sleep(2)
    else: raise TimeoutError('ComfyUI did not become ready:\n'+tail_log(SERVER_LOG))

assert 'H3IsolatedSaveVideo' in api_request('/object_info'), 'H3 workflow node did not load; inspect comfyui-ui.log.'
print('In ComfyUI: Workflows → MiniMax H3 - Full T4 Workflow → Run',flush=True)

with TUNNEL_LOG.open('w') as log:
    _h3_ui_tunnel = subprocess.Popen(
        [str(TUNNEL_BIN),'tunnel','--no-autoupdate','--protocol','http2','--url',f'http://127.0.0.1:{PORT}'],
        stdout=log, stderr=subprocess.STDOUT)

cloudflare_url = None
for _ in range(90):
    if _h3_ui_tunnel.poll() is not None:
        raise RuntimeError('Cloudflare tunnel exited:\n'+tail_log(TUNNEL_LOG))
    log_text = TUNNEL_LOG.read_text(errors='replace')
    found = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',log_text)
    if found:
        cloudflare_url = found.group(0)
        break
    time.sleep(1)
if not cloudflare_url:
    raise TimeoutError('No Cloudflare URL found. Check '+str(TUNNEL_LOG)+'\n'+tail_log(TUNNEL_LOG))
print('Your ComfyUI Cloudflare link:',cloudflare_url)
display(HTML('<a href="'+cloudflare_url+'" target="_blank" rel="noopener noreferrer">Open ComfyUI in browser</a>'))
print('Keep this Kaggle session running while using ComfyUI. The temporary link grants public access.')

(ROOT/'comfyui-cloudflare-url.txt').write_text(cloudflare_url+'\n')

def stop_comfyui_access():
    for name in ('_h3_ui_tunnel','_h3_ui_server'):
        proc = globals().get(name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait(timeout=10)
    print('ComfyUI browser access stopped.')
