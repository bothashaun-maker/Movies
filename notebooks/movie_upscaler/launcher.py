# One startup cell. Processing starts only after Run in ComfyUI.
from pathlib import Path
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REQUIRE_ATTACHED_MODELS = True
ENABLE_SEEDVR2 = True
PORT = 8188
BASE = Path('/kaggle/temp/movie-studio-v1')
OUT = Path('/kaggle/working/movie-studio')
COMFY_REV = '1332fabfcc507aa7c548cd5152071ed494c207e8'
SEED_REV = '4490bd1f482e026674543386bb2a4d176da245b9'
CF_SHA = '1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872'
# Build embeds the catalog, custom IO code, and ordinary connected workflows here.
EMBEDDED_PAYLOAD = None

BASE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
lock = open(BASE/'startup.lock', 'w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError('Movie Studio is already running; stop its earlier startup cell.')
children = []

def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)

def checkout(url, dest, revision):
    if not dest.exists():
        run(['git', 'clone', '--no-checkout', url, dest])
        run(['git', '-C', dest, 'checkout', '--detach', revision])
    actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'Unexpected source revision at {dest}; preserved.')

def spawn(args, log, env=None):
    with open(log, 'w') as f:
        p = subprocess.Popen(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    children.append(p)
    return p

try:
    with socket.socket() as check:
        try:
            check.bind(('127.0.0.1', PORT))
        except OSError:
            raise RuntimeError('Port occupied. Stop the earlier Image/Movie/H3 studio cell first.')
    required = [a for a in ASSETS if ENABLE_SEEDVR2 or a['folder'] != 'SEEDVR2']
    found = {}
    for asset in required:
        dest = BASE/'ComfyUI/models'/asset['folder']/asset['name']
        if dest.exists() and not valid(dest, asset):
            raise RuntimeError(f'Model checksum mismatch; preserved: {dest}')
        candidate = dest.resolve() if dest.exists() else find_asset(asset, ['/kaggle/input'])
        if candidate:
            found[asset['name']] = candidate
        elif REQUIRE_ATTACHED_MODELS:
            raise RuntimeError('Missing verified model '+asset['name']+'. Attach the automatically created Movie Upscaler Dataset via Add Input → Your Datasets, or attach saved Notebook Outputs. No model downloads started.')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        if os.geteuid() == 0 and shutil.which('apt-get'):
            run(['apt-get', '-qq', 'update'])
            run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise RuntimeError('FFmpeg/ffprobe installation was unavailable in this Kaggle image.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select Kaggle GPU T4 and rerun. Model-pack preparation uses Accelerator: None.')
    free, total = torch.cuda.mem_get_info(0)
    if free < 10*1024**3:
        raise RuntimeError('GPU 0 needs at least 10 GiB free; stop other GPU processes first.')
    print('GPU 0:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
    print('This version uses GPU 0. A second T4 does not pool its memory.', flush=True)
    comfy = BASE/'ComfyUI'
    checkout('https://github.com/comfyanonymous/ComfyUI.git', comfy, COMFY_REV)
    seed = comfy/'custom_nodes/SeedVR2'
    if ENABLE_SEEDVR2:
        checkout('https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git', seed, SEED_REV)
    python = BASE/'venv/bin/python'
    if not python.exists():
        run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', python.parent.parent])
    protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy']
    constraints = BASE/'constraints.txt'
    constraints.write_text('\n'.join(f'{p}=={metadata.version(p)}' for p in protected))
    requirements = BASE/'requirements.txt'
    text = (comfy/'requirements.txt').read_text()
    if ENABLE_SEEDVR2:
        text += '\n'+(seed/'requirements.txt').read_text()
    requirements.write_text(text)
    fingerprint = hashlib.sha256((text+constraints.read_text()+sys.version+SEED_REV).encode()).hexdigest()
    marker = BASE/'setup.complete'
    if not marker.exists() or marker.read_text() != fingerprint:
        run([sys.executable, '-m', 'pip', '--python', python, 'install', '-r', requirements, '-c', constraints])
        run([python, '-c', 'import torch, torchvision, numpy, cv2, aiohttp; assert torch.cuda.is_available()'])
        marker.write_text(fingerprint)
    for asset in required:
        dest = comfy/'models'/asset['folder']/asset['name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            if asset['name'] in found:
                dest.symlink_to(found[asset['name']])
            else:
                fetch(asset, dest)
    custom = comfy/'custom_nodes/MovieStudio'
    custom.mkdir(parents=True, exist_ok=True)
    for name, text in FILES.items():
        target = custom/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    workflows = {k: v for k, v in WORKFLOWS.items() if ENABLE_SEEDVR2 or 'SeedVR2' not in k}
    wf_dir = comfy/'user/default/workflows'
    wf_dir.mkdir(parents=True, exist_ok=True)
    for name, graph in workflows.items():
        (wf_dir/(name+'.json')).write_text(json.dumps(graph))
    default = 'Movie Studio - SeedVR2' if ENABLE_SEEDVR2 else 'Movie Studio - RealESRGAN'
    (custom/'web/default.json').write_text(json.dumps(workflows[default]))
    cf = BASE/'cloudflared'
    if not cf.exists():
        fetch({'url': 'https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/cloudflared-linux-amd64',
               'bytes': CF_SIZE, 'sha256': CF_SHA}, cf)
    if sha(cf) != CF_SHA:
        raise RuntimeError('cloudflared checksum mismatch.')
    cf.chmod(0o700)
    env = os.environ.copy()
    env['MOVIE_STUDIO_URL'] = f'http://127.0.0.1:{PORT}'
    server = spawn([python, comfy/'main.py', '--listen', '127.0.0.1', '--port', PORT, '--lowvram', '--force-fp16',
                    '--preview-method', 'none', '--cache-none', '--output-directory', OUT, '--disable-auto-launch'], OUT/'comfy.log', env)
    deadline = time.monotonic()+240
    while True:
        if server.poll() is not None:
            raise RuntimeError('ComfyUI stopped. See /kaggle/working/movie-studio/comfy.log.')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/object_info', timeout=3) as response:
                info = json.load(response)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError('ComfyUI startup timed out; see comfy.log.')
            time.sleep(1)
    for graph in workflows.values():
        for node in graph['nodes']:
            if node['type'] not in info:
                raise RuntimeError('Required node unavailable: '+node['type']+'. See comfy.log.')
    (OUT/'provenance.json').write_text(json.dumps({'comfy': COMFY_REV, 'seedvr2': SEED_REV, 'assets': required,
                                                  'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, indent=2))
    with open(OUT/'packages.txt', 'w') as f:
        run([sys.executable, '-m', 'pip', '--python', python, 'freeze'], stdout=f)
    tunnel = spawn([cf, 'tunnel', '--url', f'http://127.0.0.1:{PORT}', '--no-autoupdate'], OUT/'tunnel.log')
    deadline = time.monotonic()+90
    url = None
    while time.monotonic() < deadline:
        matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', (OUT/'tunnel.log').read_text())
        if matches:
            url = matches[-1]+'/?movie_studio=v1'
            break
        if tunnel.poll() is not None:
            raise RuntimeError('Cloudflare stopped; see tunnel.log.')
        time.sleep(1)
    if not url:
        raise RuntimeError('No studio URL received; see tunnel.log.')
    (OUT/'studio-url.txt').write_text(url)
    print('\nOPEN MOVIE STUDIO:', url, flush=True)
    print('Set source_path in the leftmost node to your attached MP4/MKV, then press Run.')
    videos = [p for p in Path('/kaggle/input').rglob('*') if p.is_file() and p.suffix.lower() in ('.mp4', '.mkv', '.mov', '.webm', '.avi')]
    for movie in videos[:20]: print('Attached video:', movie)
    print('No processing has started. Use queue count 1; turn OFF frontend Auto Queue.')
    print('To pause: Interrupt, then Clear Queue. Press Run with the same settings to resume.')
    print('Saved chunks and final movies:', OUT/'movie-jobs')
    print('Keep this cell running. Save Version with output files before ending the session.')
    print('Anyone with this temporary session URL can access the studio.', flush=True)
    while server.poll() is None and tunnel.poll() is None:
        time.sleep(2)
    raise RuntimeError('Studio or tunnel stopped; inspect saved logs.')
finally:
    for p in reversed(children):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
