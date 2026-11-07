"""Pinned model metadata from upstream registry and HF LFS metadata."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.request

ASSETS = [
    {'folder': 'SEEDVR2', 'name': 'seedvr2_ema_3b-Q8_0.gguf', 'bytes': 3660613984,
     'sha256': 'be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96',
     'url': 'https://huggingface.co/AInVFX/SeedVR2_comfyUI/resolve/ac66d6d98fa49975d893b58c55bff7677191c862/seedvr2_ema_3b-Q8_0.gguf'},
    {'folder': 'SEEDVR2', 'name': 'ema_vae_fp16.safetensors', 'bytes': 501324814,
     'sha256': '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1',
     'aliases': ['seedvr2_ema_vae_fp16.safetensors'],
     'url': 'https://huggingface.co/numz/SeedVR2_comfyUI/resolve/09ced71023636e9bc8cdf9cdecfb2625d1e691e8/ema_vae_fp16.safetensors'},
    {'folder': 'upscale_models', 'name': 'RealESRGAN_x4plus.pth', 'bytes': 67040989,
     'sha256': '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
     'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'},
]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def valid(path, asset):
    return path.is_file() and path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256']

def find_asset(asset, roots):
    for root in roots:
        if not Path(root).exists():
            continue
        for name in [asset['name']]+asset.get('aliases', []):
            for p in Path(root).rglob(name):
                if valid(p, asset):
                    return p.resolve()
    return None

def fetch(asset, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if valid(dest, asset):
            return dest
        raise RuntimeError(f'Existing file failed verification; preserved: {dest}')
    part = dest.with_suffix(dest.suffix+'.partial')
    offset = part.stat().st_size if part.exists() else 0
    if offset > asset['bytes']:
        raise RuntimeError(f'Oversized partial file preserved: {part}')
    if shutil.disk_usage(dest.parent).free < asset['bytes']-offset+512*1024**2:
        raise RuntimeError('Insufficient free disk for model and 512 MiB reserve.')
    if offset != asset['bytes']:
        req = urllib.request.Request(asset['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(req, timeout=120) as response:
            append = offset and response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
            if response.status == 206 and not append and offset:
                raise RuntimeError('Invalid resume range; partial file preserved.')
            if not append and offset and shutil.disk_usage(dest.parent).free < asset['bytes']+512*1024**2:
                raise RuntimeError('Server requires restart; insufficient free disk.')
            with open(part, 'ab' if append else 'wb') as f:
                shutil.copyfileobj(response, f, 8*1024**2)
                f.flush()
                os.fsync(f.fileno())
    if not valid(part, asset):
        raise RuntimeError(f'Checksum/size mismatch; partial file preserved: {part}')
    part.replace(dest)
    return dest

def build_pack(root, input_root='/kaggle/input'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'manifest.json'
    marker.unlink(missing_ok=True)
    for asset in ASSETS:
        dest = root/asset['folder']/asset['name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            cached = find_asset(asset, [input_root])
            if cached:
                if shutil.disk_usage(dest.parent).free < asset['bytes']+512*1024**2:
                    raise RuntimeError('Not enough space to copy attached model into durable output.')
                shutil.copyfile(cached, dest)
        print('Verify / prepare:', asset['name'], flush=True)
        fetch(asset, dest)
    # Actual files, never symlinks into another notebook output.
    if any((root/a['folder']/a['name']).is_symlink() for a in ASSETS):
        raise RuntimeError('Pack must contain actual model files, not symlinks.')
    temp = marker.with_suffix('.partial')
    temp.write_text(json.dumps({'complete': True, 'assets': ASSETS}, indent=2))
    temp.replace(marker)
    print('Model pack complete. Save Version with output files; attach this saved output to Movie Upscaler.', flush=True)
