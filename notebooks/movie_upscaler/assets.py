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
    print('Local model pack complete and verified.', flush=True)

def dataset_api():
    """Use private Kaggle credentials; never embed or print their values."""
    if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
        except Exception:
            pass
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        import subprocess
        import sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'kaggle'], check=True)
        from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def save_private_dataset(pack_root, owner='', api=None):
    """Automatically create a private named Dataset, following Image Studio's saver.

    Flat hardlinks expose nested weights to Kaggle's uploader without copying or
    archiving gigabytes. A successful submission is recorded; processing is remote.
    Existing Datasets are never overwritten or made public.
    """
    import re
    root = Path(pack_root)
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('assets') != ASSETS:
        raise RuntimeError('Refusing to upload an incomplete or different model pack.')
    for asset in ASSETS:
        path = root/asset['folder']/asset['name']
        if path.is_symlink() or not valid(path, asset):
            raise RuntimeError('Model failed verification before Dataset upload: '+asset['name'])
    api = api if api is not None else dataset_api()
    owner = owner or getattr(api, 'config_values', {}).get('username', '') or os.environ.get('KAGGLE_USERNAME', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise RuntimeError('Set DATASET_OWNER to your Kaggle username. Keep API credentials in Kaggle Secrets, never in the notebook.')
    fingerprint = hashlib.sha256(json.dumps(ASSETS, sort_keys=True).encode()).hexdigest()[:12]
    handle = owner+'/movie-upscaler-models-'+fingerprint
    url = 'https://www.kaggle.com/datasets/'+handle
    receipt = root/'dataset-upload.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('handle') == handle:
        print('Previously submitted private Dataset:', url, flush=True)
        return url
    stage = root/'dataset-files'
    stage.mkdir(exist_ok=True)
    expected = {a['name'] for a in ASSETS} | {'manifest.json', 'dataset-metadata.json'}
    if any(p.name not in expected for p in stage.iterdir()):
        raise RuntimeError('Unexpected upload-staging file preserved; inspect dataset-files before retrying.')
    for asset in ASSETS:
        source = root/asset['folder']/asset['name']
        target = stage/asset['name']
        if target.exists():
            if target.is_symlink() or not valid(target, asset):
                raise RuntimeError('Existing upload-staging file failed verification; preserved: '+target.name)
        else:
            os.link(source, target)
    (stage/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (stage/'dataset-metadata.json').write_text(json.dumps({
        'id': handle, 'title': 'Movie Upscaler Models '+fingerprint,
        'licenses': [{'name': 'other'}],
        'description': 'Private verified SeedVR2 and Real-ESRGAN model pack. Original weight licenses apply; no relicensing. See manifest.json for sources and SHA256 values.'
    }, indent=2))
    result = api.dataset_create_new(str(stage), public=False, quiet=False, convert_to_csv=False, dir_mode='skip')
    error = result.get('error') if isinstance(result, dict) else getattr(result, 'error', None)
    if result is None or error:
        raise RuntimeError('Dataset upload was not confirmed. Verified files are retained; no success receipt was written.')
    temp = receipt.with_suffix('.partial')
    temp.write_text(json.dumps({'handle': handle, 'url': url, 'submitted': True, 'requested_private': True,
                                'manifest_fingerprint': fingerprint}, indent=2))
    temp.replace(receipt)
    print('PRIVATE DATASET SUBMITTED:', url, flush=True)
    print('Wait for Kaggle processing, then attach it through Add Input → Your Datasets. No PC download or manual upload is needed.', flush=True)
    return url

def build_and_save(root, owner='', auto_create_dataset=True, input_root='/kaggle/input', api=None):
    # Fail authentication before spending time downloading the full model pack.
    if auto_create_dataset:
        import re
        if api is None:
            api = dataset_api()
        owner = owner or getattr(api, 'config_values', {}).get('username', '') or os.environ.get('KAGGLE_USERNAME', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
            raise RuntimeError('Set DATASET_OWNER to your Kaggle username before building. No model downloads started.')
    build_pack(root, input_root)
    if auto_create_dataset:
        return save_private_dataset(root, owner, api)
    print('Save Version with output files, then attach through Add Input → Notebook Outputs.', flush=True)
    return None
