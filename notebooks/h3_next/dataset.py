"""Private Kaggle Dataset upload for verified H3 Next packs; no weight copies."""
from pathlib import Path
import hashlib,json,os

def valid(path,asset):
    if not path.is_file() or path.stat().st_size!=asset["size"]:return False
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):digest.update(block)
    return digest.hexdigest()==asset["sha256"]

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

def save_private_dataset(pack_root, pack, assets, owner='', api=None):
    """Automatically create a private named Dataset, following Image Studio's saver.

    Flat hardlinks expose nested weights to Kaggle's uploader without copying or
    archiving gigabytes. A successful submission is recorded; processing is remote.
    Existing Datasets are never overwritten or made public.
    """
    import re
    root = Path(pack_root)
    manifest = {'complete':True,'pack':pack,'assets':assets}
    names=[Path(a['file']).name for a in assets]
    if not assets or len(names)!=len(set(names)):
        raise RuntimeError('Empty pack or duplicate filenames; refusing upload.')
    for asset in assets:
        path = root/asset['file']
        if path.is_symlink() or not valid(path, asset):
            raise RuntimeError('Model failed verification before Dataset upload: '+Path(asset['file']).name)
    api = api if api is not None else dataset_api()
    owner = owner or getattr(api, 'config_values', {}).get('username', '') or os.environ.get('KAGGLE_USERNAME', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise RuntimeError('Set DATASET_OWNER to your Kaggle username. Keep API credentials in Kaggle Secrets, never in the notebook.')
    fingerprint = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()[:12]
    handle = owner+'/h3-next-'+pack+'-'+fingerprint
    url = 'https://www.kaggle.com/datasets/'+handle
    receipt = root.parent/('h3-next-'+pack+'-dataset-upload.json')
    if receipt.exists() and json.loads(receipt.read_text()).get('handle') == handle:
        print('Previously submitted private Dataset:', url, flush=True)
        return url
    stage = root.parent/('h3-next-'+pack+'-dataset-files')
    stage.mkdir(exist_ok=True)
    expected = {Path(a['file']).name for a in assets} | {'manifest.json', 'dataset-metadata.json'}
    if any(p.name not in expected for p in stage.iterdir()):
        raise RuntimeError('Unexpected upload-staging file preserved; inspect dataset-files before retrying.')
    for asset in assets:
        source = root/asset['file']
        target = stage/Path(asset['file']).name
        if target.exists():
            if target.is_symlink() or not valid(target, asset):
                raise RuntimeError('Existing upload-staging file failed verification; preserved: '+target.name)
        else:
            os.link(source, target)
    (stage/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (stage/'dataset-metadata.json').write_text(json.dumps({
        'id': handle, 'title': 'H3 Next '+pack+' '+fingerprint,
        'licenses': [{'name': 'other'}],
        'description': 'Private verified H3 Next model pack. Original weight licenses apply; no relicensing. See manifest.json for sources and SHA256 values.'
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


def dataset_session(owner='',api=None):
    # Fail before downloading many GiB if credentials or the username are absent.
    import re
    api=api if api is not None else dataset_api()
    owner=owner or getattr(api,'config_values',{}).get('username','') or os.environ.get('KAGGLE_USERNAME','')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',owner):
        raise RuntimeError('Set DATASET_OWNER to your Kaggle username. No model downloads started.')
    return api,owner


def select_pack_storage(pack, assets, automatic=True, candidates=None, headroom=512*1024**2):
    """Choose one filesystem that fits; never add free space across mounts.

    Temporary storage is valid only when uploading a named Dataset. Saved notebook
    outputs must remain in /kaggle/working. Existing models/partials stay in place.
    """
    import shutil
    if candidates is None:
        candidates=[Path('/kaggle/working/models')]
        if automatic:
            candidates += [Path('/kaggle/temp/h3-next-packs')/pack/'models',
                           Path('/tmp/h3-next-packs')/pack/'models']
    choices=[];reports=[]
    for value in candidates:
        root=Path(value)
        try:
            root.mkdir(parents=True,exist_ok=True)
            manifest=root.parent/'h3-pack-selection.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('pack')!=pack:
                reports.append(str(root)+': belongs to another pack; preserved')
                continue
            # Verify write access without touching any existing model files.
            import tempfile
            with tempfile.TemporaryFile(dir=root):pass
            remaining=0;reusable=0
            for asset in assets:
                path=root/asset['file'];partial=path.with_suffix(path.suffix+'.part')
                if path.exists():
                    if path.is_symlink() or path.stat().st_size!=asset['size']:
                        raise ValueError('Existing model has an invalid size/type; preserved: '+str(path))
                    reusable+=asset['size']  # downloader checks the full hash before use
                else:
                    present=partial.stat().st_size if partial.is_file() and not partial.is_symlink() else 0
                    if present>asset['size']:raise ValueError('Oversized partial file preserved: '+str(partial))
                    remaining+=asset['size']-present;reusable+=present
            free=shutil.disk_usage(root).free
            reports.append(f'{root}: {free/1024**3:.2f} GiB free; {remaining/1024**3:.2f} GiB still needed + {headroom/1024**3:.2f} GiB headroom')
            if free>=remaining+headroom:choices.append((reusable,free,root))
        except (OSError,ValueError) as error:
            reports.append(str(root)+': '+str(error))
    print('Storage checks:\n'+'\n'.join(reports),flush=True)
    if not choices:
        raise RuntimeError('No single writable location has enough free space for this pack. '
                           'No files were deleted and no downloads started. '+
                           ('Start a fresh CPU session with more available disk, or attach an existing Dataset.' if automatic else
                            'Notebook Outputs require working storage. Use AUTO_CREATE_DATASET=True to allow temporary storage and private Dataset upload.'))
    # Prefer resumable files, then the location with most available headroom.
    root=max(choices,key=lambda c:(c[0],c[1]))[2]
    print('Model download and upload staging location:',root,flush=True)
    return root
