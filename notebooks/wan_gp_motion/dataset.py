"""One private Dataset, using the existing Image Studio native streaming-upload method."""
from pathlib import Path
import fcntl
import hashlib
import inspect
import json
import os
import re
import tempfile
from common import GIB, atomic_json, choose_scratch, transfer, valid


def authenticated_api():
    # Follow the working Image Studio credential flow. Never print/store credential values.
    if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
        except Exception:
            pass
    if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            os.environ['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
        except Exception:
            pass
    # Resolve Kaggle Secrets in the parent notebook: its module is not in this isolated env.
    from check_dataset_runtime import check_runtime
    KaggleApi = check_runtime()
    api = KaggleApi()
    api.authenticate()
    return api


def create_dataset(catalog, output, input_root, cache, owner='', api=None):
    output, input_root, cache = Path(output), Path(input_root), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    lock = open(output/'dataset.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = None
    try:
        api = api or authenticated_api()  # Authenticate before downloading any model.
        owner = owner or api.config_values.get('username', '') or os.environ.get('KAGGLE_USERNAME', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
            raise RuntimeError('Set DATASET_OWNER to your Kaggle username; keep credentials in Kaggle Secrets.')
        fingerprint = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()[:12]
        handle = owner+'/jjp-wangp-motion-'+fingerprint
        url = 'https://www.kaggle.com/datasets/'+handle
        receipt = output/'dataset-upload.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('handle') == handle:
            print('Previously submitted private Dataset:', url, flush=True)
            return url
        original = api.upload_files
        signature = inspect.signature(original)
        if not {'folder', 'request'} <= set(signature.parameters):
            raise RuntimeError('Installed Kaggle uploader signature differs from the reviewed native method. No downloads started.')
        sources = {}
        for asset in catalog['assets']:
            for candidate in input_root.rglob(asset['name']) if input_root.exists() else ():
                if valid(candidate, asset):
                    sources[asset['name']] = candidate.resolve()
                    break
        uploaded_all = False

        def stream_upload(*args, **kwargs):
            nonlocal uploaded_all
            request = signature.bind(*args, **kwargs).arguments['request']

            def send(folder):
                bound = signature.bind(*args, **kwargs)
                bound.arguments['folder'] = str(folder)
                return original(*bound.args, **bound.kwargs)

            uploaded = []
            for asset in catalog['assets']:
                source = sources.get(asset['name'])
                owned = source is None
                if owned:
                    source = transfer(asset, cache/asset['name'])
                if not valid(source, asset):
                    raise RuntimeError('Model checksum mismatch before upload: '+asset['name'])
                with tempfile.TemporaryDirectory(dir=output, prefix='.one-file-') as directory:
                    # The SDK uploads actual bytes through the link. Never upload link metadata or zip weights.
                    (Path(directory)/asset['name']).symlink_to(source.resolve())
                    count = len(request.files)
                    print('Uploading:', asset['name'], flush=True)
                    send(directory)
                    if len(request.files) != count+1:
                        raise RuntimeError('Native uploader did not return a file token: '+asset['name'])
                uploaded.append(asset['name'])
                atomic_json(output/'upload-progress.json', {'handle': handle, 'uploaded': uploaded, 'dataset_created': False})
                if owned:
                    source.unlink()  # Only this creator's successfully uploaded temporary download.
            with tempfile.TemporaryDirectory(dir=output, prefix='.manifest-') as directory:
                atomic_json(Path(directory)/'manifest.json', catalog)
                send(directory)
            uploaded_all = True

        api.upload_files = stream_upload
        with tempfile.TemporaryDirectory(dir=output, prefix='.request-') as directory:
            atomic_json(Path(directory)/'dataset-metadata.json', {
                'id': handle, 'title': 'JJP WanGP Motion Models '+fingerprint,
                'licenses': [{'name': 'other'}],
                'description': 'Private pinned model files for JJP WanGP Motion Test Kit. Original model licenses apply. See manifest.json for sources, revision, exact bytes and SHA256 checksums.'})
            result = api.dataset_create_new(directory, public=False, quiet=False, convert_to_csv=False, dir_mode='skip')
        field = lambda key: result.get(key) if isinstance(result, dict) else getattr(result, key, None)
        if not uploaded_all or result is None or field('error') or field('error_message') or str(field('status')).lower() in ('failed', 'failure', 'error'):
            raise RuntimeError('Dataset creation was not confirmed. Uploaded-file tokens are not a saved Dataset.')
        atomic_json(receipt, {'handle': handle, 'url': url, 'submitted': True, 'requested_private': True})
        atomic_json(output/'manifest.json', catalog)
        print('PRIVATE DATASET SUBMITTED:', url, flush=True)
        print('Wait for Kaggle processing; attach this ONE Dataset to JJP_WanGP_Motion_Test_Kit.ipynb via Add Input → Your Datasets.', flush=True)
        return url
    except BaseException:
        atomic_json(output/'dataset-error.json', {'confirmed': False, 'sources_preserved': True,
                                               'message': 'See CPU notebook log; no success receipt written. No credentials logged.'})
        raise
    finally:
        if original is not None:
            api.upload_files = original
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    import sys
    create_dataset(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], sys.argv[3], sys.argv[4], os.environ.get('JJP_DATASET_OWNER', ''))
