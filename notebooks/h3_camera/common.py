"""Bounded, standard-library-only file and process operations for the kit."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request

GIB = 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def valid(path, asset):
    path = Path(path)
    return path.is_file() and path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def bounded(command, log, timeout, cwd=None, env=None):
    """The parent stays responsive even if a child blocks in a download or file read."""
    started = time.monotonic()
    with open(log, 'a', buffering=1) as stream:
        stream.write('\nCOMMAND: '+repr(list(map(str, command)))+'\n')
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed > timeout:
                    raise TimeoutError(f'Stage exceeded {timeout}s. Log: {log}')
                print(f'Stage running: {elapsed:.0f}s; log: {log}', flush=True)
                try:
                    process.wait(timeout=min(30, max(0.1, timeout-elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                with open(log, 'rb') as error_log:
                    error_log.seek(max(0, Path(log).stat().st_size-6000))
                    tail = error_log.read().decode(errors='replace')
                raise RuntimeError(f'Stage exited {process.returncode}. Log: {log}\n{tail}')
        finally:
            stop(process)


def transfer(asset, target, reserve=GIB):
    """Resumable verified download; caller additionally enforces a wall-time limit."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if valid(target, asset):
            return target
        raise RuntimeError('Existing file failed checksum; preserved: '+str(target))
    partial = target.with_suffix(target.suffix+'.partial')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == asset['bytes'] and valid(partial, asset):
            partial.replace(target)
            return target
        if offset >= asset['bytes']:
            raise RuntimeError('Invalid partial file preserved: '+str(partial))
        if shutil.disk_usage(target.parent).free < asset['bytes']-offset+reserve:
            raise RuntimeError('Not enough disk for '+target.name)
        request = urllib.request.Request(asset['source'], headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected download range')
                written = offset if resume else 0
                with partial.open('ab' if resume else 'wb') as destination:
                    for block in iter(lambda: response.read(8*1024**2), b''):
                        destination.write(block)
                        written += len(block)
                        if written > asset['bytes']:
                            raise RuntimeError('Download exceeded pinned file size')
            if not valid(partial, asset):
                raise RuntimeError('Download checksum mismatch: '+target.name)
            partial.replace(target)
            return target
        except OSError:
            if attempt == 2:
                raise


def choose_scratch(required, roots, folder, reusable=()):
    """No assumed Kaggle quota. Never spread weights silently across mounts."""
    candidates = []
    for root in roots:
        root = Path(root)
        if root.exists() and os.access(root, os.W_OK):
            free = shutil.disk_usage(root).free
            candidates.append((free, root))
    for free, root in sorted(candidates, key=lambda row: row[0], reverse=True):
        # Reuse capacity only for exact named files. Integrity is checked by the bounded staging child.
        credit = sum(size for relative, size in reusable if (root/folder/relative).is_file()
                     and (root/folder/relative).stat().st_size == size)
        needed = max(4*GIB, required-credit)
        print(f'Scratch candidate {root}: {free/GIB:.2f} GiB free; need {needed/GIB:.2f} GiB after reusable local files', flush=True)
        if free >= needed:
            target = root/folder
            if target.is_symlink():
                raise RuntimeError('Owned scratch path must not be a symlink: '+str(target))
            target.mkdir(exist_ok=True)
            return target
    raise RuntimeError(f'No local scratch filesystem has {required/GIB:.2f} GiB free. '
                       'No GPU models were loaded. Restart a clean session or use a runtime with enough local disk; '
                       'attached Dataset storage is not writable scratch.')


def stage_models(catalog, input_root, cache):
    """Copy directly while hashing; never run inference from a Dataset mount."""
    input_root, cache = Path(input_root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    assets = catalog['assets']
    names = {a['name'] for a in assets}
    index = {}
    for directory, _, files in os.walk(input_root):
        for name in set(files) & names:
            index.setdefault(name, []).append(Path(directory)/name)
    sources, missing = {}, []
    for asset in assets:
        target = cache/asset['file']
        if target.exists():
            if not valid(target, asset):
                raise RuntimeError('Local model checksum mismatch; preserved: '+str(target))
            sources[asset['name']] = target
        else:
            match = next((p for p in index.get(asset['name'], []) if p.stat().st_size == asset['bytes']), None)
            if match is None:
                missing.append(asset['name'])
            else:
                sources[asset['name']] = match
    if missing:
        raise RuntimeError('Attach the ONE saved JJP WanGP Motion Models Dataset. Missing files:\n'+'\n'.join(missing))
    pending = sum(a['bytes'] for a in assets if sources[a['name']] != cache/a['file'])
    if shutil.disk_usage(cache).free < pending+4*GIB:
        raise RuntimeError('Not enough local disk to stage all models plus a 4 GiB reserve. No GPU models loaded.')
    for asset in assets:
        target, source = cache/asset['file'], sources[asset['name']]
        if source == target:
            print('Verified local:', asset['name'], flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix+'.partial')
        digest, written = hashlib.sha256(), 0
        print('Staging and hashing:', asset['name'], flush=True)
        with source.open('rb') as incoming, partial.open('wb') as outgoing:
            for block in iter(lambda: incoming.read(8*1024**2), b''):
                outgoing.write(block)
                digest.update(block)
                written += len(block)
                if written > asset['bytes']:
                    raise RuntimeError('Input changed during copy: '+source.name)
        if written != asset['bytes'] or digest.hexdigest() != asset['sha256']:
            raise RuntimeError('Dataset file failed verification; partial preserved: '+str(partial))
        partial.replace(target)
    atomic_json(cache/'verified.json', catalog)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'stage':
        stage_models(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3], sys.argv[4])
