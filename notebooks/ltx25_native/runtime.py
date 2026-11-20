"""One-cell saved-Dataset startup for the pinned AIQUEST LTX-2.5 engine."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from ltx25_common import GIB, atomic_json, choose_scratch, stage_models

WAN_COMMIT = 'f02a38275fa43057ea75814e8ef4368e5530dcf2'
PACKAGES = ['comfy-kitchen==0.2.35', 'optimum-quanto==0.2.7', 'transformers==4.54.0',
            'diffusers==0.36.0', 'accelerate>=1.1.1', 'gradio==5.29.0', 'einops',
            'soundfile', 'scipy', 'huggingface_hub[hf_xet]<1.0', 'hf_xet>=1.5.2']


def find_dataset(catalog, root):
    for manifest in sorted(Path(root).rglob('manifest.json')):
        try:
            if json.loads(manifest.read_text()) != catalog:
                continue
            if all((manifest.parent/a['name']).is_file() and
                   (manifest.parent/a['name']).stat().st_size == a['bytes'] for a in catalog['assets']):
                return manifest.parent
        except (OSError, ValueError):
            continue
    raise RuntimeError('Attach the ONE complete Dataset created by JJP_LTX25_Models.ipynb. No model downloads or GPU loads started.')


def check_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or len(rows) != 2 or any('T4' not in row[0] for row in rows):
        raise RuntimeError('Select Accelerator GPU T4 x2 for this test notebook.')
    for i, row in enumerate(rows):
        name, free, total = row[0].strip(), float(row[1]), float(row[2])
        print(f'GPU {i}: {name}; {free/1024:.2f} GiB free / {total/1024:.2f} GiB', flush=True)
        if free < 12000:
            raise RuntimeError('An older app is using GPU memory. Stop it first; no unrelated process was killed.')


def stop_owned(work):
    import psutil
    receipt = Path(work)/'engine-process.json'
    if not receipt.exists():
        return
    item = json.loads(receipt.read_text())
    try:
        parent = psutil.Process(item['pid'])
        expected = str((Path(work)/'run_ltx25.py').resolve())
        if parent.create_time() != item['created'] or expected not in parent.cmdline():
            raise RuntimeError('Previous process identity changed; it was preserved.')
        processes = parent.children(recursive=True)+[parent]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=10)
    except psutil.NoSuchProcess:
        pass


def checkout(repo):
    repo = Path(repo)
    if not (repo/'.git').exists():
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin',
                        'https://github.com/deepbeepmeep/Wan2GP.git'], check=True)
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode:
        subprocess.run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', WAN_COMMIT], check=True, timeout=180)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    elif head.stdout.strip() != WAN_COMMIT:
        raise RuntimeError('Existing Wan2GP checkout differs; preserved. Use a clean LTX work directory.')
    # The native engine applies idempotent reviewed numerical patches here. Do not reset unrelated files.


def budget_reason(uptime, progress, startup_seconds, job_seconds):
    if progress is None:
        return 'Engine startup exceeded its limit.' if uptime > startup_seconds else None
    if progress.get('active') and float(progress.get('elapsed', 0)) > job_seconds:
        return 'Generation exceeded its per-clip time limit.'
    return None


def run(runtime, work, startup_seconds=1200, job_seconds=1800):
    import psutil
    runtime, work = Path(runtime), Path(work)
    if startup_seconds < 60 or job_seconds < 60:
        raise ValueError('Startup/job limits must be at least 60 seconds.')
    work.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((runtime/'assets.json').read_text())
    dataset = find_dataset(catalog, '/kaggle/input')
    stop_owned(work)
    check_gpus()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 7860)) == 0:
            raise RuntimeError('Port 7860 is in use. Its existing app was preserved.')
    cache = choose_scratch(sum(a['bytes'] for a in catalog['assets'])+6*GIB,
                           ['/kaggle/temp', '/tmp', '/kaggle/working'], 'jjp-ltx25-models',
                           [(a['file'], a['bytes']) for a in catalog['assets']])
    stage_models(catalog, dataset, cache)
    checkout(work/'Wan2GP')
    # Keep Kaggle's working Torch/CUDA stack, as in the source notebook. Constrain it against pip replacement.
    import importlib.metadata
    protected = []
    for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 'triton'):
        try:
            protected.append(name+'=='+importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            pass
    constraints = work/'base-constraints.txt'
    constraints.write_text('\n'.join(protected)+'\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--constraint', str(constraints),
                    '--timeout', '60', '--retries', '2', *PACKAGES], check=True, timeout=600)
    engine = (runtime/'engine.py').read_bytes()
    provenance = json.loads((runtime/'provenance.json').read_text())
    if hashlib.sha256(engine).hexdigest() != provenance['engine_sha256']:
        raise RuntimeError('Embedded engine checksum differs from build provenance.')
    script = work/'run_ltx25.py'
    script.write_bytes(engine)
    env = dict(os.environ, LTX25_MODE='t2v', LTX25_MODELS_DIR=str(cache), LTX25_TENSOR_PARALLEL='1',
               LTX25_DTYPE='fp16', MPLBACKEND='Agg', GRADIO_ANALYTICS_ENABLED='False',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    atomic_json(work/'provenance.json', dict(provenance, catalog=catalog, wangp_commit=WAN_COMMIT))
    print('Starting AIQUEST native LTX-2.5: text / start image / first+last frame, generated audio, eight steps.', flush=True)
    print('No clip is generated automatically. Upload a Start frame in the interface, then press Generate.', flush=True)
    print(f'Limits: {startup_seconds}s engine startup; {job_seconds}s per clip. These are stop limits, not speed promises.', flush=True)
    with (work/'engine.log').open('a', buffering=1) as log:
        proc = subprocess.Popen([sys.executable, '-u', str(script)], cwd=work, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(work/'engine-process.json', {'pid': proc.pid, 'created': psutil.Process(proc.pid).create_time()})
        finished = threading.Event()
        began = time.monotonic()
        def watchdog():
            ready, last_seen = False, began
            while not finished.wait(2) and proc.poll() is None:
                progress = None
                try:
                    with urllib.request.urlopen('http://127.0.0.1:7860/aiq_progress', timeout=2) as response:
                        progress = json.load(response)
                    ready, last_seen = True, time.monotonic()
                except (OSError, ValueError):
                    if ready and time.monotonic()-last_seen <= 60:
                        continue
                    if ready:
                        print('Local app stopped responding for 60s; stopping only this owned engine.', flush=True)
                        stop_owned(work)
                        return
                reason = budget_reason(time.monotonic()-began, progress, startup_seconds, job_seconds)
                if reason:
                    print(reason+' Completed clips are retained.', flush=True)
                    stop_owned(work)
                    return
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            while True:
                block = os.read(proc.stdout.fileno(), 65536)
                if not block:
                    break
                text = block.decode(errors='replace')
                log.write(text)
                print(text, end='', flush=True)
            if proc.wait():
                raise RuntimeError('LTX engine exited. See '+str(work/'engine.log'))
        finally:
            finished.set()
            stop_owned(work)
