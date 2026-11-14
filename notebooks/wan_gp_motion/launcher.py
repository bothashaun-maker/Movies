"""Kaggle supervisor. No model is loaded until the user presses Generate in WanGP."""
from pathlib import Path
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from common import GIB, atomic_json, bounded, choose_scratch, stop, transfer, valid
from settings import MODEL, server_settings, shot_settings

WANGP_REVISION = '38758d3c73366422efcf609abb63cab8385f3552'
CF = dict(source='https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/cloudflared-linux-amd64',
          bytes=40274445, sha256='1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872')


def memory():
    return {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:'))}


def hardware_preflight():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free,driver_version',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    if not rows or 'T4' not in rows[0][1]:
        raise RuntimeError('Select Kaggle GPU T4 x2. This notebook uses GPU 0 only.')
    driver = tuple(int(s) for s in rows[0][3].strip().split('.')[:2])
    if driver < (570, 26):
        raise RuntimeError('Pinned CUDA 12.8 requires NVIDIA driver 570.26 or newer. No CUDA packages installed.')
    if float(rows[0][2]) < 13000:
        raise RuntimeError('GPU 0 needs at least 13,000 MiB free. Stop the other studio or start a fresh session.')
    ram = memory()
    if ram['MemTotal'] < 24*GIB or ram['MemAvailable'] < 20*GIB:
        raise RuntimeError('Need at least 24 GiB total and 20 GiB available host RAM before startup.')
    print(result.stdout.strip(), flush=True)
    print('GPU 0 only; GPU 1 remains unused. Memory is not pooled.', flush=True)
    return {'gpus': result.stdout.strip(), 'ram': ram}


def monitor_health(active, ram, now, limit, low_count):
    if active.get('active') and now-float(active['started']) > limit:
        return 'Generation exceeded its hard wall-time limit.', low_count
    low_count = low_count+1 if ram['MemAvailable'] < int(1.5*GIB) else 0
    return ('Available host RAM stayed below 1.5 GiB for three checks.' if low_count >= 3 else None), low_count


def headless_environment(parent=None):
    """The child is a standalone server, so it must not inherit Jupyter's inline backend."""
    env = dict(os.environ if parent is None else parent)
    env['MPLBACKEND'] = 'Agg'
    return env


def native_command(python, app, scratch, config, settings, port):
    """One command definition for the shipped launcher and CPU GUI checks."""
    return [python, '-u', Path(app)/'wgp.py', '--offline', '--gpu', 'cuda:0', '--profile', '5',
            '--fp16', '--attention', 'sdpa', '--vram-allocator', 'default', '--ram-allocator', 'default',
            '--preload', '0', '--perc-reserved-mem-max', '0.10', '--max-vram', '14',
            '--max-ram', '24', '--vae-config', '3', '--lock-config',
            '--config', config, '--settings', settings, '--loras', Path(scratch)/'loras',
            '--server-name', '127.0.0.1', '--server-port', str(port), '--no-mcp-webui', '--advanced']


def startup_error(process, log_path):
    """Keep the actual Python traceback/exit code in the notebook output on failure."""
    log_path = Path(log_path)
    try:
        with log_path.open('rb') as stream:
            stream.seek(max(0, log_path.stat().st_size-10000))
            tail = stream.read().decode(errors='replace')
    except OSError as error:
        tail = f'Could not read the startup log: {error}'
    return RuntimeError(f'WanGP GUI exited with code {process.poll()}. Log: {log_path}\n'
                        f'Last startup output:\n{tail}')


def launch(runtime, output, port=7860, job_minutes=20, session_hours=8, scratch_root='', speed_mode='sdpa'):
    from h3_attention import speed_preflight
    speed_preflight(speed_mode)
    runtime, output = Path(runtime), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output/'studio.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This kit is already running. Stop its earlier cell before rerunning.')
    children = []
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
        hardware = hardware_preflight()
        # Native WanGP imports audio/vision helpers, even with soundtrack inputs disabled.
        # Kaggle normally supplies these libraries; install missing prerequisites in a bounded stage.
        import ctypes.util
        needed = [package for program, package in [('ffmpeg', 'ffmpeg'), ('ffprobe', 'ffmpeg'), ('g++', 'build-essential')]
                  if not shutil.which(program)]
        if not ctypes.util.find_library('portaudio'):
            needed.append('libportaudio2')
        if not ctypes.util.find_library('GL'):
            needed.append('libgl1')
        if needed:
            if os.geteuid() != 0 or not shutil.which('apt-get'):
                raise RuntimeError('Kaggle image is missing system packages: '+', '.join(sorted(set(needed))))
            bounded(['apt-get', '-qq', 'update'], output/'system-packages.log', 180)
            bounded(['apt-get', '-qq', 'install', '-y', *sorted(set(needed))], output/'system-packages.log', 300)
        catalog = json.loads((runtime/'assets.json').read_text())
        model_bytes = sum(a['bytes'] for a in catalog['assets'])
        roots = [scratch_root] if scratch_root else ['/kaggle/temp', '/tmp', '/kaggle/working']
        # A complete local mirror avoids the Dataset-read stalls found in Director.
        # Full requirement is explicit: 40.54 GiB models + 8 GiB environment + 4 GiB reserve.
        reusable = [('models/'+a['file'], a['bytes']) for a in catalog['assets']]
        scratch = choose_scratch(model_bytes+12*GIB, roots, 'jjp-wangp-motion-v1', reusable)
        cache = scratch/'models'
        env = headless_environment()
        env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   NUMBA_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='60',
                   PIP_NO_CACHE_DIR='1', UV_NO_CACHE='1', UV_PYTHON_INSTALL_DIR=str(scratch/'python'),
                   PYTHONPATH=str(runtime), JJP_OUTPUT=str(output), GRADIO_ANALYTICS_ENABLED='False')
        atomic_json(output/'startup.json', {'hardware': hardware, 'scratch': scratch,
                                          'model_bytes': model_bytes, 'revision': WANGP_REVISION})
        stage_command = [sys.executable, runtime/'common.py', 'stage', runtime/'assets.json', '/kaggle/input', cache]
        if speed_mode != 'sage':
            bounded(stage_command, output/'models.log', 1200, env=env)
        from bootstrap import ensure_environment
        python, uv = ensure_environment(scratch, output, env)
        app = scratch/'WanGP'
        if not app.exists():
            bounded(['git', 'clone', '--no-checkout', 'https://github.com/deepbeepmeep/WanGP.git', app],
                    output/'checkout.log', 300, env=env)
            bounded(['git', '-C', app, 'checkout', '--detach', WANGP_REVISION], output/'checkout.log', 60, env=env)
        revision = subprocess.check_output(['git', '-C', str(app), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
        if revision != WANGP_REVISION:
            raise RuntimeError('Unexpected WanGP revision; directory preserved: '+str(app))
        marker = scratch/'dependencies.ready'
        from common import sha256
        identity = sha256(runtime/'requirements.lock')+'-torch210cu128-py31113'
        if not marker.exists() or marker.read_text() != identity:
            bounded([uv, 'pip', 'install', '--python', python, 'torch==2.10.0', 'torchvision==0.25.0',
                     'torchaudio==2.10.0', '--index-url', 'https://download.pytorch.org/whl/cu128'],
                    output/'install-torch.log', 1200, env=env)
            bounded([uv, 'pip', 'install', '--python', python, '-r', runtime/'requirements.lock'],
                    output/'install-app.log', 1800, env=env)
            bounded([uv, 'pip', 'check', '--python', python], output/'install-check.log', 60, env=env)
            marker.write_text(identity)
        # This probe must pass on the actual T4 before opening the GUI.
        env['PYTHONPATH'] = str(runtime)+os.pathsep+str(app)
        from patch import patch_wgp, patch_h3_dtype, patch_h3_numerics, patch_h3_attention
        patch_wgp(app/'wgp.py')
        patch_h3_dtype(app/'models/minimax_h3/minimax_h3_handler.py')
        patch_h3_numerics(app)
        patch_h3_attention(app)
        from h3_attention import choose
        env['JJP_H3_ATTENTION'] = choose(runtime, output, python, uv, app, env, speed_mode)
        bounded([python, runtime/'gpu_probe.py'], output/'gpu-probe.log', 120, cwd=app, env=env)
        if speed_mode == 'sage':
            bounded(stage_command, output/'models.log', 1200, env=env)
        print('H3 attention:', env['JJP_H3_ATTENTION'], '| Qwen and VAE: SDPA. Numerical safeguards retained.', flush=True)
        config = output/'config'
        settings = output/'settings'
        config.mkdir(exist_ok=True)
        settings.mkdir(exist_ok=True)
        atomic_json(config/'wgp_config.json', server_settings(cache, output))
        atomic_json(settings/(MODEL+'_settings.json'), shot_settings())
        atomic_json(output/'A-baseline-settings.json', shot_settings())
        atomic_json(output/'B-guided-settings.json', shot_settings(True))
        # The auto-selected PDD LoRA lives in WanGP's family LoRA directory, not ckpts/loras.
        loras = scratch/'loras'/'minimax_h3'
        loras.mkdir(parents=True, exist_ok=True)
        lora_name = 'minimax_h3_ref2va_pdd_8step_lora_bf16.safetensors'
        link = loras/lora_name
        if not link.exists():
            link.symlink_to(cache/'loras'/lora_name)
        if link.resolve() != (cache/'loras'/lora_name).resolve():
            raise RuntimeError('Unexpected existing LoRA link; preserved.')
        cloudflared = scratch/'cloudflared'
        # Download the small pinned utility in a bounded child, not model weights in the GPU studio.
        bounded([sys.executable, runtime/'bootstrap.py', 'cloudflare', cloudflared], output/'cloudflare-install.log', 180, env=env)
        cloudflared.chmod(0o700)
        atomic_json(output/'active-job.json', {'active': False})
        command = native_command(python, app, scratch, config, settings, port)
        with open(output/'wangp.log', 'a', buffering=1) as log:
            server = subprocess.Popen(list(map(str, command)), cwd=app, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        children.append(server)
        local_url = f'http://127.0.0.1:{port}'
        deadline = time.monotonic()+300
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise startup_error(server, output/'wangp.log')
            try:
                with urllib.request.urlopen(local_url+'/config', timeout=3) as response:
                    info = json.load(response)
                if info.get('components'):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('WanGP GUI readiness exceeded 5 minutes. See wangp.log; no inference started.')
        with open(output/'cloudflare.log', 'w', buffering=1) as log:
            tunnel = subprocess.Popen([str(cloudflared), 'tunnel', '--url', local_url, '--protocol', 'http2',
                                       '--no-autoupdate', '--metrics', '127.0.0.1:0'],
                                      env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(tunnel)
        deadline, url = time.monotonic()+120, None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError('Cloudflare stopped; see cloudflare.log.')
            tunnel_log = (output/'cloudflare.log').read_text()
            matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', tunnel_log)
            if matches:
                url = matches[-1]
                if 'Registered tunnel connection' in tunnel_log:
                    try:
                        with urllib.request.urlopen(url+'/config', timeout=5) as response:
                            remote_info = json.load(response)
                        if remote_info.get('components'):
                            break
                    except (OSError, ValueError):
                        pass
            time.sleep(2)
        else:
            raise TimeoutError('Cloudflare public GUI readiness exceeded 2 minutes. See cloudflare.log; no ready link was announced.')
        atomic_json(output/'connection.json', {'url': url, 'local': local_url})
        print('\nJJP WanGP Motion Test Kit is ready. Open:', url, flush=True)
        print('Upload ONE appearance image. Generate A. Then choose ONE Reference Video, upload the silent Blender guide and generate B.', flush=True)
        print(f'Keep this cell running. No automatic render. Per-shot hard limit: {job_minutes} min.', flush=True)
        print('Video, settings and diagnostics:', output, flush=True)
        started, low_count, last_report = time.monotonic(), 0, 0
        while time.monotonic()-started < session_hours*3600:
            if server.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError('WanGP or Cloudflare stopped. Logs and completed outputs are preserved.')
            active = json.loads((output/'active-job.json').read_text())
            ram = memory()
            reason, low_count = monitor_health(active, ram, time.time(), job_minutes*60, low_count)
            if reason:
                atomic_json(output/'watchdog-stop.json', {'reason': reason, 'active': active, 'ram': ram})
                raise RuntimeError(reason+' WanGP is being stopped to release its CUDA context. Completed files are preserved.')
            if shutil.disk_usage(output).free < GIB or shutil.disk_usage(scratch).free < GIB:
                raise RuntimeError('Local free disk fell below 1 GiB. Stopping before further output writes.')
            if time.monotonic()-last_report > 30:
                print(f'Alive | {active.get("label", "idle") if active.get("active") else "idle"} | RAM available {ram["MemAvailable"]/GIB:.1f} GiB', flush=True)
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                     timeout=5, stdin=subprocess.DEVNULL, check=True).stdout.strip()
                with open(output/'resources.jsonl', 'a') as report:
                    report.write(json.dumps({'time': time.time(), 'job': active, 'ram': ram, 'gpus': gpu})+'\n')
                last_report = time.monotonic()
            time.sleep(5)
        print('Session time limit reached. Completed outputs are preserved.', flush=True)
    finally:
        cleanup_errors = []
        for process in reversed(children):
            try:
                stop(process)
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(str(error))
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        if cleanup_errors:
            print('Owned process cleanup could not be confirmed:', '; '.join(cleanup_errors), flush=True)


if __name__ == '__main__':
    launch(Path(__file__).parent, Path(os.environ.get('JJP_OUTPUT', '/kaggle/working/jjp-wangp-motion')),
           int(os.environ.get('JJP_PORT', '7860')), int(os.environ.get('JJP_JOB_MINUTES', '20')),
           int(os.environ.get('JJP_SESSION_HOURS', '8')), os.environ.get('JJP_SCRATCH', ''))
