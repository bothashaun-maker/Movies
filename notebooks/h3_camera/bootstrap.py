"""One-cell Dataset wiring around AIQuest's retained engine and interface."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import socket
from camera_common import GIB,atomic_json,choose_scratch,stage_models

def find_dataset(catalog,input_root=Path('/kaggle/input')):
    # Do not assemble a runtime from a mix of unrelated attached packs.
    for path in sorted(Path(input_root).rglob('manifest.json')):
        try:match=json.loads(path.read_text())==catalog
        except (OSError,ValueError):continue
        if match:
            missing=[a['name'] for a in catalog['assets'] if not (path.parent/a['name']).is_file() or (path.parent/a['name']).stat().st_size!=a['bytes']]
            if not missing:return path.parent
    raise RuntimeError('Attach the ONE saved JJP H3 Camera Models Dataset created by JJP_H3_Camera_Models.ipynb. No models were downloaded or loaded.')

def stop_camera_engine(work):
    import psutil
    receipt=Path(work)/'engine-process.json'
    if not receipt.is_file():return
    item=json.loads(receipt.read_text())
    try:
        process=psutil.Process(item['pid'])
        if abs(process.create_time()-item['created'])>.01:return
        if Path(process.cwd()).resolve() not in (Path(work).resolve(),(Path(work)/'Wan2GP').resolve()):return
        if not any(Path(a).name=='run_h3.py' for a in process.cmdline()):return
        owned=process.children(recursive=True)+[process]
        for child in owned:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(owned,timeout=15)
        for child in alive:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(alive,timeout=10)
    except psutil.NoSuchProcess:pass

def remember_camera_engine(work,process):
    import psutil
    try:created=psutil.Process(process.pid).create_time()
    except psutil.NoSuchProcess:return  # Let AIQuest's reader expose the child's startup error.
    atomic_json(Path(work)/'engine-process.json',{'pid':process.pid,'created':created})

def gpu_names():
    result=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    names=result.stdout.strip().splitlines()
    if result.returncode or len(names)!=2 or not all('T4' in n for n in names):
        raise RuntimeError('Select GPU T4 x2 for the AIQuest renderer.')
    return names

def main(runtime,work=Path('/kaggle/working/jjp-h3-camera')):
    runtime,work=Path(runtime),Path(work)
    catalog=json.loads((runtime/'assets.json').read_text())
    dataset=find_dataset(catalog)
    gpu_names()
    # Same AIQuest host Python/dependency method; do not introduce a new CUDA,
    # attention implementation or encoder into its published inference code.
    try:import psutil
    except ModuleNotFoundError:
        subprocess.run([sys.executable,'-m','pip','install','psutil'],check=True)
    stop_camera_engine(work)
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',7860))==0:
            raise RuntimeError('Port 7860 is already in use. Its app was left untouched; stop it before starting this notebook.')
    total=sum(a['bytes'] for a in catalog['assets'])
    cache=choose_scratch(total+8*GIB,[Path('/kaggle/temp'),Path('/tmp'),Path('/kaggle/working')],
        'jjp-h3-camera-models',[(a['file'],a['bytes']) for a in catalog['assets']])
    stage_models(catalog,dataset,cache)
    work.mkdir(parents=True,exist_ok=True)
    os.chdir(work)
    settings={'H3_VARIANT':'pdmd','H3_MODELS_DIR':str(cache/'h3_models'),'H3_LORA_DIR':str(cache/'h3_loras'),
        'H3_DIT_FILE':'MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors','H3_TE_FILE':'qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf',
        'H3_TURBO_LORA':'lora_model_0.safetensors'}
    os.environ.update(settings)
    atomic_json(work/'h3_settings.json',settings)
    # Source-author code remains in its two original stages, run consecutively.
    scope={'__name__':'__main__','_CAMERA_WORK':work,'stop_camera_engine':stop_camera_engine,
        'remember_camera_engine':remember_camera_engine,'RUN_SCRIPT_CODE':(runtime/'aiquest_engine.py').read_text(encoding='utf-8'),
        'codecs':__import__('codecs'),'re':__import__('re')}
    exec(compile((runtime/'aiquest_setup.py').read_text(encoding='utf-8'),'AIQuest setup','exec'),scope)
    exec(compile((runtime/'aiquest_launcher.py').read_text(encoding='utf-8'),'AIQuest launcher','exec'),scope)
