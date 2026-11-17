"""Run the pinned TeamAIQ FlashVSR engine, retaining its inference implementation."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

AIQ_COMMIT = '237383fb084b671c1eb5224db822d88fca6fc8fa'
AIQ_BLOB = '17e92df9535e0b3d7d386431712e4a759113c76d'
AIQ_URL = f'https://raw.githubusercontent.com/TeamAIQ/Colab-notebooks/{AIQ_COMMIT}/kaggle_notebooks/flashvsr-video-upscaler.ipynb'
FLASH_COMMIT = 'b548a2cc2c929d3d4676f2973abe27876ec4dee7'

def extract_engine(raw):
    actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if actual != AIQ_BLOB:
        raise RuntimeError('TeamAIQ source identity differs from the reviewed notebook. Refusing to run it.')
    nb = json.loads(raw)
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id=='RUN_SCRIPT_CODE' for n in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError('Pinned notebook has no native FlashVSR engine.')

def inference_identity(source):
    # Every function/class is retained, including kernels, pipelining, loading,
    # splitting/rebalancing, media IO, progress, callbacks and numerical guards.
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)

class PresentationDefaults(ast.NodeTransformer):
    """Change only initial UI choices; never modify a function or class."""
    def visit_FunctionDef(self, node): return node
    def visit_AsyncFunctionDef(self, node): return node
    def visit_ClassDef(self, node): return node
    def visit_Assign(self, node):
        names = {n.id for n in node.targets if isinstance(n, ast.Name)}
        if names == {'SIZE_CHOICES'}:
            choices = ast.literal_eval(node.value)
            choices.remove('2x upscale'); choices.insert(0, '2x upscale')
            node.value = ast.parse(repr(choices), mode='eval').body
        if isinstance(node.value, ast.Call):
            desired = False if names=={'compare_cb'} else 14 if names=={'crf_slider'} else None
            if desired is not None:
                found = False
                for kw in node.value.keywords:
                    if kw.arg == 'value': kw.value=ast.Constant(desired); found=True
                if not found: raise RuntimeError('Upstream UI default changed unexpectedly.')
        return self.generic_visit(node)

def prepare_engine(source):
    tree = PresentationDefaults().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree)+'\n'
    if inference_identity(result) != inference_identity(source):
        raise RuntimeError('Native inference changed during UI adaptation.')
    compile(result, 'run_flashvsr.py', 'exec')
    return result

def stop_owned(work):
    import psutil
    receipt = Path(work)/'engine-process.json'
    if not receipt.exists(): return
    record = json.loads(receipt.read_text())
    try:
        parent = psutil.Process(record['pid'])
        command = parent.cmdline()
        expected = str((Path(work)/'run_flashvsr.py').resolve())
        if parent.create_time()!=record['created'] or expected not in command:
            raise RuntimeError('Previous process identity differs; it was preserved.')
        children = parent.children(recursive=True)
        for process in reversed(children+[parent]):
            try: process.terminate()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(children+[parent], timeout=10)
        for process in alive:
            try: process.kill()
            except psutil.NoSuchProcess: pass
        psutil.wait_procs(alive, timeout=10)
    except psutil.NoSuchProcess:
        pass

def checkout(repo):
    repo = Path(repo)
    if not (repo/'.git').exists():
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','init','-q',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'remote','add','origin','https://github.com/OpenImagingLab/FlashVSR_Plus.git'],check=True)
    head = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
    if head.returncode:
        subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',FLASH_COMMIT],check=True)
        subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True)
    elif head.stdout.strip()!=FLASH_COMMIT:
        raise RuntimeError('Existing FlashVSR checkout differs; preserved.')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('FlashVSR code contains local edits; preserved.')

def find_dataset(catalog, root):
    for manifest in Path(root).rglob('manifest.json'):
        try:
            if json.loads(manifest.read_text()).get('assets')!=catalog['assets']:continue
            names={p.name for p in manifest.parent.rglob('*') if p.is_file()}
            if all(a['name'] in names for a in catalog['assets']):return manifest.parent
        except (OSError,ValueError):continue
    raise RuntimeError('Attach the ONE complete JJP FlashVSR model Dataset. SeedVR2/H3 model packs are different; no model downloads started.')

def check_gpus():
    # NVML query only: do not create parent CUDA contexts beside the engine.
    text=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.total',
                                  '--format=csv,noheader,nounits'],text=True)
    rows=[line.split(',') for line in text.splitlines() if line.strip()]
    if len(rows)<2:
        raise RuntimeError('Select GPU T4 ×2. No weights were downloaded or loaded.')
    for device,row in enumerate(rows[:2]):
        name,free,total=row[0].strip(),float(row[1]),float(row[2])
        print(f'GPU {device}: {name}; {free/1024:.2f} GiB free / {total/1024:.2f} GiB',flush=True)
        if free<10240:
            raise RuntimeError('Stop the older H3/SeedVR2 session first. No unrelated process was killed.')

def budget_reason(uptime, progress, startup_limit, job_limit):
    if progress is None:
        return 'Engine startup exceeded the time budget.' if uptime>startup_limit else None
    if progress.get('active') and float(progress.get('elapsed',0))>job_limit:
        return 'Upscaling exceeded the per-clip time budget.'
    return None

def run(runtime, work, max_job_seconds=480, max_startup_seconds=360):
    from flash_common import GIB, atomic_json, choose_scratch, stage_models
    import psutil
    runtime, work = Path(runtime), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stop_owned(work)
    check_gpus()
    catalog = json.loads((runtime/'assets.json').read_text())
    # Check attachment before installs/staging; inference never downloads weights.
    input_root=find_dataset(catalog,'/kaggle/input')
    cache = choose_scratch(sum(a['bytes'] for a in catalog['assets'])+4*GIB,
                           ['/kaggle/temp','/tmp','/kaggle/working'], 'jjp-flashvsr-models',
                           [(a['file'],a['bytes']) for a in catalog['assets']])
    stage_models(catalog,input_root,cache)
    repo = cache.parent/'jjp-flashvsr-code'
    checkout(repo)
    # Same application packages as TeamAIQ. Retain Kaggle's own Torch/CUDA/Triton.
    packages=['gradio==5.29.0','huggingface_hub[hf_xet]<1.0','safetensors>=0.5','einops>=0.8','imageio-ffmpeg>=0.6']
    try: import cv2
    except ImportError: packages.append('opencv-python-headless>=4.10')
    subprocess.run([sys.executable,'-m','pip','install','--quiet',*packages],check=True)
    upstream = work/'teamaiq-flashvsr.ipynb'
    if not upstream.exists():
        with urllib.request.urlopen(AIQ_URL,timeout=60) as response:
            raw=response.read(2*1024**2)
        extract_engine(raw)  # verify before caching or executing
        upstream.write_bytes(raw)
    original = extract_engine(upstream.read_bytes())
    engine = prepare_engine(original)
    script = work/'run_flashvsr.py'; script.write_text(engine)
    atomic_json(work/'provenance.json',{'teamaiq_commit':AIQ_COMMIT,'teamaiq_blob':AIQ_BLOB,
                'flashvsr_commit':FLASH_COMMIT,'native_inference_ast_unchanged':True,
                'original_engine_sha256':hashlib.sha256(original.encode()).hexdigest(),
                'engine_sha256':hashlib.sha256(engine.encode()).hexdigest()})
    env = dict(os.environ,FLASHVSR_REPO=str(repo),FLASHVSR_MODELS=str(cache),FLASHVSR_WORK_DIR=str(work),
               MPLBACKEND='Agg',GRADIO_ANALYTICS_ENABLED='False',PYTHONWARNINGS='ignore')
    env.pop('FLASHVSR_ENGINE_TEST',None)
    print('Starting TeamAIQ native FlashVSR: resident FP16 weights, two-GPU pipeline, compact decoder.',flush=True)
    print('Initial choices: Turbo / 2× / normal motion / CRF 14 / veryfast / comparison OFF.',flush=True)
    print('Use the Input video upload box, then Upscale Video. Real phase timings and GPU timings are printed.',flush=True)
    print(f'Time limits: {max_startup_seconds}s engine startup and {max_job_seconds}s per clip. These are stop limits, not speed claims.',flush=True)
    with (work/'engine.log').open('a',buffering=1) as log:
        proc=subprocess.Popen([sys.executable,'-u',str(script)],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(work/'engine-process.json',{'pid':proc.pid,'created':psutil.Process(proc.pid).create_time()})
        done=threading.Event()
        started=time.monotonic()
        ready=[False]
        last_seen=[started]
        def watch_budget():
            while not done.wait(2) and proc.poll() is None:
                progress=None
                try:
                    with urllib.request.urlopen('http://127.0.0.1:7860/aiq_progress',timeout=2) as response:
                        progress=json.load(response)
                    ready[0]=True
                    last_seen[0]=time.monotonic()
                except (OSError,ValueError):
                    if ready[0]:
                        if time.monotonic()-last_seen[0]>30:
                            print('Local engine progress became unreachable for 30s; stopping owned engine to protect the budget.',flush=True)
                            stop_owned(work)
                            return
                        continue
                reason=budget_reason(time.monotonic()-started,progress,max_startup_seconds,max_job_seconds)
                if reason:
                    print(reason+' Stopping only this engine and its owned children; completed videos are retained.',flush=True)
                    stop_owned(work)
                    return
        threading.Thread(target=watch_budget,daemon=True).start()
        try:
            while True:
                data=os.read(proc.stdout.fileno(),65536)
                if not data:break
                text=data.decode(errors='replace');log.write(text);print(text,end='',flush=True)
            if proc.wait():raise RuntimeError('FlashVSR stopped; see '+str(work/'engine.log'))
        finally:
            done.set()
            stop_owned(work)
