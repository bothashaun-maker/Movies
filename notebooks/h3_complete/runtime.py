from contextlib import contextmanager
import socket


def api_request(route, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API+route, data=body,
        headers={'Content-Type':'application/json'} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'ComfyUI HTTP {error.code}: {error.read().decode()[:8000]}') from error


def tail_log(path, limit=10000):
    if not path.exists():
        return '(no log file)'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        return stream.read().decode(errors='replace')


def check_node_catalog(info, graphs):
    for graph in graphs:
        for node_id, node in graph.items():
            kind = node['class_type']
            if kind not in info:
                raise RuntimeError(f'Required node did not load: {kind}')
            specs = {**info[kind]['input'].get('required',{}), **info[kind]['input'].get('optional',{})}
            for key, value in node['inputs'].items():
                if isinstance(value,list) and len(value) == 2:
                    assert value[0] in graph, f'Dangling link: {node_id}.{key}'
                elif key in specs:
                    spec = specs[key]
                    choices = spec[0] if isinstance(spec[0],list) else (
                        spec[1].get('options') if len(spec)>1 and isinstance(spec[1],dict) else None)
                    if choices and isinstance(choices[0],dict):
                        choices = [option['key'] for option in choices]
                    if choices is not None and value not in choices:
                        raise RuntimeError(f'Unavailable {kind}.{key}: {value}')


@contextmanager
def comfy_process(phase, directory, graphs):
    stop_owned_servers()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError(f'Port {PORT} is used by another service. Change PORT in settings.')
    log_path = directory/(phase+'.log')
    flags = ['--cuda-device','all','--vram-headroom',str(VRAM_HEADROOM_GB),
             '--preview-method','none','--disable-auto-launch']
    flags += ['--use-ck-attention'] if USE_KITCHEN_ATTENTION else ['--use-pytorch-cross-attention']
    with log_path.open('a', buffering=1) as log:
        log.write(f'\n--- {phase} started {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())} ---\n')
        process = subprocess.Popen([str(PY),'-u','main.py','--listen','127.0.0.1',
            '--port',str(PORT),*flags], cwd=COMFY, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('ComfyUI startup failed:\n'+tail_log(log_path))
                try:
                    api_request('/system_stats', timeout=3)
                    break
                except (OSError,RuntimeError):
                    time.sleep(2)
            else:
                raise TimeoutError('ComfyUI startup timed out:\n'+tail_log(log_path))
            check_node_catalog(api_request('/object_info'), graphs)
            # The pinned fork must activate its protected FP16 path on both T4s.
            if 'fp16 mixed precision enabled' not in tail_log(log_path, 100000):
                raise RuntimeError('H3 FP16 patch did not activate. See '+str(log_path))
            yield process, log_path
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            if process.poll() is None:
                raise RuntimeError('ComfyUI did not exit; the next GPU stage is blocked.')
            print(f'{phase.capitalize()} process exited; its CUDA allocations have been released.', flush=True)


def execute_graph(graph, process, log_path, directory, phase):
    started = time.monotonic()
    queued = api_request('/prompt', {'prompt':graph, 'client_id':uuid.uuid4().hex})
    if 'prompt_id' not in queued or queued.get('node_errors'):
        raise RuntimeError('Workflow validation failed: '+json.dumps(queued)[:8000])
    prompt_id = queued['prompt_id']
    (directory/(phase+'-job.json')).write_text(json.dumps(queued, indent=2))
    deadline = started+TIMEOUT_MINUTES*60
    next_report = started
    log_offset = log_path.stat().st_size
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{phase} process exited unexpectedly:\n'+tail_log(log_path))
        history = api_request('/history/'+prompt_id)
        entry = history.get(prompt_id)
        if entry:
            (directory/(phase+'-history.json')).write_text(json.dumps(entry, indent=2))
            status = entry.get('status',{})
            errors = [detail for kind,detail in status.get('messages',[]) if kind in ('execution_error','execution_interrupted')]
            if errors or status.get('status_str') == 'error':
                details = []
                for error in errors:
                    details.append(f"{error.get('node_type','node')} ({error.get('node_id','?')}): "
                                   f"{error.get('exception_type','interrupted')}: {error.get('exception_message','')}")
                    details.extend(error.get('traceback',[]))
                message = '\n'.join(details) or json.dumps(status)
                (directory/(phase+'-error.txt')).write_text(message)
                raise RuntimeError(message+'\nFull log: '+str(log_path))
            if status.get('completed'):
                print(f'{phase.capitalize()} completed in {(time.monotonic()-started)/60:.1f} min.', flush=True)
                return entry
        if time.monotonic() >= next_report:
            print(f'{phase.capitalize()}: {(time.monotonic()-started)/60:.1f} min elapsed', flush=True)
            with log_path.open('rb') as stream:
                stream.seek(log_offset)
                recent = stream.read()
                log_offset = stream.tell()
            if recent:
                print(recent[-1800:].decode(errors='replace').replace('\r','\n'), flush=True)
            next_report = time.monotonic()+30
        time.sleep(2)
    raise TimeoutError(f'{phase} exceeded {TIMEOUT_MINUTES} minutes. Log: {log_path}')


def validate_checkpoint(run_id):
    path = COMFY/'output/h3_latents'/(run_id+'.safetensors')
    if not path.is_file():
        return False
    code = '''import sys,torch
from safetensors import safe_open
with safe_open(sys.argv[1],framework='pt',device='cpu') as f:
 assert f.metadata().get('format')=='h3-joint-latent-v1','Wrong checkpoint format'
 assert set(f.keys())=={'video','audio'}
 v,a=f.get_tensor('video'),f.get_tensor('audio')
 assert v.ndim==5 and v.shape[0]==1 and v.shape[1]==24
 assert a.ndim==4 and a.shape[0]==1 and a.shape[1]==32 and a.shape[2]==2
 assert v.shape[-2:]==(int(sys.argv[3])//16,int(sys.argv[2])//16)
 assert torch.isfinite(v).all() and torch.isfinite(a).all(),'Non-finite checkpoint'
 print('Verified saved latents:',list(v.shape),list(a.shape))
'''
    width, height = canvas_size(MEGAPIXELS, ASPECT)
    run([PY,'-c',code,path,width,height])
    return True


def video_paths(entry):
    paths = []
    for output in entry.get('outputs',{}).values():
        for value in output.values():
            if not isinstance(value,list):
                continue
            for item in value:
                if not isinstance(item,dict) or not item.get('filename','').lower().endswith('.mp4'):
                    continue
                if item.get('type','output') != 'output':
                    continue
                path = (COMFY/'output'/item.get('subfolder','')/item['filename']).resolve()
                if not path.is_relative_to((COMFY/'output').resolve()):
                    raise RuntimeError('Unexpected output path from ComfyUI.')
                if path.is_file():
                    paths.append(path)
    return list(dict.fromkeys(paths))


def verify_video(path, report_path):
    # Decode the entire short clip: file existence alone is not success.
    code = '''import sys,json,av
with av.open(sys.argv[1]) as f:
 assert len(f.streams.video)==1 and len(f.streams.audio)>=1,'Missing video or audio stream'
 stream=f.streams.video[0]
 assert stream.width==int(sys.argv[3]) and stream.height==int(sys.argv[4])
 frames=sum(1 for _ in f.decode(video=0))
 assert frames==int(sys.argv[5]),f'Wrong frame count: {frames}'
with av.open(sys.argv[1]) as f:
 samples=sum(frame.samples for frame in f.decode(audio=0))
 assert samples>0,'Empty audio stream'
report={'file':sys.argv[1],'frames':frames,'width':int(sys.argv[3]),'height':int(sys.argv[4]),'audio_samples':samples,'fps':24}
open(sys.argv[2],'w').write(json.dumps(report,indent=2))
print('MP4 verified:',json.dumps(report))
'''
    width, height = canvas_size(MEGAPIXELS, ASPECT)
    run([PY,'-c',code,path,report_path,width,height,frame_count(DURATION)])
    return json.loads(report_path.read_text())


def render_clip():
    assert STACK_READY, 'Run the installation cell successfully first.'
    require_models_ready()
    run_id, directory, sample, decode = prepare_job()
    print('Run:', run_id, '|', canvas_size(MEGAPIXELS, ASPECT), '|', frame_count(DURATION), 'frames', flush=True)
    report_path = directory/'result.json'
    # Repeated Run All does not spend GPU time on an already finished identical clip.
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        path = Path(previous['file'])
        if path.is_file():
            verify_video(path, report_path)
            print('Reusing completed clip. Change the seed or prompt for a new render.')
            return path
    try:
        if validate_checkpoint(run_id):
            print('Saved sampling result found; going directly to decoding.', flush=True)
        else:
            with comfy_process('sampling', directory, [sample,decode]) as (process, log_path):
                execute_graph(sample, process, log_path, directory, 'sampling')
            # This executes AFTER process.wait(): the sampler cannot retain VRAM during decode.
            if not validate_checkpoint(run_id):
                raise RuntimeError('Sampling returned without its checkpoint; refusing to decode.')
        print('Starting decoding in a fresh process.', flush=True)
        with comfy_process('decoding', directory, [decode]) as (process, log_path):
            entry = execute_graph(decode, process, log_path, directory, 'decoding')
        paths = video_paths(entry)
        if len(paths) != 1:
            raise RuntimeError(f'Expected one MP4, found {len(paths)}. See decoding-history.json.')
        verify_video(paths[0], report_path)
        return paths[0]
    except BaseException as error:
        (directory/'last-error.txt').write_text(type(error).__name__+': '+str(error))
        print('Stopped. Diagnostics:', directory, flush=True)
        if (COMFY/'output/h3_latents'/(run_id+'.safetensors')).is_file():
            print('Latents are saved. Rerun this render cell to retry decoding without resampling.', flush=True)
        raise
