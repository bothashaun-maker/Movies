"""Dataset-only integration of a hash-verified AIQuest notebook snapshot."""
import ast
import copy
import hashlib
import json
from pathlib import Path

UPSTREAM_BLOB='c0cb628304f4e736760c5db09db2ab53136faf4d'

def replace_node(source,node,replacement):
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def replace_function(source,name,replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return replace_node(source,node,replacement)

def sources(snapshot):
    raw=Path(snapshot).read_bytes().replace(b'\r\n',b'\n')
    actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if actual!=UPSTREAM_BLOB:raise RuntimeError('AIQuest source snapshot differs from the reviewed Git blob.')
    nb=json.loads(raw)
    setup=''.join(nb['cells'][2]['source'])
    wrapper=''.join(nb['cells'][3]['source'])
    assignment=next(n for n in ast.parse(wrapper).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUN_SCRIPT_CODE' for t in n.targets))
    original=ast.literal_eval(assignment.value)
    # Keep the complete AIQuest tensor-parallel engine, numerical protections,
    # original Gradio interface, prompt builder and persistent model residency.
    engine=original
    catalog=next(n for n in ast.parse(engine).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LORA_CATALOG' for t in n.targets))
    choices=ast.literal_eval(catalog.value)
    choices={k:v for k,v in choices.items() if v['repo'] in ('Jojocodex/minimax-h3-Camera-Motion-lora','pablodawson/MiniMax-H3-360-Orbit-LoRA')}
    if len(choices)!=2:raise RuntimeError('AIQuest camera catalog changed.')
    engine=replace_node(engine,catalog,'LORA_CATALOG = '+repr(choices))
    engine=replace_function(engine,'fetch_lora','''def fetch_lora(entry_or_url):
    """Only the two verified camera adapters from the saved Dataset; no GPU-time downloads."""
    if not isinstance(entry_or_url,dict):
        raise ValueError('Use the saved camera/orbit adapter, or upload your own local LoRA. URL downloads are disabled in this Dataset edition.')
    path=os.path.join(LORA_DIR,entry_or_url['file'])
    if not os.path.isfile(path):raise FileNotFoundError('Saved Dataset adapter is missing: '+entry_or_url['file'])
    return path
''')
    # Enforce the promise that inference was not rewritten: only this catalog and
    # its local-file resolver may differ from the upstream module's AST.
    clean=lambda s:[ast.dump(n,include_attributes=False) for n in ast.parse(s).body if not (
        isinstance(n,ast.FunctionDef) and n.name=='fetch_lora' or
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LORA_CATALOG' for t in n.targets))]
    if clean(original)!=clean(engine):raise RuntimeError('Unexpected change to AIQuest inference/UI code.')
    setup=setup[:setup.index('# ---- 3. Model weights')]
    setup=replace_function(setup,'stop_old_engine','''def stop_old_engine():
    stop_camera_engine(_CAMERA_WORK)
''')
    setup=setup.replace('MODELS_DIR = "/kaggle/tmp/h3_models" if os.path.isdir("/kaggle") else os.path.abspath("h3_models")','MODELS_DIR = os.environ["H3_MODELS_DIR"]')
    setup=setup.replace('LORA_DIR = "/kaggle/tmp/h3_loras" if os.path.isdir("/kaggle") else os.path.abspath("h3_loras")','LORA_DIR = os.environ["H3_LORA_DIR"]')
    wrapper=wrapper[wrapper.index('with open("run_h3.py"'):]
    wrapper=replace_function(wrapper,'stop_old_engine','''def stop_old_engine():
    stop_camera_engine(_CAMERA_WORK)
''')
    tree=ast.parse(wrapper)
    launch=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='proc' for t in n.targets))
    lines=wrapper.splitlines(keepends=True)
    lines.insert(launch.end_lineno,"remember_camera_engine(_CAMERA_WORK,proc)\n")
    wrapper=''.join(lines)
    for name,s in [('setup',setup),('engine',engine),('launcher',wrapper)]:compile(s,name,'exec')
    return setup,engine,wrapper,hashlib.sha256(original.encode()).hexdigest()
