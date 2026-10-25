"""Build the standalone Kaggle notebook from reviewable Python sources."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks/h3_complete'

def read(name):
    return (SOURCE/(name+'.py')).read_text()

def code(source, name, hidden=False):
    compile(source, name, 'exec')
    # Cell compilation cannot detect syntax errors inside Python -c payloads.
    # Check the embedded programs after assembling the final cell as well.
    programs = {'probe', 'kernel_probe', 'metadata_probe', 'script', 'code',
                'compat_source', 'checkpoint_source'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in programs and isinstance(node.value.value, str):
                    compile(node.value.value, f'{name}:{target.id}', 'exec')
    return nbformat.v4.new_code_cell(source, id=name, metadata={
        'jupyter':{'source_hidden':hidden}, 'tags':['implementation'] if hidden else []})

def md(source, name):
    return nbformat.v4.new_markdown_cell(source, id=name)

install = 'STACK_READY=False\n'+read('bootstrap')+'\n'+read('stack')
install += '\ncheckpoint_source = '+repr(read('checkpoint_nodes'))+'\n'
install += "checkpoint_plugin=COMFY/'custom_nodes/H3CheckpointRecovery'\ncheckpoint_plugin.mkdir(parents=True,exist_ok=True)\n(checkpoint_plugin/'__init__.py').write_text(checkpoint_source)\nSTACK_READY=True\nprint('Installation and both T4 kernel probes passed.')\n"
# The browser node invokes a separate controller. Its GPU workers use another
# port and never stop the persistent browser server.
worker = read('settings')+'\n'+read('assets')+'\n'+read('graphs')+'\n'+read('runtime')
worker += '''
job=json.loads(Path(sys.argv[1]).read_text())
PROMPT=job['prompt'];SEED=job['seed'];MEGAPIXELS=job['megapixels'];DURATION=job['duration']
ASPECT=tuple(job['aspect']);IMAGE_PATH=job['image_path'];MODE='image' if IMAGE_PATH else 'text'
MODELS_READY=True;STACK_READY=True
PORT+=1;API=f'http://127.0.0.1:{PORT}'
def stop_owned_servers():
    pass  # Only comfy_process owns worker PIDs; never stop the browser server.
def run(args,**kwargs):
    kwargs.setdefault('env',child_env)
    return subprocess.run([str(a) for a in args],check=True,**kwargs)
path=render_clip()
Path(job['result']).write_text(json.dumps({'file':str(path)}))
'''
compile(worker,'browser_worker','exec')
install += '\nbrowser_node_source = '+repr(read('browser_node'))+'\n'
install += 'browser_worker_source = '+repr(worker)+'\n'
install += read('browser_workflow')
install += '''
browser_plugin=COMFY/'custom_nodes/H3KaggleBrowser'
browser_plugin.mkdir(parents=True,exist_ok=True)
(browser_plugin/'__init__.py').write_text(browser_node_source)
# Carry the selected family/bridge into the isolated controller at runtime.
browser_worker_source=browser_worker_source.replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(ROOT)!r})')
worker_settings = f"PORT={PORT!r};MODEL_FAMILY={MODEL_FAMILY!r};USE_BRIDGE={USE_BRIDGE!r};BRIDGE_ALPHA={BRIDGE_ALPHA!r};"
browser_worker_source=browser_worker_source.replace("job=json.loads(Path(sys.argv[1]).read_text())",worker_settings+"job=json.loads(Path(sys.argv[1]).read_text())")
worker_path=ROOT/'h3_browser_worker.py';worker_path.write_text(browser_worker_source)
(browser_plugin/'config.json').write_text(json.dumps({'root':str(ROOT),'python':str(PY),'worker':str(worker_path)}))
workflow_dir=COMFY/'user/default/workflows'
workflow_dir.mkdir(parents=True,exist_ok=True)
(workflow_dir/'MiniMax H3 - Kaggle T4.json').write_text(json.dumps(browser_workflow(PROMPT,SEED,MEGAPIXELS,DURATION),indent=2))
print('Installed ComfyUI workflow: MiniMax H3 - Kaggle T4')
'''
# Wrap statements, not text lines: indenting a multiline string changes the
# program passed to Python -c even though the enclosing notebook still compiles.
weight_tree = ast.parse("""def prepare_models():
    global MODELS_READY
    assert STACK_READY, 'The installation stage must complete successfully before rendering.'
    try:
        pass
    except BaseException:
        MODELS_READY=False
        raise

prepare_models()
""")
guard = next(node for node in weight_tree.body[0].body if isinstance(node, ast.Try))
guard.body = ast.parse(read('weights')).body
weights = ast.unparse(ast.fix_missing_locations(weight_tree))+'\n'
# There is exactly one executable notebook cell. A cell's Play button and Run All
# therefore both execute the entire workflow, without relying on frontend queues.
stages = {
    'catalog':read('processes')+'\n'+read('assets'),
    'install':install,
    'weights':weights,
    'workflow':read('graphs')+'\n'+read('runtime'),
    'access':read('cloudflare'),
}
for name, source in stages.items():
    code(source, name)  # Also validate embedded Python before packaging the stages.
from pprint import pformat
entry = read('settings')
entry += "\n# Embedded runtime below. Edit the settings above; no other cells need to be started.\n"
entry += '_H3_PIPELINE_SOURCES = '+pformat(stages, sort_dicts=False, width=110)+'\n\n'
entry += read('pipeline')+"\nrun_pipeline(_H3_PIPELINE_SOURCES, globals())\n"
nb = nbformat.v4.new_notebook(cells=[
    md('''# MiniMax H3 — open ComfyUI through Cloudflare

Select **GPU T4 ×2** and **Internet ON**, then press the single code cell's **▶ button once**. It installs the pinned runtime, verifies existing or downloaded models, starts ComfyUI and prints **Your ComfyUI Cloudflare link**. Click **Open ComfyUI in browser**, then open **Workflows → MiniMax H3 - Kaggle T4** and press **Run**. The workflow is installed automatically; no JSON download is needed.

This notebook does not queue a render, embed video previews, or copy video downloads. Work in ComfyUI through the link. Keep the Kaggle session running; the temporary public link expires when its tunnel or session stops.

Existing weights in the same session are reused. The first download is roughly 41.38 GiB and uses network, disk and CPU. Logs are saved as `comfyui-ui.log` and `cloudflared.log` under `/kaggle/working/minimax_h3_dual_t4`. The link is also saved as `comfyui-cloudflare-url.txt`.''','intro'),
    code(entry,'run'),
    md('''## Runtime and workflow notes

Uses Python 3.12.12, PyTorch 2.10 / CUDA 13, pinned ComfyUI and the dual-T4 tensor-parallel fork. Both GPU kernel probes run before missing weights download. CUDA 13 requires driver 580 or newer. Four-step Turbo LoRA weights are included; Semantic Bridge is optional for FL2VA. FastVideo VSA kernels are not enabled for T4.

The adapted **MiniMax H3 - Kaggle T4** workflow is installed in ComfyUI’s workflow menu. Edit the prompt and seed, then press Run. For image input, add a Load Image node and connect it to the render node’s image input. The original supplied canvas remains a reference. Browser workflows must use nodes compatible with this pinned stack. Avoid the old dual-helper VAE configuration that exhausted GPU memory. The existing isolated sampling/decoding controller remains available in the embedded runtime, but is not started automatically and may restart ComfyUI when explicitly invoked.

The uploaded run successfully completed a 124-frame, 512 × 768 video with audio using that controller. The new browser-access path has local lifecycle checks; a live Cloudflare connection from your Kaggle session still needs to be established there.

To close access without ending the Kaggle session, call `stop_comfyui_access()`. No stop cell is queued automatically by Run All.''','details'),
], metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12.12'},
             'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'}})
nbformat.validate(nb)
for filename in ('minimax_h3_kaggle_complete.ipynb','minimax_h3_cloudflare_fixed.ipynb'):
    output=ROOT/filename
    nbformat.write(nb, output)
    print(output)

workflow_scope={}
exec(read('browser_workflow'),workflow_scope)
settings_tree=ast.parse(read('settings'))
prompt=next(ast.literal_eval(n.value) for n in settings_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in n.targets))
import json
workflow_path=ROOT/'workflows/minimax_h3_kaggle_t4.json'
workflow_path.parent.mkdir(exist_ok=True)
workflow_path.write_text(json.dumps(workflow_scope['browser_workflow'](prompt),indent=2)+'\n')
