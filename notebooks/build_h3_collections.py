"""Build four named entry points; Pixa and Director embed their complete canvases."""
from pathlib import Path
import runpy,copy,json,base64,zipfile,io
import nbformat
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'notebooks/h3_complete';U=ROOT/'notebooks/h3_suite'
base=runpy.run_path(str(ROOT/'notebooks/build_h3_complete.py'))
read=lambda name:(S/(name+'.py')).read_text()
u=lambda name:(U/(name+'.py')).read_text()
canvas_scope={};exec(u('canvas'),canvas_scope)
source=ROOT/'workflows/sources'
media=(source/'pixa_example_media.zip').read_bytes()
common_assets={}
# Asset catalog only; definitions require runtime settings, but no file access beyond ROOT.
exec("from pathlib import Path\nimport json,hashlib\nROOT=Path('/tmp/h3-suite-catalog');ROOT.mkdir(exist_ok=True)\n"+read('assets')+'\n'+u('assets'),common_assets)
assets=common_assets['SUITE_ASSETS']
collections={
 'Pixa_5_Workflows':[
 ('Pixa - Text to Video','Minimax H3 - Text to Video + Speed Lora.json'),
 ('Pixa - Image to Video','Minimax H3 - Image to video FF + Speed Lora.json'),
 ('Pixa - First and Last Frame','Minimax H3 - Image to video FFLF + Speed Lora.json'),
 ('Pixa - Speaking Audio','Minimax H3 - Reference Image + Audio Sync - SPEAK + Speed Lora.json'),
 ('Pixa - Singing Audio','Minimax H3 - Reference Image + Audio Sync - SING + Speed Lora.json')],
 'Director':[
 ('Director - Three Step','TA_MiniMax_H3_3Step_Director_v2.json'),
 ('Director - PDD Eight Step','MiniMax H3 Director - PDD8.json')]
}
for collection,items in collections.items():
    canvases={title:canvas_scope['adapt_canvas'](json.loads((source/file).read_text()),title,collection=='Director') for title,file in items}
    for title,g in canvases.items():
        for n in g['nodes']:
            if n['type']=='PixaromaLoadImageMini' and n['widgets_values'][0]=='BunnyMan.jpeg':n['widgets_values'][0]='WomanPortraitRed.png'
            if n['type']=='PixaromaLoadAudio' and 'Speaking' in title:n['properties']['loadAudioState']['file']=''
    generated=ROOT/'workflows'/collection;generated.mkdir(exist_ok=True)
    for title,g in canvases.items():(generated/(title+'.json')).write_text(json.dumps(g,indent=2)+'\n')
    plugin=read('diagnostics')+'\n'+read('checkpoint_nodes')+'\nCHECKPOINT_NODE_MAPPINGS=NODE_CLASS_MAPPINGS.copy()\n'+read('browser_node')+'\n'+u('nodes')
    worker=read('settings')+'\n'+read('assets')+'\n'+u('assets')+'\n'+read('graphs')+'\n'+read('runtime')+'\n'+read('browser_job')+'\n'+u('preflight')+'\n'+u('job')+'''
job=json.loads(Path(sys.argv[1]).read_text())
MODELS_READY=True;STACK_READY=True
PORT+=1;API=f'http://127.0.0.1:{PORT}'
def stop_owned_servers():pass
def run(args,**kwargs):
    kwargs.setdefault('env',child_env)
    return subprocess.run([str(a) for a in args],check=True,**kwargs)
path=run_browser_job(job)
Path(job['result']).write_text(json.dumps({'file':str(path)}))
'''
    compile(plugin,'suite-plugin','exec');compile(worker,'suite-worker','exec')
    installation='STACK_READY=False\n'+read('bootstrap')+'\n'+read('stack')+'''
pinned_checkout('https://github.com/Pixaroma/ComfyUI-Pixaroma.git','88260de112af70a25f82edcd6ee2d31dccff8f13',COMFY/'custom_nodes/ComfyUI-Pixaroma')
pinned_checkout('https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director.git','5846ce8c3dac33cbdc629c2e01606e2a740f563c',COMFY/'custom_nodes/ComfyUI-MiniMaxH3-Director')
pinned_checkout('https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc.git','311a65dd53832d8a5f8177a9d5fb923c09e35a90',COMFY/'custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc')
run([PY,'-m','pip','install','-c',constraints,'imageio-ffmpeg==0.6.0','opencv-python-headless==4.11.0.86'])
# Read core module source for PDD compatibility; the T4 precision wrapper replaces forward().
pdd=COMFY/'custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc/nodes.py'
src=pdd.read_text();old='inspect.getsource(MiniMaxH3Model.forward)';new='inspect.getsource(__import__(MiniMaxH3Model.__module__, fromlist=["MiniMaxH3Model"]))'
if old in src:src=src.replace(old,new);pdd.write_text(src)
else:assert new in src,'Unexpected PDD source.'
'''
    installation+='\nplugin_source='+repr(plugin)+'\nworker_source='+repr(worker)+'\ncanvases='+repr(canvases)+'\ncollection='+repr(collection)+'\nmedia_archive='+repr(base64.b64encode(media).decode())+'\n'
    installation+='''
plugin_dir=COMFY/'custom_nodes/H3KaggleBrowser';plugin_dir.mkdir(parents=True,exist_ok=True)
(plugin_dir/'__init__.py').write_text(plugin_source)
worker_source=worker_source.replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(ROOT)!r})')
worker_path=ROOT/'h3_collection_worker.py';worker_path.write_text(worker_source)
(plugin_dir/'config.json').write_text(json.dumps({'root':str(ROOT),'python':str(PY),'worker':str(worker_path),'assets':SUITE_ASSETS}))
# Separate recovery plugin may exist from the working notebook; identical public classes are harmless.
web=plugin_dir/'web';web.mkdir(exist_ok=True)
'''
    js=(S/'open_workflow.js').read_text();start=js.index('            if (!(');end=js.index('            await app.loadGraphData',start)
    js=js[:start]+'''            if (!graph.nodes?.some(node => node.type === "H3SuiteSaveVideo")) throw new Error("Collection save node missing.");
'''+js[end:]
    installation+="(web/'open_workflow.js').write_text("+repr(js)+")\n"
    installation+='''
workflow_dir=COMFY/'user/default/workflows';workflow_dir.mkdir(parents=True,exist_ok=True)
for title,canvas in canvases.items():(workflow_dir/(title+'.json')).write_text(json.dumps(canvas))
(web/'full-workflow.json').write_text(json.dumps(next(iter(canvases.values()))))
import zipfile,io,base64
input_dir=COMFY/'input';input_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(media_archive))) as archive:
    for name in archive.namelist():
        assert Path(name).name==name,'Unsafe example media path'
        target=input_dir/name
        if not target.exists():target.write_bytes(archive.read(name))
STACK_READY=True
print('Installed workflows: '+', '.join(canvases),flush=True)
'''
    # Prepare the first workflow; other families are fetched only when selected for Run.
    default_names=(['minimax_h3_fl2va_pruned_int8_convrot.safetensors','qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','minimax_h3_audio_vae_fp32.safetensors']+
                   (['minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors','minimax_h3_video_vae_fp16.safetensors'] if collection=='Pixa_5_Workflows' else ['minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors','minimax_h3_video_vae_int8_convrot.safetensors']))
    weights='suite_prepare_assets([a for a in SUITE_ASSETS if Path(a["file"]).name in '+repr(default_names)+'])\n'
    access=read('cloudflare').replace("assert 'H3IsolatedSaveVideo' in api_request('/object_info')", "assert 'H3SuiteSaveVideo' in api_request('/object_info')").replace("print('In ComfyUI: Workflows → MiniMax H3 - Full T4 Workflow → Run',flush=True)","print('In ComfyUI, select a workflow from: '+', '.join(canvases),flush=True)")
    stages={'catalog':read('processes')+'\n'+read('assets')+'\n'+u('assets'),'install':installation,'weights':weights,'workflow':read('graphs')+'\n'+read('runtime'),'access':access}
    for key,text in stages.items():compile(text,key,'exec')
    from pprint import pformat
    entry=read('settings')+'\n_H3_PIPELINE_SOURCES = '+pformat(stages,sort_dicts=False,width=110)+'\n'+read('pipeline')+'\nrun_pipeline(_H3_PIPELINE_SOURCES,globals())\n'
    nb=copy.deepcopy(base['nb']);nb.cells[1]=base['code'](entry,'run')
    names='\n'.join('- **'+title+'**' for title,_ in items)
    nb.cells[0].source=f'''# H3 — {collection.replace('_',' ')} · Kaggle T4 ×2

Select **T4 ×2** and **Internet ON**, then run the single code cell once. Open the Cloudflare link; select these installed workflows from ComfyUI's Workflows sidebar:

{names}

The real custom-node interfaces are installed. Model loading and final saving are adapted for dual-T4 sampling and separate decoding. No automatic render or notebook video preview. Upscaling stays in CapCut.

Default size is 512×768 (or the visible small preset), up to 124 frames at 24 fps. Upload your own image/audio in its node; Speaking Audio requires your recording because its sample WAV was unreadable in the supplied archive. Director opens with a clean five-second timeline; add reference cards and switch Refs ON to use reference conditioning.

The first workflow's weights are prepared during setup. Switching model families may download another ~19.53 GiB checkpoint. When space is insufficient, only tracked notebook-downloaded model caches may be reclaimed; input datasets, uploaded media, latents and outputs are preserved. Attach both model families as Kaggle inputs to avoid repeat family downloads. Storage is measured, not assumed.

**GPU rendering of these integrated collections has not been measured here.** Local checks validate schemas, browser editing and worker partitioning; they do not guarantee model numerics or performance.
'''
    nb.cells[2].source='''## Session and outputs

Keep Kaggle running while using ComfyUI. Completed MP4s appear at the final Save node and in `ComfyUI/output`. Use **Open render diagnostics** for the selected job's logs. Changing controls and reference/audio inputs changes the job fingerprint. Sampling and decoding remain separate processes; supplied speech/song audio is saved across that boundary rather than re-encoding it in the decoder process.

Do not run two notebook setups concurrently on port 8188. Starting another collection reuses the pinned runtime and restarts this notebook's browser process.
'''
    nbformat.validate(nb);nbformat.write(nb,ROOT/(collection+'.ipynb'))
# Named aliases requested by the user; preserve original notebooks for existing links.
current=copy.deepcopy(base['nb']);current.cells[0].source=current.cells[0].source.replace('# MiniMax H3 — FULL-GRAPH-v1 · 17 nodes · 19 links','# Current Working Workflow — MiniMax H3 · T4 ×2')
nbformat.write(current,ROOT/'Current_Working_Workflow.ipynb')
three=nbformat.read(ROOT/'minimax_h3_taomate3_experimental.ipynb',as_version=4);three.cells[0].source=three.cells[0].source.replace('# MiniMax H3 — EXPERIMENTAL TaoMate 3-step + INT8 VAE · T4 ×2','# Three-Step Workflow — experimental TaoMate + INT8 VAE · T4 ×2')
nbformat.write(three,ROOT/'Three_Step_Workflow.ipynb')
