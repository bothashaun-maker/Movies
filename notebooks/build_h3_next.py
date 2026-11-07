"""Build the separate H3 Next renderer and its reusable CPU model packs."""
from pathlib import Path
import copy,json,tempfile,hashlib
from pprint import pformat
import nbformat
R=Path(__file__).resolve().parents[1];B=R/'notebooks/h3_complete';S=R/'notebooks/h3_suite';N=R/'notebooks/h3_next'
r=lambda p:p.read_text()
new=json.loads(r(N/'assets.json'))
with tempfile.TemporaryDirectory() as t:
    scope={'Path':Path,'json':json,'hashlib':hashlib,'ROOT':Path(t)};exec(r(B/'assets.py'),scope)
shared=[a.copy() for a in scope['ASSETS'] if Path(a['file']).name in ['minimax_h3_video_vae_int8_convrot.safetensors','minimax_h3_audio_vae_fp32.safetensors']]
for a in shared:a['pack']='support'
assets=new+shared
pins={'comfy':'1332fabfcc507aa7c548cd5152071ed494c207e8','molmo':'72b3ed2a1a881a1548f8ed0b8a89fff939fb0216','dials':'6e47483e26ca086af240464721eff15f4bf76522'}
settings=r(B/'settings.py').replace('/kaggle/working/minimax_h3_dual_t4','/kaggle/working/h3_next_t4').replace('73c9bad4d21e7addbe1d13bc92eee0f1431b017d',pins['comfy']).replace('# v0.37.0','# H3 Next pinned Sol-compatible revision')
settings=settings.replace("SAMPLING_PRESET = 'turbo4'", "SAMPLING_PRESET = 'turbo4'")
# Reuse storage and download verification without changing the old notebooks/catalog.
asset_code=r(B/'assets.py')+'\n'+r(S/'assets.py')+'\nASSETS = '+repr(assets)+'\nSUITE_ASSETS = ASSETS\n'
asset_code+='\ndef selected_model_assets():\n    return [a for a in SUITE_ASSETS if a.get("pack")=="molmo" or a.get("pack")=="singularity" or Path(a["file"]).name in '+repr(['pika_h3_flex_turbo_4step_ref2va.safetensors','minimax_h3_sol_vae_int8_convrot.safetensors','minimax_h3_video_vae_int8_convrot.safetensors','minimax_h3_audio_vae_fp32.safetensors'])+']\n'
plugin=r(B/'diagnostics.py')+'\n'+r(B/'checkpoint_nodes.py')+'\nCHECKPOINT_NODE_MAPPINGS=NODE_CLASS_MAPPINGS.copy()\n'+r(B/'browser_node.py')+'\n'+r(S/'nodes.py')+'\n'+r(N/'nodes.py')
# Include code identity in checkpoint keys; never reuse latents across different patches.
job=r(S/'job.py').replace("'suite_version':1", "'suite_version':'h3-next-v1', 'extension_pins':"+repr(pins)).replace("'H3SuiteCLIPLoader'}", "'H3SuiteCLIPLoader','H3NextMolmo','H3NextReference','H3NextMotion'}")
code_identity=hashlib.sha256((plugin+r(N/'job.py')+r(N/'patch_molmo.py')+r(N/'weight_reader.py')+repr(pins)).encode()).hexdigest()
settings+='\nNEXT_CODE_IDENTITY = '+repr(code_identity)+'\n'
worker=settings+'\n'+asset_code+'\n'+r(B/'graphs.py')+'\n'+r(B/'runtime.py')+'\n'+r(B/'browser_job.py')+'\n'+r(S/'preflight.py')+'\n'+job+'\n'+r(N/'job.py')+'''
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
ns={};exec(r(B/'browser_workflow.py'),ns);exec(r(N/'canvas.py'),ns)
base=ns['browser_workflow']('A woman in a green dress walks through a sunlit garden. Natural movement, detailed face, gentle camera tracking. Audio: birds and quiet footsteps. If a reference image is connected, preserve the person in <Picture 1>.')
recipes=[('Singularity - Pika 4 Step Final','Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors','pika_h3_flex_turbo_4step_ref2va.safetensors',4),('Singularity - Pika 2 Step Draft','Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors','pika_h3_flex_turbo_2step_draft_ref2va.safetensors',2),('DenoMix - 8 Step Quality','MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors','MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors',8)]
canvases={title:ns['next_canvas'](base,title,model,lora,steps) for title,model,lora,steps in recipes}
wf=R/'workflows/H3_Next';wf.mkdir(exist_ok=True)
for name,g in canvases.items():(wf/(name+'.json')).write_text(json.dumps(g,indent=2)+'\n')
def new_stack(s):return s.replace('comfy-kitchen==0.2.35','comfy-kitchen==0.2.41').replace("version('comfy-kitchen')=='0.2.35'","version('comfy-kitchen')=='0.2.41'").replace('comfy-aimdo==0.5.5','comfy-aimdo==0.6.0').replace("version('comfy-aimdo')=='0.5.5'","version('comfy-aimdo')=='0.6.0'")
install='STACK_READY=False\n'+new_stack(r(B/'bootstrap.py'))+'\n'+new_stack(r(B/'stack.py'))+'\n'
install+="(MGPU/'h3_next_weights.py').write_text("+repr(r(N/'weight_reader.py'))+")\n"
install+='''
p=MGPU/'tp_source.py'
s=p.read_text();old='return safe_open(source, framework="pt", device="cpu")';new='return H3NextWeights(source)'
if old in s:
    s=s.replace(old,new).replace('from collections import OrderedDict','from collections import OrderedDict\\nfrom .h3_next_weights import H3NextWeights')
    p.write_text(s)
else:assert new in s,'Unexpected TP source reader.'
'''
install+=f"pinned_checkout('https://github.com/alisson-anjos/ComfyUI-Molmo2-H3.git',{pins['molmo']!r},COMFY/'custom_nodes/ComfyUI-Molmo2-H3')\n"
install+=f"pinned_checkout('https://github.com/matlowai/ComfyUI-Dials.git',{pins['dials']!r},COMFY/'custom_nodes/ComfyUI-Dials')\n"
install+="run([PY,'-m','pip','install','-c',constraints,'accelerate==1.15.0','molmo-utils==0.0.1'])\n"+r(N/'patch_molmo.py')+"\npatch_molmo(COMFY/'custom_nodes/ComfyUI-Molmo2-H3')\n"
# Check Molmo's actual INT8 kernel on its encoder GPU before downloading any weights.
install+='''
run([PY,'-c',"import torch; a=torch.ones((32,256),dtype=torch.int8,device='cuda:0'); b=torch.ones((256,256),dtype=torch.int8,device='cuda:0'); assert (torch._int_mm(a,b)==256).all(); print('Molmo INT8 GEMM probe passed')"])
run([PY,'-m','pip','check'])
plugin_source='''+repr(plugin)+'\nworker_source='+repr(worker)+'\ncanvases='+repr(canvases)+'\n'+'''
plugin_dir=COMFY/'custom_nodes/H3KaggleBrowser';plugin_dir.mkdir(parents=True,exist_ok=True)
(plugin_dir/'__init__.py').write_text(plugin_source)
worker_path=ROOT/'h3_next_worker.py';worker_path.write_text(worker_source)
(plugin_dir/'config.json').write_text(json.dumps({'root':str(ROOT),'python':str(PY),'worker':str(worker_path),'assets':SUITE_ASSETS}))
web=plugin_dir/'web';web.mkdir(exist_ok=True)
workflow_dir=COMFY/'user/default/workflows';workflow_dir.mkdir(parents=True,exist_ok=True)
for title,canvas in canvases.items():(workflow_dir/(title+'.json')).write_text(json.dumps(canvas))
(web/'full-workflow.json').write_text(json.dumps(next(iter(canvases.values()))))
'''
js=r(B/'open_workflow.js');start=js.index('            if (!(');end=js.index('            await app.loadGraphData',start)
js=js[:start]+'            if (!graph.nodes?.some(node => node.type === "H3SuiteSaveVideo")) throw new Error("H3 Next save node missing.");\n'+js[end:]
install+="(web/'open_workflow.js').write_text("+repr(js)+")\nSTACK_READY=True\n"
access=r(B/'cloudflare.py').replace("assert 'H3IsolatedSaveVideo' in api_request('/object_info')","assert 'H3SuiteSaveVideo' in api_request('/object_info')").replace("print('In ComfyUI: Workflows → MiniMax H3 - Full T4 Workflow → Run',flush=True)","print('In ComfyUI choose: '+', '.join(canvases),flush=True)")
stages={'catalog':r(B/'processes.py')+'\n'+asset_code,'install':install,'weights':'suite_prepare_assets(selected_model_assets())\n','workflow':r(B/'graphs.py')+'\n'+r(B/'runtime.py'),'access':access}
for name,src in [*stages.items(),('plugin',plugin),('worker',worker)]:compile(src,name,'exec')
entry=settings+'\n_H3_PIPELINE_SOURCES = '+pformat(stages,sort_dicts=False,width=110)+'\n'+r(B/'pipeline.py')+'\nrun_pipeline(_H3_PIPELINE_SOURCES,globals())\n'
intro='''# H3 Next — Molmo2 · Sol · Pika · Singularity · DenoMix

**New experimental notebook for Kaggle T4 ×2. Existing notebooks are unchanged.**

1. Set **GPU T4 ×2** and **Internet ON**.
2. Attach private Datasets from **H3_Next_Model_Packs.ipynb** if you have prepared them. Otherwise, missing files download automatically. The old H3 packs supply the shared reference/audio VAEs but do not contain the new models.
3. Press the single code cell's Play button. Wait for **Your full H3 workflow link**.
4. Open that Cloudflare link. Under **Workflows**, choose:
   - **Singularity - Pika 4 Step Final** — default comparison candidate.
   - **Singularity - Pika 2 Step Draft** — faster-step draft recipe; audio and picture failures are possible.
   - **DenoMix - 8 Step Quality** — single generation pass, without Todd's extra upscaling/refinement.
5. Edit the prompt and **seconds** in the conditioning node. Default: five seconds, 512 × 768. For a reference, upload an image in Load Image and connect its IMAGE dot to **reference_image**. Use `<Picture 1>` in your prompt. Press **Run**.

The **Motion control** node defaults to **0 (off)**. It changes movement inside the video, not rendering speed. Start small if you enable it. Outputs appear in ComfyUI's final Save node; no notebook preview or automatic rendering.

Molmo2 uses an experimental T4 FP16 adaptation while retaining INT8 weights and FP32 scales. Prompt/reference preparation, sampling, and decoding run in three separate processes, releasing Molmo before loading the diffusion model. Sol decodes after sampling exits. The original H3 VAE encodes references; Sol is used for output decoding. No sparse-attention patch, D-RoPE pass or upscaler is enabled.

**Validation limits:** CPU/browser checks cannot establish real T4 speed, memory fit or visual/audio quality. These new combinations require a Kaggle render test. No advertised YouTube/RTX/H200 timings are promised here. A 5-second request produces 124 frames, about 5.17 seconds at 24 fps.

The runtime lives separately under `/kaggle/working/h3_next_t4`. Do not run this and an older ComfyUI notebook simultaneously: both use port 8188. Stop the older session/server first. Switching between Singularity and DenoMix may download a different 19.53 GiB checkpoint. Saved model inputs avoid repeated downloads. Internet remains required for runtime setup and the tunnel.
'''
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(entry),nbformat.v4.new_markdown_cell('Keep Kaggle running while using ComfyUI. For failures, open the diagnostics link. Prepared conditioning, saved joint latents and completed clips can be reused on retry; a changed model, prompt, image or setting creates a new job. Use `stop_comfyui_access()` to stop this notebook’s browser server and tunnel.')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
for i,c in enumerate(nb.cells):c.id='run' if c.cell_type=='code' else f'h3-next-{i}'
nbformat.validate(nb);nbformat.write(nb,R/'H3_Next_T4.ipynb')
# CPU-only preparation: four distinct saved outputs, all under 20 GiB each.
packs={key:[a for a in assets if a['pack']==key] for key in ('singularity','denomix','molmo','support')}
assert all(sum(a['size'] for a in group)<20*1024**3 for group in packs.values())
old=nbformat.read(R/'H3_Model_Packs.ipynb',4)
code=next(c.source for c in old.cells if c.cell_type=='code')
start=code.index('PACKS=');end=code.index('\nassert PACK in PACKS',start)
code=code[:start]+'PACKS='+repr(packs)+code[end:]
code=code.replace("PACK='ref2va'  # ref2va, fl2va, encoder, support","PACK='singularity'  # singularity, denomix, molmo, support").replace('PACK must be ref2va, fl2va, encoder, or support','PACK must be singularity, denomix, molmo, or support')
# Upload verified real files as a private named Dataset by default.
code="AUTO_CREATE_DATASET = True\nDATASET_OWNER = ''  # Your Kaggle username, if not auto-detected.\n"+r(N/'dataset.py')+'\n'+code
code=code.replace("OUTPUT.mkdir(parents=True,exist_ok=True)", "if AUTO_CREATE_DATASET:\n    dataset_client,DATASET_OWNER=dataset_session(DATASET_OWNER)\nOUTPUT.mkdir(parents=True,exist_ok=True)")
# Replace the inherited working-only quota check with measured storage selection.
start=code.index("OUTPUT.mkdir(parents=True,exist_ok=True)")
end=code.index("manifest.write_text",start)
code=code[:start]+"OUTPUT=select_pack_storage(PACK,selected,AUTO_CREATE_DATASET)\nmanifest=OUTPUT.parent/'h3-pack-selection.json'\n"+code[end:]
code=code.replace("print('Save this CPU notebook version with its output files. Attach the saved output to your renderer using Add Input.')", "if AUTO_CREATE_DATASET:\n    save_private_dataset(OUTPUT,PACK,selected,DATASET_OWNER,dataset_client)\nelse:\n    print('Save Version with output files, then attach Notebook Outputs.')")
md='''# H3 Next model packs — automatic private Dataset creator

These are the new models for **H3_Next_T4.ipynb**. Your old packs remain valid for the older notebooks.

Create four separate copies of this preparation notebook. Rename each and change only the `PACK` line as follows:

| Notebook name | PACK | Size |
| --- | --- | --- |
'''
for key,group in packs.items():md+=f"| H3 Next Models - {key} | `{key}` | {sum(a['size'] for a in group)/1024**3:.2f} GiB |\n"
md+='''
For each copy:

1. Set **Accelerator: None**, **Internet ON**, and keep it **private**.
2. In **Add-ons → Secrets**, add your Kaggle API token as **KAGGLE_API_TOKEN** and enable it for this notebook. Never paste the token in the code. Existing configured Kaggle authentication also works.
3. Leave `AUTO_CREATE_DATASET = True`. Enter your Kaggle username in `DATASET_OWNER` if it is not detected automatically.
4. Run the single cell. It checks authentication first, downloads/verifies the selected pack, then automatically uploads it as a **private Kaggle Dataset** and prints its link.
5. Wait for Kaggle to finish processing. In H3 Next, choose **Add Input → Your Datasets** and attach it. Repeat for each pack you need. Leave `MODEL_DATASET=''` in the renderer.

**No manual Save Version is required to submit the Dataset.** Saving the notebook and logs is separate. Use a fresh CPU session/copy for each pack. All four packs are shared by the three H3 Next workflows; they total roughly 62 GiB. Each is below 20 GiB. Account storage limits still apply.

The builder measures free space in `/kaggle/working`, `/kaggle/temp` and `/tmp`, then uses one location that fits the entire pack. It reuses partial downloads when space permits; it never assumes a fixed temporary-disk quota. Temporary files last only for the current session, so wait for the Dataset upload and Kaggle processing before ending it.

Uploads use real files without a second full-size model copy. Existing Datasets are never overwritten. Successful submission is recorded for same-session retries; an existing remote Dataset without a local receipt may cause a duplicate-name error—check Your Datasets before retrying. Failed uploads retain verified files; rerun after fixing authentication or connectivity. Submission is not confirmation that Kaggle processing has completed.

Optional fallback: set `AUTO_CREATE_DATASET=False`, run the cell, then **Save Version with output files** and attach through **Notebook Outputs**. Actual authenticated Kaggle uploads have not been tested from this development environment.

'''
pnb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(md),nbformat.v4.new_code_cell(code)],metadata=nb.metadata)
for i,c in enumerate(pnb.cells):c.id=f'h3-next-models-{i}'
compile(code,'packs','exec');nbformat.validate(pnb);nbformat.write(pnb,R/'H3_Next_Model_Packs.ipynb')
print('Built H3 Next renderer, three canvases and four CPU model packs:',{k:round(sum(a['size'] for a in v)/1024**3,2) for k,v in packs.items()})
