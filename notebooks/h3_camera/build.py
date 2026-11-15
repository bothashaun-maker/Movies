"""Build only the separate AIQuest Dataset edition and its automatic CPU saver."""
from pathlib import Path
import base64,hashlib,io,json,zipfile
from integrate import sources

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def payload(names):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,(2026,11,15,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(HERE/name).read_bytes())
    return base64.b64encode(out.getvalue()).decode()

def extract(encoded,folder):
    return f'''from pathlib import Path
import base64,io,zipfile,sys,importlib
WORK=Path('/kaggle/working/{folder}')
RUNTIME=WORK/'runtime'
RUNTIME.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as archive:
    for name in archive.namelist():
        assert Path(name).name==name
        (RUNTIME/name).write_bytes(archive.read(name))
sys.path.insert(0,str(RUNTIME))
importlib.invalidate_caches()
for name in ('camera_common','camera_dataset','camera_check_dataset','camera_bootstrap'):
    old=sys.modules.get(name)
    if old is not None and Path(getattr(old,'__file__','')).parent==RUNTIME:
        sys.modules.pop(name,None)
'''

def notebook(title,intro,code,gpu=False):
    result={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'kaggle':{'isInternetEnabled':True,'isGpuEnabled':gpu,'accelerator':'nvidiaTeslaT4' if gpu else 'none'}},
        'cells':[{'cell_type':'markdown','id':'instructions','metadata':{},'source':('# '+title+'\n\n'+intro).splitlines(keepends=True)},
            {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    compile(code,title,'exec')
    return result

def build():
    setup,engine,launcher,original_hash=sources(HERE/'aiquest_upstream.ipynb')
    for name,text in [('aiquest_setup.py',setup),('aiquest_engine.py',engine),('aiquest_launcher.py',launcher)]:
        (HERE/name).write_text(text,encoding='utf-8')
    # Keep the proven saver/storage algorithms; isolate their module names.
    (HERE/'camera_common.py').write_text((HERE/'common.py').read_text(encoding='utf-8').replace('JJP WanGP Motion Models','JJP H3 Camera Models'),encoding='utf-8')
    (HERE/'camera_check_dataset.py').write_bytes((HERE/'check_dataset_runtime.py').read_bytes())
    (HERE/'camera_dataset.py').write_text((HERE/'dataset.py').read_text(encoding='utf-8').replace('from common import','from camera_common import').replace('from check_dataset_runtime import check_runtime','from camera_check_dataset import prepare_runtime').replace('KaggleApi = check_runtime()','KaggleApi = prepare_runtime()'),encoding='utf-8')
    (HERE/'camera_bootstrap.py').write_bytes((HERE/'bootstrap.py').read_bytes())
    catalog=json.loads((HERE/'assets.json').read_text());size=sum(a['bytes'] for a in catalog['assets'])/1024**3
    (HERE/'provenance.json').write_text(json.dumps({'upstream_blob':catalog['upstream_blob'],'original_engine_sha256':original_hash,
        'changes':['only camera/orbit catalog choices','resolve adapters from verified saved files','owned process cleanup','Dataset verification/staging; original inference/UI preserved']},indent=2)+'\n',encoding='utf-8')
    common=['camera_common.py','assets.json']
    gpu=extract(payload(common+['camera_bootstrap.py','aiquest_setup.py','aiquest_engine.py','aiquest_launcher.py']),'jjp-h3-camera')+'''
from camera_bootstrap import main
main(RUNTIME,WORK)
'''
    cpu='DATASET_OWNER = ""  # Your Kaggle username if authentication does not expose it.\n'+extract(payload(common+['camera_dataset.py','camera_check_dataset.py']),'jjp-h3-camera-dataset')+'''
import json
from camera_dataset import authenticated_api,create_dataset
from camera_common import choose_scratch,GIB
catalog=json.loads((RUNTIME/'assets.json').read_text())
api=authenticated_api()  # Resolve secrets before SDK import, then check dependencies; before downloads.
peak=max(a['bytes'] for a in catalog['assets'])
scratch=choose_scratch(peak+2*GIB,[Path('/kaggle/temp'),Path('/tmp'),Path('/kaggle/working')],'jjp-h3-camera-builder')
create_dataset(catalog,WORK/'saved-dataset',Path('/kaggle/input'),scratch/'download-cache',DATASET_OWNER,api)
'''
    intro=f'''Based on [AIQuest's actual Kaggle notebook]({catalog['upstream_notebook']}). Its Wan2GP engine, native Qwen GGUF, tensor-parallel inference and Gradio interface are retained. **This is the Dataset edition of AIQuest's notebook, not a new ComfyUI engine.**

1. First run **JJP_H3_Camera_Models.ipynb** on CPU and wait for its private Dataset to finish processing.
2. Attach that **ONE complete Dataset** here. No H3-Next Dataset is required. Select **GPU T4 x2**, **Internet ON**.
3. Run the single cell. It verifies/stages the saved files, performs AIQuest's setup and opens AIQuest's interface. Keep the cell running; its model stays resident between clips.
4. Upload your picture to **Start frame**, not the character-reference inputs. Use **480p / 5 s / four steps** initially. PDMD supports text, first-frame and first/last-frame generation. The two optional camera/orbit adapters are included; other variants and model downloads are excluded from this edition.

The startup model load still occurs once. No startup/render speed or visual quality has been measured for this Dataset edition. CPU checks are not a full T4 render. Blender is not required.

For the current arc shot, paste this complete **Pixaroma first-frame format** in Scene / action, with the crucifix picture uploaded as Start frame:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Jesus on the main crucifix shown in <Picture 1> remains still. The camera arcs to the right around the main crucifix with small amplitude at slow speed, continuously facing Jesus and keeping the complete main crucifix visible throughout one continuous shot.

overall_soundscape: Quiet wind.

non_diegetic_music: N/A
```

Prompt grammar and references: [Pixaroma skills](https://workflows.pixaroma.com/gems/). This wording is a test input, not a guarantee of an exact 30-degree path. The dedicated 360 adapter's published recipe uses 28 steps and 73 frames; its four-step compatibility is not established by this build.
'''
    models=f'''**CPU only: Accelerator None, Internet ON. Automatic private Dataset creation is ON.**

This saves **one independent, complete Dataset: 13 pinned files, {size:.2f} GiB**. It includes AIQuest's FL2VA pruned INT8 model, native Qwen GGUF plus six config/tokenizer files, video/audio VAEs, PDMD four-step adapter and camera/orbit adapters. No H3-Next Dataset is required.

Enable your existing Kaggle Secrets **KAGGLE_API_TOKEN**, or **KAGGLE_USERNAME + KAGGLE_KEY**. Run the one cell. The saver reuses the working GitHub streaming-upload/authentication method. It uploads one verified model at a time through Kaggle's SDK and automatically submits a **private Dataset**. No manual Save Version or PC upload is needed.

Only the largest missing file plus reserve is needed locally: **19.61 GiB + 2 GiB**. Free scratch is measured; no fixed Kaggle quota is assumed. Existing attached source files are never deleted. Credentials are not printed or written into the notebook.

After **PRIVATE DATASET SUBMITTED**, wait for Kaggle processing, then attach the printed Dataset to **JJP_H3_Native_Camera.ipynb**. Submission is not confirmation of completed processing. Original model licenses apply. Live authenticated uploading is not claimed from local mock checks.
'''
    for name,nb in [('JJP_H3_Native_Camera.ipynb',notebook('AIQuest H3 — native Qwen camera / saved-Dataset edition',intro,gpu,True)),
                    ('JJP_H3_Camera_Models.ipynb',notebook('AIQuest H3 Camera Models — ONE automatic private Dataset',models,cpu))]:
        (ROOT/name).write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
        print('Built',name)

if __name__=='__main__':build()
