"""Build local native FlashVSR replacements; never modifies the camera or SeedVR2 notebooks."""
from pathlib import Path
import base64,io,json,zipfile
import nbformat
R=Path(__file__).parent
REPO=R.parent.parent
files={'flash_common.py':(R/'flash_common.py').read_text(),
       'flash_check_dataset.py':(R/'flash_check_dataset.py').read_text(),
       'flash_dataset.py':(R/'dataset.py').read_text(),
       'flash_runtime.py':(R/'runtime.py').read_text(),
       'assets.json':(R/'assets.json').read_text()}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,text in files.items():
  if name.endswith('.py'):compile(text,name,'exec')
  info=zipfile.ZipInfo(name,(2026,11,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,text)
payload=base64.b64encode(buf.getvalue()).decode()
entry='''from pathlib import Path
import base64,io,zipfile,sys,importlib
WORK=Path('/kaggle/working/jjp-flashvsr')
RUNTIME=WORK/'runtime'
RUNTIME.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('''+repr(payload)+'''))) as archive:
    for name in archive.namelist():
        assert Path(name).name==name
        (RUNTIME/name).write_bytes(archive.read(name))
sys.path.insert(0,str(RUNTIME))
importlib.invalidate_caches()
for name in ('flash_runtime','flash_common','flash_dataset','flash_check_dataset'):
    old=sys.modules.get(name)
    if old is not None and Path(getattr(old,'__file__','')).parent==RUNTIME:sys.modules.pop(name,None)
'''
render_md='''# Native FlashVSR+ for two T4s — TeamAIQ engine

This replaces the slow SeedVR2 path with [TeamAIQ's dedicated FlashVSR+ upscaler](https://github.com/TeamAIQ/Colab-notebooks/blob/237383fb084b671c1eb5224db822d88fca6fc8fa/kaggle_notebooks/flashvsr-video-upscaler.ipynb). TeamAIQ/AIQUEST credit and native Gradio interface are retained. Their **inference functions/classes are unchanged**, checked by AST identity before launch. Source and model revisions are pinned and verified.

1. First run **JJP_FlashVSR_Models.ipynb** on **CPU / Accelerator None**. It automatically submits one private Dataset using the working camera-model uploader. Wait for processing. These are **different weights**: your old SeedVR2/Real-ESRGAN Dataset cannot substitute.
2. Attach that Dataset here. Select **GPU T4 ×2**, Internet ON. Stop the older H3/SeedVR2 studio. Run this one cell.
3. Wait for **Engine ready** and the native **Cloudflare/Gradio** links. Initial model load and a small upstream warm-up happen once. No uploaded clip runs automatically.
4. In **Input video**, upload your clip. Leave **Turbo / 2x upscale / Normal motion**, then press **Upscale Video**. Progress reports chunks, elapsed time, and measured per-GPU timings.
5. Completed MP4s live under `/kaggle/working/jjp-flashvsr/outputs/`. Preserve notebook output files before ending the session.

The native engine pipelines chunks across both GPUs with resident FP16 weights, FP32 attention accumulation/norm safeguards, a compact decoder, and streamed output. It deliberately skips slow Triton attention on T4 and chooses a tested PyTorch attention backend on the actual devices. It does not use SeedVR2's BF16 VAE or all-block CPU swapping.

Initial UI changes only: **2×**, **CRF 14**, **veryfast** encoder, comparison copy **OFF**. Turbo favours speed; Base is available for a finer-detail comparison. Colour correction remains the native optional setting. No arbitrary temporal downsampling or reduced model precision has been added.

**Budget protection:** the parent stops only this owned engine if startup exceeds 6 minutes or a clip exceeds the editable 8-minute limit. Those are spending limits, **not measured render-time promises**. Partial work is not a completed video; already completed clips are preserved. The engine's native Stop button can stop earlier.

This is a new backend, not a claim that SeedVR2 was made fast. Full-size T4 speed, fit and restoration quality have not been measured here. Local tests cover source identity, native CPU attention, media/output helpers and process/budget controls. No new user GPU session is required for those tests. AIQuest H3/Qwen camera and existing notebooks are unchanged. Nothing is pushed to GitHub by this notebook.
'''
pack_md='''# Native FlashVSR model Dataset — CPU only

Set **Accelerator: None**, **Internet ON**. Enable your existing private Kaggle Secrets (`KAGGLE_API_TOKEN` or `KAGGLE_USERNAME` + `KAGGLE_KEY`). Set DATASET_OWNER if automatic detection cannot find your username. Never paste credentials into code.

Run the single cell. It reuses the working native streaming Dataset uploader from your camera notebook and automatically submits **one private Dataset** containing Turbo, Base and the compact decoder: approximately **12.89 GiB total**. Only one missing model is downloaded/uploaded at a time; the largest is about **5.29 GiB**, plus working reserve. It measures scratch capacity. It reuses verified attached files and releases only its own successfully uploaded temporary downloads.

Wait for **PRIVATE DATASET SUBMITTED** and Kaggle processing. Attach that one Dataset to **JJP_FlashVSR_T4.ipynb**. No manual PC download/upload or Save Version step is required to submit the Dataset. Live authenticated upload has not been performed here; local API tests are not proof of Kaggle processing.

The existing SeedVR2/H3 Dataset is different and remains unchanged. All five files have immutable revisions, exact sizes and SHA256 values. Original model licences apply. TeamAIQ's engine is referenced from its pinned public notebook; its code is not embedded in this model Dataset.
'''
render_code='MAX_JOB_SECONDS = 480\nMAX_STARTUP_SECONDS = 360\n'+entry+'''from flash_runtime import run
run(RUNTIME,WORK,MAX_JOB_SECONDS,MAX_STARTUP_SECONDS)
'''
pack_code="DATASET_OWNER = ''\n"+entry+'''import json
from flash_common import GIB,choose_scratch
from flash_dataset import create_dataset,authenticated_api
catalog=json.loads((RUNTIME/'assets.json').read_text())
api=authenticated_api()
cache=choose_scratch(max(a['bytes'] for a in catalog['assets'])+GIB,
                     ['/kaggle/temp','/tmp','/kaggle/working'],'jjp-flashvsr-pack-cache')
create_dataset(catalog,WORK/'dataset-receipt',Path('/kaggle/input'),cache,DATASET_OWNER,api=api)
'''
for name,md,code in [('JJP_FlashVSR_T4.ipynb',render_md,render_code),('JJP_FlashVSR_Models.ipynb',pack_md,pack_code)]:
 compile(code,name,'exec')
 nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(md),nbformat.v4.new_code_cell(code)],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
 for i,c in enumerate(nb.cells):c.id=f'flash-native-{i}'
 nbformat.validate(nb);nbformat.write(nb,REPO/name)
 print('Built',name)
