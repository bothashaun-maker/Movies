"""Build the two independent self-contained LTX-2.5 Kaggle notebooks."""
from pathlib import Path
import base64
import io
import json
import zipfile
import nbformat
from integrate import build_engine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def payload(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            if name.endswith('.py'):
                compile(source.decode(), name, 'exec')
            info = zipfile.ZipInfo(name, (2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source)
    return base64.b64encode(buf.getvalue()).decode()


def entry(files, work):
    return f'''from pathlib import Path
import base64,io,zipfile,sys,importlib
WORK=Path('/kaggle/working/{work}')
RUNTIME=WORK/'runtime'
RUNTIME.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload(files)!r}))) as archive:
    for name in archive.namelist():
        if Path(name).name!=name:
            raise RuntimeError('Invalid embedded runtime path')
        (RUNTIME/name).write_bytes(archive.read(name))
sys.path.insert(0,str(RUNTIME))
importlib.invalidate_caches()
for name in ('ltx25_runtime','ltx25_common','ltx25_dataset','ltx25_check_dataset'):
    previous=sys.modules.get(name)
    if previous is not None and Path(getattr(previous,'__file__','')).parent==RUNTIME:
        sys.modules.pop(name,None)
'''


def notebook(filename, text, code, gpu):
    compile(code, filename, 'exec')
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text), nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'},
                  'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': gpu,
                             'accelerator': 'nvidiaTeslaT4' if gpu else 'none'}})
    for i, cell in enumerate(nb.cells):
        cell.id = f'ltx25-native-{i}'
    nbformat.validate(nb)
    nbformat.write(nb, ROOT/filename)
    print('Built', filename)


def build():
    build_engine(HERE)
    assets = json.loads((HERE/'assets.json').read_text())['assets']
    total = sum(a['bytes'] for a in assets)/1024**3
    peak = max(a['bytes'] for a in assets)/1024**3
    common = {name: (HERE/name).read_bytes() for name in ('assets.json', 'ltx25_common.py')}
    gpu_files = dict(common, **{'ltx25_runtime.py': (HERE/'runtime.py').read_bytes(),
                              'engine.py': (HERE/'engine.py').read_bytes(),
                              'provenance.json': (HERE/'provenance.json').read_bytes()})
    cpu_files = dict(common, **{'ltx25_dataset.py': (HERE/'dataset.py').read_bytes(),
                              'ltx25_check_dataset.py': (HERE/'ltx25_check_dataset.py').read_bytes()})
    cpu = "DATASET_OWNER = ''  # Optional: your Kaggle username, never an API key.\n"+entry(cpu_files, 'jjp-ltx25-dataset')+'''
import json
from ltx25_common import GIB,choose_scratch
from ltx25_dataset import create_dataset,authenticated_api
catalog=json.loads((RUNTIME/'assets.json').read_text())
api=authenticated_api()  # Check the working SDK/authentication before downloading models.
scratch=choose_scratch(max(a['bytes'] for a in catalog['assets'])+2*GIB,
    ['/kaggle/temp','/tmp','/kaggle/working'],'jjp-ltx25-builder')
create_dataset(catalog,WORK/'dataset-receipt',Path('/kaggle/input'),scratch/'download-cache',DATASET_OWNER,api=api)
'''
    gpu = 'MAX_STARTUP_SECONDS = 1200\nMAX_JOB_SECONDS = 1800\n'+entry(gpu_files, 'jjp-ltx25')+'''
from ltx25_runtime import run
run(RUNTIME,WORK,MAX_STARTUP_SECONDS,MAX_JOB_SECONDS)
'''
    cpu_intro = f'''# JJP LTX-2.5 Models — automatic private Dataset

**CPU only / Accelerator None / Internet ON.** One cell automatically submits **one complete private Dataset**. This uses the existing native Kaggle streaming uploader/authentication method from your working notebooks.

1. Enable your existing Kaggle Secrets: `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`. Do not paste credentials into code.
2. Leave `DATASET_OWNER` blank if authentication finds your username; otherwise set only your Kaggle username.
3. Run the single code cell. It verifies and uploads one file at a time and prints **PRIVATE DATASET SUBMITTED** with a link.
4. Wait for Dataset processing. Attach that ONE Dataset to **JJP_LTX25_T4.ipynb** through Add Input → Your Datasets.

The Dataset contains **13 pinned files / {total:.3f} GiB**: LTX-2.5 22B Distilled INT8 ConvRot, Gemma-4 12B INT8 with its tokenizer/configuration, projection/connectors, video/audio VAEs, vocoder and the spatial upscaler required by the native loader. No H3, SeedVR2, MSR LoRA or unrelated weights.

The largest missing file is **{peak:.3f} GiB**. Scratch is measured for that file plus a 2 GiB reserve, rather than storing the entire Dataset twice. Existing verified attached files are reused. Only this builder's successfully uploaded temporary downloads are released. Files in your attached Datasets are preserved. Every file has an immutable revision, exact byte count and SHA-256.

No manual Save Version or PC upload is needed to submit the Dataset. A submission receipt confirms submission, not finished Kaggle processing. Original model licences apply. Local SDK tests do not constitute a live authenticated upload.
'''
    gpu_intro = f'''# JJP LTX-2.5 — native dual-T4 test studio

Based on [AIQUEST's native LTX-2.5 Kaggle engine](https://github.com/TeamAIQ/Colab-notebooks/blob/e297aa0fb6eb1c9eaa57c855fc1cf66657b4b66b/kaggle_notebooks/ltx-2-5-22b-distilled-video-generator.ipynb), with AIQUEST credit and Gradio interface retained. Separate from your working H3, image and upscale notebooks.

1. Run **JJP_LTX25_Models.ipynb** on CPU first. Wait for the automatic private Dataset to finish processing.
2. Attach that **ONE Dataset** here. Choose **GPU T4 x2**, **Internet ON**, and stop any older studio using those GPUs.
3. Run this **one startup cell**. It verifies/stages saved models, installs the native dependencies and starts the resident engine. Open the printed Cloudflare or Gradio link. No clip runs automatically.
4. To test an image: expand **Image → Video**, upload a **Start frame**, write the scene/action/camera prompt, and press **Generate Video**. An End frame is optional for first/last-frame control. With no images it is text-to-video.
5. Start at the native **480p preset / 5 s / 8 steps**. The exact dimensions are logged: landscape text-to-video is **832×512**; image inputs use their own rounded aspect ratio. “480p” is a preset label, not an exact output height. 121 frames at 24 fps is about 5.04 s.

**Recommendation:** use one approved first image, one clear action and one camera movement for the first test. Use separate clips for edit points requiring exact character count. This edition generates its own audio; it does not synchronize to your existing song. Keep generated ambience if useful, or use your song in the editor. Audio-driven avatars and MSR are separate features, not implied here.

The INT8 transformer is tensor-parallel across both T4s; Gemma runs through the native offloader. FP32 encoder/norm protections and native INT8 kernels are retained. This test edition keeps the official eight-step schedule and refuses non-finite conditioning, latents or decoded media instead of zeroing invalid reference values. Kernel availability and timings are reported by the native engine; no speedup or model-quality guarantee is made.

The **{total:.3f} GiB** model set is staged from the attached Dataset to verified local scratch; no inference-time model downloads. Staging requires the missing files plus a 6 GiB reserve on one writable filesystem. Temporary space is measured; do not assume a fixed Kaggle quota. Outputs are MP4 + generated AAC audio under `/kaggle/working/jjp-ltx25/outputs/`; preserve your outputs before ending the session. The model Dataset does not save generated videos.

Initial engine load occurs once; leave the cell running between clips. Native Stop cancels a generation. Editable limits are 20 minutes for engine startup and 30 minutes per clip; they stop only this notebook's owned processes, preserve completed clips, and are **spending limits, not runtime promises**. Setup/staging time is separate.

No full-size T4 inference, visual quality, live Dataset upload or public tunnel was executed in development. Local validation covers source identity, generated notebooks, storage, actual SDK upload control flow with remote calls stubbed, and numerical-guard behavior.
'''
    notebook('JJP_LTX25_Models.ipynb', cpu_intro, cpu, False)
    notebook('JJP_LTX25_T4.ipynb', gpu_intro, gpu, True)


if __name__ == '__main__':
    build()
