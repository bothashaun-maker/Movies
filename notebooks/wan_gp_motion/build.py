"""Build two self-contained notebooks. Existing notebooks are never regenerated."""
from pathlib import Path
import base64
import io
import json
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def notebook(title, instructions, code):
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'},
                         'kaggle': {'accelerator': 'none', 'isInternetEnabled': True, 'language': 'python',
                                    'sourceType': 'notebook', 'isGpuEnabled': False}},
            'cells': [cell('markdown', '# '+title+'\n\n'+instructions), cell('code', code)]}


def payload():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.iterdir()):
            if path.suffix in ('.py', '.json', '.lock') and path.name not in ('build.py', 'test_kit.py', 'notebook_review.json'):
                info = zipfile.ZipInfo(path.name, date_time=(2026, 11, 13, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode()


def extraction(encoded):
    return '''from pathlib import Path
import base64, io, os, sys, zipfile
OUTPUT = Path('/kaggle/working/jjp-wangp-motion')
RUNTIME = OUTPUT/'runtime'
RUNTIME.mkdir(parents=True, exist_ok=True)
PAYLOAD = '''+repr(encoded)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    for name in archive.namelist():
        assert Path(name).name == name
        (RUNTIME/name).write_bytes(archive.read(name))
sys.path.insert(0, str(RUNTIME))
'''


def build():
    encoded = payload()
    intro = '''**WanGP GUI → Cloudflare. No ComfyUI. No automatic generation.**

1. Run **JJP_WanGP_Motion_Models.ipynb** once with **Accelerator: None, Internet ON**. It automatically submits **ONE private Dataset** with all 13 required files (40.53 GiB). Use Kaggle Secrets as described in that notebook.
2. Import a fresh copy of this notebook. Select **GPU T4 ×2**, **Internet ON**. Add Input → Your Datasets → attach that **ONE JJP WanGP Motion Models Dataset**. Do not substitute Director, H3 Next or Image Studio packs.
3. Run the single cell below. It verifies/stages weights, installs the pinned isolated runtime, probes the actual T4, and opens **WanGP's native GUI** through a printed Cloudflare URL. Keep the cell running.
4. Upload ONE Qwen appearance still. The default is **A: image only**. Keep seed `8675309`, `864×480`, `124 frames`, `24 fps`, **PDD 8 steps / Euler**. Press Generate.
5. For **B**, keep every setting and the same image/prompt/seed. Select **Use One Reference Video** and upload a **silent 124-frame 864×480 / 24 fps Blender guide**. Press Generate. Do not use In-Context Guide, Video-to-Video Edit or ControlNet for this test.
6. Compare camera arc, image fidelity, cross geometry, proxy leakage, time and VRAM. Export/save the MP4s and settings from the GUI and **Save Version with outputs before ending the session**. Files live under `/kaggle/working/jjp-wangp-motion`. Native generated audio should be muted in Movavi; never upload the song master to H3.

**Hardware/storage checks:** GPU 0 only; the second T4 does not pool memory. FP16 + SDPA, INT8 weights, Profile 5 offloading, no compilation/CUDA graphs, conservative VAE tiling. The 864×480 uploaded guide is internally reduced to **288×160** using WanGP's native reference-size control; output stays 864×480. All weights are mirrored to local scratch to avoid the Dataset-read stalls seen in Director. The initial free-space gate is **52.54 GiB** (40.53 GiB models + 8 GiB runtime allowance + 4 GiB reserve). Actual free space is measured; if no scratch mount has room, startup stops before model load. Attached Dataset space is separate from local scratch.

**Bounded failures:** model staging 20 min; Torch install 20 min; application install 30 min; GUI readiness 5 min; tunnel 2 min; each requested render 20 min. The supervisor stops WanGP if the render deadline expires or available host RAM stays below 1.5 GiB. Stopping the notebook cell also terminates its owned processes. Logs and completed clips are retained; a stopped render is not resumed automatically.

**Validation boundary:** CPU package installation, native GUI/browser checks, model-manifest/source checks, Blender export and watchdog tests are recorded in `notebooks/wan_gp_motion/VALIDATION.md`. They do not establish T4 model fit, generation speed or camera accuracy. Actual Kaggle H3 inference and private Dataset creation remain unverified until run with your GPU/account. The notebook is a guarded experiment, not a measured GPU guarantee.
'''
    code = '''# JJP WanGP Motion Test Kit — one executable cell
PORT = 7860
JOB_TIMEOUT_MINUTES = 20
SESSION_HOURS = 8
SCRATCH_ROOT = ''  # Auto: measured /kaggle/temp, /tmp, /kaggle/working. Optional writable local override.
'''+extraction(encoded)+'''
from launcher import launch
launch(RUNTIME, OUTPUT, PORT, JOB_TIMEOUT_MINUTES, SESSION_HOURS, SCRATCH_ROOT)
'''
    studio = notebook('JJP WanGP Motion Test Kit', intro, code)
    studio['metadata']['kaggle'].update(accelerator='nvidiaTeslaT4', isGpuEnabled=True)
    models_intro = '''**Build: CPU-BUILDER-v1.2 · 14 November 2026.** Uses the notebook's existing Python/Kaggle packages, matching the working H3 Next builder. Checks the real uploader imports (including `six`) before model downloads; no separate CPU virtual environment.

**Run once on CPU: Accelerator None, Internet ON.** This creates **ONE private reusable Dataset**, automatically, using the working Image Studio streaming-uploader method. It includes all 13 pinned files: Ref2VA pruned rank-8 INT8, Qwen3-VL layer-50 GGUF Q4_K_M + six tokenizer/config files, Ref2VA PDD heads + LoRA, FP16 video VAE, FP32 audio VAE and native latent upscaler. Total: **40.53 GiB**.

Enable Kaggle Secret **KAGGLE_API_TOKEN**, or **KAGGLE_USERNAME** + **KAGGLE_KEY**, for this notebook. Set `DATASET_OWNER` below if token authentication does not expose your Kaggle username. Credentials are passed privately to the child; none are written to the notebook or logs. No `datasets.get/status` permission prerequisite is added.

Run the one cell. The existing notebook Python and installed Kaggle SDK are reused, as in H3 Next; a missing SDK or `six` is installed only if needed. Authentication is checked before model downloads. The creator uploads one verified file at a time through Kaggle's own SDK, then releases only its own uploaded temporary download. Existing attached models are reused without copying/deleting them. Peak missing-model storage is **19.61 GiB + 2 GiB reserve**, measured on local scratch, rather than a second 40 GiB copy. The largest file will not fit a 19.5 GiB-only mount, so the notebook checks `/kaggle/temp` and `/tmp` too and stops clearly if none has room.

When **PRIVATE DATASET SUBMITTED** appears, wait for Kaggle processing. In the GPU notebook: Add Input → Your Datasets → attach this **ONE Dataset**. No manual PC download/upload, and no manual Save Version is required for Dataset submission. An upload receipt means submitted, not that Kaggle has finished processing.

Model sources, exact revisions, byte sizes and SHA256 values are embedded. All component licenses remain in force. Live Kaggle upload is not claimed from mock tests.
'''
    models_code = '''# JJP WanGP Motion Models — CPU-BUILDER-v1.2 — one saved Dataset
DATASET_OWNER = ''  # Your Kaggle username if not auto-detected.
'''+extraction(encoded)+'''
# Resolve secrets in the Kaggle kernel; kaggle_secrets is not inside the isolated child environment.
if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
    except Exception:
        pass
if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        os.environ['KAGGLE_USERNAME'] = client.get_secret('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = client.get_secret('KAGGLE_KEY')
    except Exception:
        pass
if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
    raise RuntimeError('Enable KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY in Kaggle Secrets. No model downloads started.')
import json
from common import GIB, bounded, choose_scratch
catalog = json.loads((RUNTIME/'assets.json').read_text())
largest = max(a['bytes'] for a in catalog['assets'])
scratch = choose_scratch(largest+2*GIB, ['/kaggle/temp','/tmp','/kaggle/working'], 'jjp-wangp-model-builder-v1')
env = os.environ.copy()
env.update(JJP_DATASET_OWNER=DATASET_OWNER, PYTHONUNBUFFERED='1')
# H3 Next's working CPU setup: use the notebook interpreter and its Kaggle SDK.
bounded([sys.executable,RUNTIME/'check_dataset_runtime.py'], OUTPUT/'dataset-runtime-check.log', 300, env=env)
bounded([sys.executable,RUNTIME/'dataset.py',RUNTIME/'assets.json',OUTPUT/'model-dataset','/kaggle/input',scratch/'download-cache'],
        OUTPUT/'dataset-create.log', 4*3600, env=env)
receipt = json.loads((OUTPUT/'model-dataset'/'dataset-upload.json').read_text())
print('PRIVATE DATASET SUBMITTED:', receipt['url'])
print('Wait for Kaggle processing, then attach this ONE Dataset to the Motion Test Kit.')
'''
    models = notebook('JJP WanGP Motion Models — ONE saved Dataset', models_intro, models_code)
    for name, contents in [('JJP_WanGP_Motion_Test_Kit.ipynb', studio), ('JJP_WanGP_Motion_Models.ipynb', models)]:
        # Stable cell IDs are required by modern nbformat validators.
        for index, c in enumerate(contents['cells']):
            c['id'] = ('instructions' if index == 0 else 'run-kit')
        (REPO/name).write_text(json.dumps(contents, indent=1)+'\n')
        print('Built', name)


if __name__ == '__main__':
    build()
