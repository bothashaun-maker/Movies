from pathlib import Path
import os, sys, json, subprocess, shutil, time, hashlib, urllib.request, urllib.error, uuid

# Edit these settings, then press this cell's Play button once. Kaggle: T4 x2, Internet ON.
PROMPT = ('A woman in a green dress dances under warm courtyard string lights. '
          'Her dress and curls swirl as the camera gently tracks her. '
          'Audio: soft acoustic guitar, footsteps and a light evening breeze.')
MODE = 'text'                       # 'text' or 'image'
IMAGE_PATH = ''                     # e.g. /kaggle/input/my-image/first-frame.png
SEED = 42
MEGAPIXELS = 0.4                    # default canvas: 512 x 768
ASPECT = (2, 3)
DURATION = 5.0                      # snaps to the H3 frame grid: 124 frames at 24 fps
MODEL_FAMILY = 'ref2va'             # 'ref2va' or 'fl2va'; matching Turbo LoRA selected
USE_BRIDGE = False                 # optional Semantic Bridge: requires MODEL_FAMILY='fl2va'
BRIDGE_ALPHA = 0.10
MODEL_STORAGE = 'auto'              # checks /kaggle/temp and /tmp; never assumes a quota
MODEL_DATASET = ''                  # optional; attached /kaggle/input files are auto-detected
ALLOW_MODEL_DOWNLOADS = True

# Keep this path to reuse weights/environment from the previous notebook in this session.
ROOT = Path('/kaggle/working/minimax_h3_dual_t4')
COMFY = ROOT/'ComfyUI'
ENV = ROOT/'venv'
PY = ENV/'bin/python'
PORT = 8188
API = f'http://127.0.0.1:{PORT}'
COMFY_COMMIT = '73c9bad4d21e7addbe1d13bc92eee0f1431b017d'  # v0.37.0
MGPU_COMMIT = 'e0232dc1d1321e23ef308a0e6a30c39c8e062b46'   # kaggle-ok-64
STEPS = 4
USE_KITCHEN_ATTENTION = True
VRAM_HEADROOM_GB = 2.0
TIMEOUT_MINUTES = 90
MODELS_READY = False
STACK_READY = False
ROOT.mkdir(parents=True, exist_ok=True)
assert MODEL_FAMILY in {'ref2va','fl2va'} and MODE in {'text','image'}
assert 0.2 <= MEGAPIXELS <= 0.4 and 1 <= DURATION <= 5 and STEPS == 4
assert len(ASPECT) == 2 and min(ASPECT) > 0 and PROMPT.strip()
assert 0 <= SEED < 2**64
assert not USE_BRIDGE or MODEL_FAMILY == 'fl2va', 'Semantic Bridge requires FL2VA; its author excludes Ref2VA.'
if MODE == 'image':
    assert Path(IMAGE_PATH).is_file(), 'Set IMAGE_PATH to an attached image.'
child_env = os.environ.copy()
for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','UV_SYSTEM_PYTHON'):
    child_env.pop(key, None)
child_env.update(PIP_NO_CACHE_DIR='1', HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='60',
                 CUDA_VISIBLE_DEVICES='0,1', MMH3_H3_FP16='1', MMH3_TP_CHECK='1',
                 MMH3_TP_CHUNKS='4', MMH3_TP_COMPILE='1',
                 PIP_CACHE_DIR=str(ROOT/'pip-cache'), HF_HOME=str(ROOT/'hf-cache'))
print('Settings loaded. Preparing ComfyUI browser access; no automatic rendering.')
