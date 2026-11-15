# Step 1: Environment, pinned Wan2GP, dependencies and model downloads

# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  👉 CHOOSE YOUR MODEL: "pdmd" (default) · "dmad" · "fused" · "official"              ║
# ║     pdmd / dmad: newest 4-step distillations; text, image, first + last frame        ║
# ║     fused / official: also Reference → Video + voice clone (see the table above)     ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
H3_VARIANT = "pdmd"

import os
import sys
import time
import shutil
import subprocess
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor

def stop_old_engine():
    stop_camera_engine(_CAMERA_WORK)


stop_old_engine()

# ---- Settings ----
WAN2GP_COMMIT = "b8b18f8114e432eea8f3d7e853a51dd91fa99571"  # Wan2GP, 1 Oct 2026 (the version this notebook was built on)
MODELS_DIR = os.environ["H3_MODELS_DIR"]
LORA_DIR = os.environ["H3_LORA_DIR"]

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"  # no second copy of 40 GB in the xet cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print("🎬 MiniMax H3 Turbo INT8 - Video + Audio Generator")
print("📺 Created by: AIQUEST Academy | YouTube & X: @AIQuestAcademy")
print("=" * 70)
MODEL_MENU = {
    "pdmd": "Base H3 + PDMD 4-step LoRA       | text, image, first+last frame | 4 steps",
    "dmad": "Base H3 + DMAD 4-step LoRA       | text, image, first+last frame | 4 steps",
    "fused": "MATLOWAI fused turbo (Ref2VA)    | + Reference → Video, voice    | 4 steps",
    "official": "Ref2VA + lightx2v 8-step LoRA    | + Reference → Video, voice    | 8 steps",
}
if H3_VARIANT not in MODEL_MENU:
    raise SystemExit(f'❌ Unknown H3_VARIANT "{H3_VARIANT}". Set it to one of: ' + ", ".join(f'"{k}"' for k in MODEL_MENU))
print("🧠 Model (H3_VARIANT, first line of this cell):")
for key, text in MODEL_MENU.items():
    print(f"   {'👉' if key == H3_VARIANT else '  '} {key:<9} {text}")
print("   To switch: edit H3_VARIANT at the top of this cell and re-run Step 1 + Step 2.")
print("=" * 70)
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    for i, line in enumerate(smi):
        print(f"✅ GPU {i}: {line}")
    if len(smi) < 2:
        print("❌ This notebook needs 2 GPUs: Settings → Accelerator → GPU T4 x2")
except Exception:
    print("❌ No GPU found: Settings → Accelerator → GPU T4 x2")

# ---- 1. Wan2GP, pinned to a tested commit ----
t0 = time.time()
if not os.path.isdir("Wan2GP/.git"):
    print("\n📥 Fetching Wan2GP...")
    subprocess.run(["git", "init", "-q", "Wan2GP"], check=True)
    subprocess.run(["git", "-C", "Wan2GP", "remote", "add", "origin", "https://github.com/DeepBeepMeep/Wan2GP.git"], check=True)
subprocess.run(["git", "-C", "Wan2GP", "fetch", "-q", "--depth", "1", "origin", WAN2GP_COMMIT], check=True)
subprocess.run(["git", "-C", "Wan2GP", "reset", "-q", "--hard", "FETCH_HEAD"], check=True)  # also undoes stale edits on re-runs
print(f"✅ Wan2GP @ {WAN2GP_COMMIT[:10]} ({time.time() - t0:.0f}s)")

# ---- 2. Dependencies: only what the H3 path imports ----
# (Wan2GP's full requirements.txt pins an onnxruntime nightly that is no longer on PyPI.)
print("📦 Installing dependencies...")
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "--timeout", "120", "--retries", "5",
                "mmgp==3.8.2", "comfy-kitchen==0.2.35", "transformers==4.54.0", "diffusers==0.36.0", "gguf==0.17.1", "optimum-quanto", "accelerate>=1.1.1", "gradio==5.29.0", "huggingface_hub[hf_xet]", "hf_xet>=1.5.2", "sentencepiece", "soundfile"], check=True)
print(f"✅ Dependencies installed ({time.time() - t0:.0f}s)")

