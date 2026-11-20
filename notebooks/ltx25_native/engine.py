"""LTX-2.5 22B Distilled video + audio engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Step 2 of the notebook writes this file as run_ltx25.py and runs it. It drives Wan2GP's LTX-2.5 code directly:

* Wan2GP is pinned (Step 1) to the commit this engine was built on. Later Wan2GP versions read settings from the
  full Wan2GP UI module (wgp), which a standalone engine does not load.
* Transformer: the official 22B distilled checkpoint in INT8 ConvRot. Every INT8 linear runs on Comfy Kitchen's INT8
  tensor-core GEMM (W8A8 with the ConvRot Hadamard rotation) instead of being dequantized to FP16 at every step.
* Tensor parallel on GPU T4 x2: every transformer block is split in half over the two T4s (attention heads and FFN
  units), both halves stay resident, and the GPUs compute every block together. Nothing is streamed during denoising.
* Gemma-4 12B text encoder: pinned in RAM, streamed by MMGP through GPU 0 once per new prompt, on the exact INT8 path
  (LLM activations have outliers) and on the prompt's real tokens only: LTX pads every prompt to 1024 tokens, and the
  padding never reaches the video model. (With LTX25_TENSOR_PARALLEL=0 or one GPU: the transformer is streamed from
  pinned RAM on GPU 0 and Gemma-4 stays resident on GPU 1.)
* One 8-step distilled pass at the requested resolution in FP16 (Turing has FP16 tensor cores, not BF16 ones).
"""
import faulthandler
import gc
import glob
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")
faulthandler.enable(file=sys.__stderr__, all_threads=True)  # a hard crash (segfault, abort) prints where it happened
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

WAN2GP_DIR = os.path.abspath("Wan2GP")
WORK_DIR = os.path.dirname(WAN2GP_DIR)
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
MODELS_DIR = os.environ.get("LTX25_MODELS_DIR", "/kaggle/tmp/ltx25_models")
GEMMA_DIR = os.path.join(MODELS_DIR, "gemma4-12b-ltx-v1")
TRANSFORMER_FILE = os.path.join(MODELS_DIR, "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
GEMMA_FILE = os.path.join(GEMMA_DIR, "gemma4-12b-ltx-v1_int8_convrot.safetensors")
SERVER_PORT = 7860
FPS = 24
# "t2v": text / image → video with generated audio. "a2v": lip-sync avatar (reference image + your audio → video).
# "msr": multi-subject reference (up to 4 subjects + a background → one video, with LiconStudio's MSR V2 LoRA).
APP_MODE = os.environ.get("LTX25_MODE", "t2v")
APP_MODE = APP_MODE if APP_MODE in ("a2v", "msr") else "t2v"
APP_NAME = {"a2v": "LTX-2.5 Lip-Sync Avatar", "msr": "LTX-2.5 Multi-Subject Reference"}.get(APP_MODE, "LTX-2.5 22B Distilled")
MSR_LORA_FILE = os.path.join(MODELS_DIR, "LTX-2.5-Licon-MSR-V2.safetensors")
MSR_SLOTS_FILE = os.path.join(MODELS_DIR, "LTX-2.5-Licon-MSR-V2_image_slots.safetensors")
# Compute dtype of the transformer / sampler / text path: FP16 (Turing has FP16 tensor cores, BF16 runs ~20% slower
# per step here). LTX25_DTYPE=bf16 switches to BF16, as Wan2GP runs LTX. The VAE decoders always run in FP16.
_DTYPE_NAME = os.environ.get("LTX25_DTYPE", "fp16").lower()
if not os.path.isdir(WAN2GP_DIR):
    raise SystemExit("❌ Wan2GP is missing: run Step 1 first.")
_missing = [p for p in (TRANSFORMER_FILE, GEMMA_FILE) + ((MSR_LORA_FILE,) if APP_MODE == "msr" else ())
            if not os.path.exists(p)]
if _missing:
    raise SystemExit(f"❌ Model files missing ({', '.join(map(os.path.basename, _missing))}): run Step 1 first.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)

import numpy as np
import psutil
import torch
from PIL import Image

import gradio as gr


def _gb(n):
    return f"{n / 1024**3:.1f} GB"


def _ram_in_use():
    """total - available: unlike psutil's 'used', this counts pinned (page-locked) memory, which Linux files as shared."""
    vm = psutil.virtual_memory()
    return f"{_gb(vm.total - vm.available)} / {_gb(vm.total)}"


def _fmt_secs(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class ConsoleBar:
    """One live progress line per phase, redrawn every second with a carriage return (Kaggle's cell output overwrites it in place).
    A finished phase is replaced by a one-line summary. Step-based phases (denoising) interpolate inside the current
    step from the previous step's duration, so the bar keeps moving and the ETA counts down between callbacks."""

    WIDTH, LINE = 24, 104
    WATCHED = {"decode": 300, "audio": 120, "save": 120}  # phase -> seconds before a stack dump

    def __init__(self, stream):
        self.out = stream
        self.lock = threading.RLock()
        self.phase = None
        self.visible = False
        self.text_pending = False  # another writer is in the middle of a line: don't draw over it
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, key, label, total=None, unit=""):
        with self.lock:
            if self.phase is not None and self.phase["key"] == key:
                return
            self.finish()
            now = time.time()
            self.phase = dict(key=key, label=label, total=total, unit=unit, done=0, t0=now, t_done=now, dt=None)
            if key in self.WATCHED:  # a phase that should take seconds: dump every thread's stack if it hangs
                faulthandler.dump_traceback_later(self.WATCHED[key], file=sys.__stderr__)
            self._draw()

    def update(self, done, total=None):
        with self.lock:
            p = self.phase
            if p is None:
                return
            now = time.time()
            if total:
                p["total"] = total
            if done > p["done"]:
                p["dt"] = (now - p["t_done"]) / (done - p["done"])
                p["t_done"], p["done"] = now, done
            self._draw()

    def finish(self):
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            elapsed = time.time() - p["t0"]
            line = f"  ✅ {p['label']}"
            if p["total"]:
                line += f" {p['done']}/{p['total']} {p['unit']}"
            line += f" in {_fmt_secs(elapsed)}"
            if p["unit"] == "steps" and p["done"]:
                line += f" ({elapsed / p['done']:.1f}s/step)"
            self._clear()
            self.out.write(line + "\n")
            self.out.flush()

    def abort(self):
        """End the live line of a phase that failed (no ✅ summary)."""
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            self._clear()
            self.out.write(f"  ❌ {p['label']} stopped after {_fmt_secs(time.time() - p['t0'])}\n")
            self.out.flush()

    def _render(self):
        p, now = self.phase, time.time()
        elapsed = now - p["t0"]
        if not p["total"]:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(now * 4) % 10]
            return f"  {spinner} {p['label']} | {_fmt_secs(elapsed)}"
        progress = p["done"]
        if p["dt"] and p["done"] < p["total"]:
            progress += min(0.97, (now - p["t_done"]) / p["dt"])
        frac = min(1.0, progress / p["total"])
        fill = int(round(frac * self.WIDTH))
        line = (f"  ⏳ {p['label']:<16} [{'█' * fill}{'░' * (self.WIDTH - fill)}] {frac * 100:3.0f}% | "
                f"{p['done']}/{p['total']} {p['unit']}")
        if p["dt"]:
            eta = max(0.0, (p["total"] - p["done"]) * p["dt"] - (now - p["t_done"]))
            line += f" | {p['dt']:.1f}s each | ETA {_fmt_secs(eta)}"
        return line + f" | {_fmt_secs(elapsed)}"

    def _draw(self):
        if self.phase is None or self.text_pending:
            return
        self.out.write("\r" + self._render().ljust(self.LINE))
        self.out.flush()
        self.visible = True

    def _clear(self):
        if self.visible:
            self.out.write("\r" + " " * self.LINE + "\r")
            self.visible = False

    def _loop(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                self._draw()


class _BarAwareStream:
    """stdout / stderr wrapper: ordinary log lines clear the live bar first and print above it."""

    def __init__(self, stream, bar):
        self._stream, self._bar = stream, bar

    def write(self, data):
        with self._bar.lock:
            if data:
                self._bar._clear()
                self._bar.text_pending = not data.endswith("\n")
            return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


BAR = ConsoleBar(sys.stdout)
sys.stdout = _BarAwareStream(sys.stdout, BAR)
sys.stderr = _BarAwareStream(sys.stderr, BAR)

if not torch.cuda.is_available():
    raise SystemExit("❌ No GPU found. Settings → Accelerator → GPU T4 x2.")
TWO_GPUS = torch.cuda.device_count() >= 2
# Tensor parallel (default with two GPUs): the transformer is split over both T4s. LTX25_TENSOR_PARALLEL=0 keeps it on
# GPU 0 (streamed from RAM) with Gemma-4 resident on GPU 1 instead.
TENSOR_PARALLEL = TWO_GPUS and os.environ.get("LTX25_TENSOR_PARALLEL", "1") != "0"
MAIN_DEV = torch.device("cuda", 0)                        # sampler state, VAEs, MMGP models
AUX_DEV = torch.device("cuda", 1 if TWO_GPUS else 0)      # second half of the transformer, or Gemma-4
TE_DEV = AUX_DEV if TWO_GPUS and not TENSOR_PARALLEL else MAIN_DEV  # Gemma-4 text encoder
torch.cuda.set_device(MAIN_DEV)
print("=" * 70)
print(f"🎬 {APP_NAME} - AIQUEST Academy")
print("=" * 70)
for _i in range(torch.cuda.device_count()):
    _p = torch.cuda.get_device_properties(_i)
    print(f"  GPU {_i}: {_p.name} ({_p.total_memory / 1024**3:.1f} GB, sm_{_p.major}{_p.minor})")
print(f"  RAM: {_ram_in_use()} in use")
if not TWO_GPUS:
    print("  ⚠️ Only one GPU: the transformer is streamed from RAM at every step. Use GPU T4 x2 for full speed.")
sys.stdout.flush()

# T4 (sm_75) has no FlashAttention kernel: memory-efficient SDPA is the fast path. cuDNN autotuning stays off: on a T4
# it benchmarks every new conv3d shape of the video VAE, which costs far more than it saves on a single decode.
torch.backends.cudnn.benchmark = False
PIPELINE_DTYPE = torch.bfloat16 if _DTYPE_NAME == "bf16" else torch.float16
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# ==== Wan2GP source fixes (Step 1 resets the repo, so they are applied on every start) ====
def _patch(rel_path, old, new, marker, regex=False):
    path = os.path.join(WAN2GP_DIR, rel_path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if marker in text:
        return True
    if regex:
        text, count = re.subn(old, new, text, flags=re.M)
    else:
        count = text.count(old)
        text = text.replace(old, new)
    if count == 0:
        print(f"  ⚠️ Wan2GP fix not applied, the code it targets changed: {rel_path}")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


_LTX = "models/ltx2/"
_FIXES = [
    # The distilled pipeline hard-codes BF16: read it from a module variable instead (FP16 or BF16, see PIPELINE_DTYPE).
    (_LTX + "ltx_pipelines/distilled.py", r"^(\s+)(self\.)?dtype = torch\.(bfloat16|float16)(  # AIQ fp16)?$",
     r"\1\2dtype = AIQ_PIPELINE_DTYPE  # AIQ dtype", "# AIQ dtype", True),
    (_LTX + "ltx_pipelines/distilled.py", "\nimport torch\n",
     "\nimport torch\nAIQ_PIPELINE_DTYPE = torch.bfloat16  # AIQ dtype var\n", "# AIQ dtype var", False),
    # PixelNorm squares its input: FP16 overflows past 256 (flat grey frames). Compute the RMS in FP32.
    (_LTX + "ltx_core/model/common/normalization.py",
     "        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)\n        # Normalize by the root-mean-square (RMS).\n"
     "        rms = torch.sqrt(mean_sq + self.eps)\n        return x / rms",
     "        orig_dtype = x.dtype  # AIQ fp32 rms\n        x_float = x.float()\n"
     "        mean_sq = torch.mean(x_float**2, dim=self.dim, keepdim=True)\n        rms = torch.sqrt(mean_sq + self.eps)\n"
     "        return (x_float / rms).to(orig_dtype)", "# AIQ fp32 rms", False),
    # Same overflow in the per-token RMS of the Gemma hidden states (all-NaN text features in FP16).
    (_LTX + "ltx_core/text_encoders/gemma/feature_extractor.py",
     "    b, t, d, l = encoded_text.shape\n    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)\n"
     "    normed = encoded_text * torch.rsqrt(variance + 1e-6)\n    normed = normed.reshape(b, t, d * l)\n",
     "    orig_dtype = encoded_text.dtype  # AIQ fp32 token rms\n    enc_f = encoded_text.float()\n    b, t, d, l = enc_f.shape\n"
     "    variance = torch.mean(enc_f**2, dim=2, keepdim=True)\n    normed = enc_f * torch.rsqrt(variance + 1e-6)\n"
     "    normed = normed.reshape(b, t, d * l).to(orig_dtype)\n", "# AIQ fp32 token rms", False),
    # Latents reach the VAE convolutions in the pipeline dtype; the FP32 video encoder (I2V) needs them cast.
    (_LTX + "ltx_core/model/video_vae/convolution.py", "        x = self.conv(x)",
     "        if x.dtype != self.conv.weight.dtype:  # AIQ conv cast\n            x = x.to(dtype=self.conv.weight.dtype)\n"
     "        x = self.conv(x)", "# AIQ conv cast", False),
    (_LTX + "ltx_core/model/audio_vae/audio_vae.py", "    decoded_audio = audio_decoder(latent)",
     "    if hasattr(audio_decoder, 'conv_in') and hasattr(audio_decoder.conv_in, 'weight'):  # AIQ audio cast\n"
     "        latent = latent.to(dtype=audio_decoder.conv_in.weight.dtype)\n    decoded_audio = audio_decoder(latent)\n"
     "    if hasattr(vocoder, 'vocoder') and hasattr(vocoder.vocoder, 'conv_pre'):\n"
     "        decoded_audio = decoded_audio.to(dtype=vocoder.vocoder.conv_pre.weight.dtype)", "# AIQ audio cast", False),
    (_LTX + "ltx_core/model/audio_vae/vocoder.py",
     "return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)",
     "filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)  # AIQ filter cast\n"
     "        return F.conv1d(x, filt, stride=self.stride, groups=n_channels)", "# AIQ filter cast", False),
    (_LTX + "ltx_core/model/audio_vae/vocoder.py",
     "spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)",
     "basis = self.forward_basis.to(dtype=y.dtype, device=y.device)  # AIQ basis cast\n"
     "        spec = F.conv1d(y, basis, stride=self.hop_length, padding=0)", "# AIQ basis cast", False),
]
# Modality guidance: the distilled pipeline forces alt_guidance_scale to 1 (off). Lip sync uses it: an extra pass without
# the audio <-> video cross-attention, extrapolated away from (LTX-2 team: stronger audio / video coupling, lip sync).
_FIXES.append((_LTX + "ltx_pipelines/distilled.py", "        alt_guidance_scale = 1.0\n",
               "        alt_guidance_scale = float(alt_guidance_scale)  # AIQ modality guidance\n",
               "# AIQ modality guidance", False))
_fixed = sum(_patch(path, old, new, marker, regex) for path, old, new, marker, regex in _FIXES)
print(f"🔧 Wan2GP fixes: {_fixed}/{len(_FIXES)} applied")
if _fixed != len(_FIXES):
    raise RuntimeError("A required pinned numerical patch did not apply. No model loaded.")

# ==== Wan2GP + MMGP ====
from mmgp import offload, quant_router

quant_router.unregister_handler(".fp8_quanto_bridge")
quant_router.register_handler("shared.qtypes.int8_convrot")  # the INT8 ConvRot checkpoints (wgp.py does this at import)

# Comfy Kitchen INT8 tensor-core GEMM ("INT8 Kernels: Auto" in the Wan2GP UI, which a standalone engine must turn on).
# Kitchen quantizes activations per row in chunks of this scratch size: 16 MB means ~160-row GEMMs on LTX's FFN,
# 128 MB lets each call cover ~1300 rows, a much better shape for the tensor cores.
from shared.kernels import int8_backend

int8_backend._SCRATCH_BYTES = 128 * 1024 * 1024
try:
    int8_backend.configure("auto", 1)
except Exception as e:
    print(f"⚠️ INT8 kernels unavailable ({e}): INT8 layers are dequantized to FP16 instead (slower)")
INT8_LABEL = {"kitchen": "INT8 tensor-core kernels (Comfy Kitchen)", "triton": "INT8 Triton kernels"}.get(
    int8_backend._backend, "PyTorch INT8 fallback (slow)")

from optimum.quanto.tensor.weights import qbytes as _qbytes
from shared.utils import files_locator as fl

fl.set_checkpoints_paths([MODELS_DIR, "ckpts", "."])
from models.ltx2.ltx2_handler import family_handler
from models.ltx2.ltx_pipelines import distilled as ltx_distilled

ltx_distilled.AIQ_PIPELINE_DTYPE = PIPELINE_DTYPE
import models.ltx2.ltx_pipelines.utils.helpers as ltx_helpers
from models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder import RawTextEmbeddings, _apply_connectors
from models.ltx2.ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig


@contextmanager
def exact_int8():
    """Run INT8 layers on the exact path (dequantized weights, unquantized activations) inside this block.
    For the LLM text encoder: its activations have extreme outlier channels, and the per-row INT8 activation
    quantization of the fast kernels flattens everything else (worse prompt understanding)."""
    saved_backend = int8_backend._backend
    saved_forward = _qbytes.WeightQBytesLinearFunction.__dict__.get("forward")
    if int8_backend._original_forward is not None:
        _qbytes.WeightQBytesLinearFunction.forward = staticmethod(int8_backend._original_forward)
    int8_backend._backend = "pytorch"
    try:
        yield
    finally:
        int8_backend._backend = saved_backend
        if saved_forward is not None:
            _qbytes.WeightQBytesLinearFunction.forward = saved_forward


# ==== Text encoding: Gemma-4 on its own GPU, real tokens only ====
TIMES = {}  # phase timings of the current generation


def _encode_one(text_encoder, prompt):
    """Same hidden states as text_encoder.encode_raw(prompt) for the real tokens, without running Gemma on padding.
    The tokenizer left-pads every prompt to 1024 tokens. Gemma never lets a real token attend to padding, and the
    feature extractor and connectors ignore padded positions, so Gemma runs on the n real tokens alone, at the
    positions they have in the padded sequence (1024-n .. 1023, so RoPE sees the same angles), and the hidden
    states are padded back with zeros."""
    pairs = text_encoder.tokenizer.tokenize_with_weights(prompt)["gemma"]
    ids = torch.tensor([[t[0] for t in pairs]], dtype=torch.long)
    mask = torch.tensor([[t[1] for t in pairs]], dtype=torch.long)
    total, n = ids.shape[1], int(mask.sum())
    if n == 0 or not bool(mask[0, total - n:].all()):  # not plain left padding: run the original code
        raw = text_encoder.encode_raw(prompt)
        return RawTextEmbeddings(tuple(h.to(MAIN_DEV) for h in raw.hidden_states), raw.attention_mask.to(MAIN_DEV),
                                 raw.padding_side), 0
    gemma = text_encoder.model
    language_model = gemma.model if hasattr(gemma, "model") else gemma
    device = gemma.device  # what encode_raw uses: GPU 1 here, or MMGP's device when it manages Gemma
    real_ids = ids[:, total - n:].to(device)
    positions = torch.arange(total - n, total, device=device).unsqueeze(0)
    out = language_model(input_ids=real_ids, attention_mask=torch.ones_like(real_ids), position_ids=positions,
                         output_hidden_states=True, use_cache=False)
    # The real tokens only: postprocess_trimmed pads after the feature projection (49 x 1024 x 3840 padded hidden states
    # and their FP32 norms would cost ~3 GB of VRAM for nothing).
    hidden = tuple(h.to(MAIN_DEV) for h in out.hidden_states)
    return RawTextEmbeddings(hidden, mask.to(MAIN_DEV), "left"), n


def encode_text_fast(text_encoder, prompts):
    t0 = time.time()
    BAR.start("text", "Prompt encoding")
    results, tokens = [], []
    try:
        with torch.no_grad(), exact_int8(), torch.cuda.device(TE_DEV):
            for prompt in prompts:
                raw, n = _encode_one(text_encoder, prompt)
                results.append(raw)
                tokens.append(n)
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    _unload_mmgp()  # Gemma's resident part leaves GPU 0 before the projection and connectors load
    TIMES["text"] = TIMES.get("text", 0.0) + time.time() - t0
    bad = sum(int((~torch.isfinite(r.hidden_states[-1][:, r.hidden_states[-1].shape[1] - k:])).sum())
              for r, k in zip(results, tokens) if k)
    print(f"  📝 Gemma-4 on GPU {TE_DEV.index}: {', '.join(map(str, tokens))} token(s) in {time.time() - t0:.1f}s"
          + (f" | ⚠️ {bad} non-finite values in the text features" if bad else ""))
    if bad:
        raise RuntimeError("Prompt encoder produced NaN/Inf; no video generated")
    return results


def _unload_mmgp():
    """Unload what MMGP holds on GPU 0 (it keeps a model resident until another one needs the room)."""
    offloader = globals().get("OFFLOAD")
    if offloader is not None and offloader.active_models_ids:
        offloader.unload_all()


_orig_postprocess = ltx_distilled.postprocess_text_embeddings


def _left_pad(x, total):
    if x is None or x.shape[1] == total:
        return x
    out = x.new_zeros((x.shape[0], total, *x.shape[2:]))
    out[:, total - x.shape[1]:] = x
    return out


def postprocess_trimmed(embeddings, feature_extractor_linear, embeddings_connector, audio_embeddings_connector,
                        return_attention_masks=False):
    """postprocess_text_embeddings with the feature projection run on the real tokens only. The projection works token
    by token, and the connectors replace every padded position with their learnable registers, so the values at padded
    positions are never read: the contexts are the same as with the 1024-token projection. Exact INT8 path."""
    results = []
    with exact_int8():
        for item in embeddings:
            mask = item.attention_mask.to(MAIN_DEV)
            total, n = mask.shape[1], int(mask.sum())
            if mask.shape[0] != 1 or item.padding_side != "left" or n == 0 or not bool(mask[0, total - n:].all()):
                results += _orig_postprocess([item], feature_extractor_linear, embeddings_connector,
                                             audio_embeddings_connector, return_attention_masks=return_attention_masks)
                continue
            hidden = tuple(h[:, h.shape[1] - n:].to(MAIN_DEV) for h in item.hidden_states)
            video_in, audio_in = feature_extractor_linear(hidden, torch.ones_like(mask[:, total - n:]), padding_side="left")
            hidden = None
            out = _apply_connectors(_left_pad(video_in, total), _left_pad(audio_in, total), mask, embeddings_connector,
                                    audio_embeddings_connector)
            results.append(out if return_attention_masks else out[:2])
    _unload_mmgp()
    return results


ltx_distilled.encode_text = encode_text_fast  # distilled.py imported these by name
ltx_distilled.postprocess_text_embeddings = postprocess_trimmed


# ==== Image-to-Video conditioning ====
# The video VAE encoder overflows in FP16 and returns an all-NaN conditioning latent, which spreads through the joint
# audio/video attention (black clip, noise audio, no error). The encoder runs in FP32 (set up after loading) and the
# latent is handed back in the pipeline dtype. A latent that is still non-finite is re-encoded on the CPU in FP32.
def _cond_attr(item):
    return "latent" if hasattr(item, "latent") else ("keyframes" if hasattr(item, "keyframes") else None)


def _wrap_conditioner(fn, label):
    def inner(*args, **kwargs):
        pipe_dtype = kwargs.get("dtype")
        encoder = kwargs.get("video_encoder")
        try:
            enc_dtype = next(encoder.parameters()).dtype
        except Exception:
            enc_dtype = None
        if enc_dtype is not None and enc_dtype != pipe_dtype:
            kwargs = dict(kwargs, dtype=enc_dtype)
        if kwargs.get("images"):
            BAR.start("encode", "Image encoding")
        items = fn(*args, **kwargs)

        def nonfinite(entries):
            for it in entries:
                a = _cond_attr(it)
                v = getattr(it, a) if a else None
                if torch.is_tensor(v) and not bool(torch.isfinite(v).all()):
                    return True
            return False

        if encoder is not None and nonfinite(items):
            print(f"  ⚠️ {label}: non-finite conditioning latent, re-encoding on the CPU in FP32")
            origin = next(encoder.parameters()).device
            try:
                encoder.to(device="cpu", dtype=torch.float32)
                cpu_items = fn(*args, **dict(kwargs, dtype=torch.float32, device=torch.device("cpu")))
                for it in cpu_items:
                    a = _cond_attr(it)
                    if a is not None and kwargs.get("device") is not None:
                        setattr(it, a, getattr(it, a).to(device=kwargs["device"]))
                items = cpu_items
            except Exception as retry_error:
                raise RuntimeError("Reference encoding failed; refusing to discard identity conditioning") from retry_error
            finally:
                encoder.to(device=origin)
        for item in items:
            a = _cond_attr(item)
            lat = getattr(item, a) if a else None
            if not torch.is_tensor(lat):
                continue
            fixed = lat
            if not bool(torch.isfinite(fixed).all()):
                raise RuntimeError("Reference conditioning contains NaN/Inf after FP32 retry; no video generated")
            if pipe_dtype is not None and fixed.dtype != pipe_dtype:
                fixed = fixed.to(pipe_dtype)
            if fixed is not lat:
                setattr(item, a, fixed)
        if kwargs.get("images"):
            BAR.finish()
        return items
    return inner


for _name, _label in (("image_conditionings_by_replacing_latent", "start frame"),
                      ("image_conditionings_by_adding_guiding_latent", "end frame")):
    _orig = getattr(ltx_helpers, _name, None)
    if _orig is not None:
        _wrapped = _wrap_conditioner(_orig, _label)
        setattr(ltx_helpers, _name, _wrapped)
        if hasattr(ltx_distilled, _name):  # imported by name in distilled.py
            setattr(ltx_distilled, _name, _wrapped)


# ==== VAE decode: timed, with a tiled retry if a large clip does not fit ====
_orig_decode_video = ltx_distilled.vae_decode_video_to_tensor
_orig_decode_audio = ltx_distilled.vae_decode_audio


def _tiling_with_spatial(tiling_config, tile):
    overlap = max(0, (tile // 4) // 32 * 32)
    temporal = tiling_config.temporal_config if tiling_config is not None else None
    return TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile, tile_overlap_in_pixels=overlap),
                        temporal_config=temporal)


# ==== Multi-subject references: Wan2GP's MSR encode, in the pipeline dtype, with a tiled retry ====
from models.ltx2 import msr as ltx_msr

_orig_msr_encode = ltx_msr.MSRReferenceImages.encode


def _msr_encode(self, height, width, video_encoder, dtype, device, tiling_config):
    """Each reference: 25 / 33 repeated frames through the (FP32) video encoder, plus its learned slot embedding. The
    encoder returns FP32 latents: they are handed back in the pipeline dtype (FP32 reference tokens would promote the
    whole token sequence to FP32). If a reference does not fit next to the transformer half, it is encoded in 512 or
    256 px tiles (Wan2GP's tiled encode)."""
    configs = [tiling_config] + [_tiling_with_spatial(tiling_config, tile) for tile in (512, 256)]
    BAR.start("encode", f"Reference encoding ({len(self.images)} x {self.frame_count} frames)")
    try:
        for attempt, config in enumerate(configs):
            try:
                conditions = _orig_msr_encode(self, height, width, video_encoder, dtype, device, config)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == len(configs) - 1:
                    raise
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  ⚠️ Reference encoding out of memory, retrying with "
                      f"{configs[attempt + 1].spatial_config.tile_size_in_pixels} px tiles")
    except BaseException:
        BAR.abort()
        raise
    for condition in conditions:
        if not bool(torch.isfinite(condition.latent).all()):
            raise RuntimeError("A reference image encoded to non-finite values")
        condition.latent = condition.latent.to(dtype)
    BAR.finish()
    return conditions


ltx_msr.MSRReferenceImages.encode = _msr_encode


VAE_BYTES_PER_PIXEL_FRAME = 150  # FP16 LTX-2.5 video decoder activations, whole frame: ~146 measured on a T4
VAE_DECODER_BYTES = 0.85 * 1024**3
VAE_REPLICA = None  # tensor parallel: a copy of the video decoder on GPU 1, so both GPUs decode chunks at the same time


def _fit_vae_tiling(config, height, width, frames):
    """With the transformer resident on both GPUs (tensor parallel) ~4-5 GB per GPU is left for decoding. Temporal chunks
    sized to that keep whole-frame decoding (no tile seams), with LTX's 24-frame overlap; among the sizes that fit, the
    one with the least decoding per GPU (chunks are decoded two at a time) wins."""
    if (not TENSOR_PARALLEL or config is None or config.spatial_config is not None or config.temporal_config is None
            or not height or not width or not frames):
        return config
    margin = 0.5 * 1024**3
    free = torch.cuda.mem_get_info(MAIN_DEV)[0] - VAE_DECODER_BYTES - margin  # MMGP loads the decoder on GPU 0 next
    if VAE_REPLICA is not None:
        free = min(free, torch.cuda.mem_get_info(AUX_DEV)[0] - margin)
    max_frames = int(free // (VAE_BYTES_PER_PIXEL_FRAME * width * height)) // 8 * 8
    if max_frames < 48:
        return _tiling_with_spatial(config, 512)
    lanes = 2 if VAE_REPLICA is not None else 1
    latent_frames = (int(frames) - 1) // 8 + 1
    overlap = 3  # latent frames of a 24-frame overlap
    best = None
    for chunk in range(48, max_frames + 1, 8):
        tile = (chunk - 1) // 8 + 1
        groups = 1 if latent_frames <= tile else -(-(latent_frames - tile) // (tile - overlap)) + 1
        cost = -(-groups // lanes) * min(tile, latent_frames)  # latent frames decoded per GPU
        if best is None or cost < best[0]:
            best = (cost, chunk)
        if groups == 1:
            break
    return TilingConfig(spatial_config=None,
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=best[1], tile_overlap_in_frames=24))


def _parallel_tiled_decode(self, latent, tiling_config=None, timestep=None, generator=None, interrupt_check=None,
                           copy_emitted_chunks=True):
    """VideoDecoder.tiled_decode with the temporal chunks decoded two at a time, on GPU 0 (this decoder) and GPU 1
    (VAE_REPLICA). The chunks are independent (the LTX-2.5 decoder adds no noise); they are blended on the CPU in the
    original order with the original arithmetic, so the frames are the same as with the one-GPU loop."""
    from concurrent.futures import ThreadPoolExecutor
    from models.ltx2.ltx_core.types import VideoLatentShape

    full_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
    total_frames = full_shape.frames
    groups = self._group_tiles_by_temporal_slice(self._prepare_tiles(latent, tiling_config))
    if not groups:
        return
    lanes = [(self, latent.device, latent)]
    if VAE_REPLICA is not None and len(groups) > 1:
        lanes.append((VAE_REPLICA, AUX_DEV, latent.to(AUX_DEV)))
    inference = torch.is_inference_mode_enabled()

    def decode(index):
        decoder, device, lat = lanes[index % len(lanes)]
        group = groups[index]
        span = self._normalize_temporal_slice(group[0].out_coords[2], total_frames)
        shape = full_shape._replace(frames=span.stop - span.start).to_torch_shape()
        with torch.inference_mode(inference), torch.no_grad(), torch.cuda.device(device):
            buffer = torch.zeros(shape, device=device, dtype=lat.dtype)
            weights = torch.zeros((1, 1, shape[2], shape[3], shape[4]), device=device, dtype=lat.dtype)
            weights = decoder._accumulate_temporal_group_into_buffer(
                group_tiles=group, buffer=buffer, latent=lat, timestep=timestep, generator=generator,
                temporal_slice=span, total_frames=total_frames, interrupt_check=None, weights_buffer=weights)
            return span, buffer.cpu(), weights.cpu()

    previous_chunk = previous_weights = previous_span = None
    with ThreadPoolExecutor(max_workers=1) as helper:
        for first in range(0, len(groups), len(lanes)):
            if interrupt_check is not None and interrupt_check():
                return
            jobs = [helper.submit(decode, i) for i in range(first + 1, min(first + len(lanes), len(groups)))]
            results = [decode(first)] + [job.result() for job in jobs]
            for span, buffer, weights in results:
                if previous_chunk is not None:
                    if previous_span.stop > span.start:
                        overlap_len = previous_span.stop - span.start
                        overlap = slice(span.start - previous_span.start, None)
                        previous_chunk[:, :, overlap] += buffer[:, :, :overlap_len]
                        previous_weights[:, :, overlap] += weights[:, :, :overlap_len]
                        buffer[:, :, :overlap_len] = previous_chunk[:, :, overlap]
                        weights[:, :, :overlap_len] = previous_weights[:, :, overlap]
                    previous_weights = previous_weights.clamp_(min=1e-8)
                    emit = span.start - previous_span.start
                    previous_chunk[:, :, :emit] /= previous_weights[:, :, :emit]
                    chunk = previous_chunk[:, :, :emit]
                    previous_chunk, previous_weights, previous_span = buffer, weights, span
                    yield chunk.clone() if copy_emitted_chunks else chunk
                    continue
                previous_chunk, previous_weights, previous_span = buffer, weights, span
    if previous_chunk is not None:
        previous_chunk /= previous_weights.clamp_(min=1e-8)
        yield previous_chunk


def decode_video(latent, video_decoder, tiling_config=None, *args, **kwargs):
    latent_tensor = latent[0] if isinstance(latent, list) else latent  # the decoder empties the list: keep it for a retry
    if isinstance(latent, list):
        latent.clear()
    decoder_weight = next(video_decoder.parameters(), None) if isinstance(video_decoder, torch.nn.Module) else None
    if decoder_weight is not None:  # BF16 sampler -> FP16 decoder
        latent_tensor = latent_tensor.to(decoder_weight.dtype)
    t0 = time.time()
    if _LORA_LAYERS:
        torch.cuda.synchronize(MAIN_DEV)
        torch.cuda.synchronize(AUX_DEV)
        freed = park_loras()
        if freed:
            print(f"  🅿️ LoRA parked in pinned RAM during the VAE decode ({_gb(freed / 2)} per GPU, {time.time() - t0:.1f}s)")
    tiling_config = _fit_vae_tiling(tiling_config, kwargs.get("expected_height"), kwargs.get("expected_width"),
                                    kwargs.get("expected_frames"))
    base = torch.cuda.memory_allocated(MAIN_DEV)
    torch.cuda.reset_peak_memory_stats(MAIN_DEV)
    BAR.start("decode", "VAE decoding")
    try:
        configs = [tiling_config] + [_tiling_with_spatial(tiling_config, t) for t in (512, 256)]
        for attempt, config in enumerate(configs):
            try:
                out = _orig_decode_video([latent_tensor], video_decoder, config, *args, **kwargs)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == len(configs) - 1:
                    raise
                torch.cuda.synchronize()  # never release memory while kernels queued before the failure still run
                gc.collect()
                torch.cuda.empty_cache()
                tile = configs[attempt + 1].spatial_config.tile_size_in_pixels
                print(f"  ⚠️ VAE decode out of memory, retrying with {tile} px tiles")
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    TIMES["decode"] = TIMES.get("decode", 0.0) + time.time() - t0
    t = config.temporal_config.tile_size_in_frames if config is not None and config.temporal_config is not None else "all"
    tile = config.spatial_config.tile_size_in_pixels if config is not None and config.spatial_config is not None else "whole frame"
    gpus = "GPU 0 + GPU 1" if VAE_REPLICA is not None else "GPU 0"
    print(f"  🎞️ VAE decode on {gpus}: {t}-frame chunks, {tile} | peak +{_gb(torch.cuda.max_memory_allocated(MAIN_DEV) - base)} on GPU 0")
    return out


def decode_audio(latent, audio_decoder, *args, **kwargs):
    # The latent comes in the sampler dtype (BF16 for lip sync), the audio decoder runs in FP16: match its weights
    # (its first layer wraps the convolution, so the dtype is read from the parameters).
    decoder_weight = next(audio_decoder.parameters(), None) if isinstance(audio_decoder, torch.nn.Module) else None
    if decoder_weight is not None and torch.is_tensor(latent):
        latent = latent.to(decoder_weight.dtype)
    t0 = time.time()
    BAR.start("audio", "Audio decoding")
    try:
        out = _orig_decode_audio(latent, audio_decoder, *args, **kwargs)
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    TIMES["decode"] = TIMES.get("decode", 0.0) + time.time() - t0
    return out


ltx_distilled.vae_decode_video_to_tensor = decode_video
ltx_distilled.vae_decode_audio = decode_audio


# ==== Denoising steps ====
# The distilled pipeline ignores sampling_steps: it walks the module-level 9-sigma schedule (8 steps). Shorter
# schedules drop sigmas from the 1.0 -> 0.975 cluster, which barely moves the latent, and keep the ones that do the work.
FULL_SIGMAS = list(ltx_distilled.DISTILLED_SIGMA_VALUES)
STEP_SCHEDULES = {
    8: FULL_SIGMAS,
    7: [1.0, 0.99375, 0.9875, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    6: [1.0, 0.9875, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    5: [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    4: [1.0, 0.909375, 0.725, 0.421875, 0.0],
}


def set_denoising_steps(num_steps):
    if int(num_steps) != 8:
        raise ValueError("This test edition uses the official eight-step distilled schedule")
    schedule = FULL_SIGMAS
    ltx_distilled.DISTILLED_SIGMA_VALUES = schedule
    return len(schedule) - 1


# ==== Sizes ====
RESOLUTIONS = {"384p (fastest)": 384, "480p (recommended)": 480, "704p (high quality, slow)": 704,
               "1088p (very slow)": 1088}
ASPECTS = {"16:9 Landscape": 16 / 9, "9:16 Portrait": 9 / 16, "1:1 Square": 1.0, "4:3 Standard": 4 / 3,
           "3:4 Portrait": 3 / 4}
DURATIONS = {"2 s (49 frames)": 49, "3 s (73 frames)": 73, "5 s (121 frames)": 121, "8 s (193 frames)": 193,
             "10 s (241 frames)": 241}


def snap64(value):
    # LTX rounds width / height up to a multiple of 64 before building the latent grid: snap here as well, so the start
    # frame is prepared at exactly the size the video is generated at.
    return max(64, int(round(float(value) / 64.0)) * 64)


def size_for(resolution, aspect):
    base = RESOLUTIONS.get(resolution, 480)
    ratio = ASPECTS.get(aspect, 16 / 9)
    if ratio >= 1.0:
        return snap64(base * ratio), snap64(base)
    return snap64(base), snap64(base / ratio)


def fit_to_image(image, width, height):
    # Image-to-Video: follow the frame's own aspect ratio at the preset's pixel budget, so nothing is cropped.
    img_w, img_h = image.size
    scale = ((width * height) / float(img_w * img_h)) ** 0.5
    return snap64(img_w * scale), snap64(img_h * scale)


def prepare_conditioning_image(image, width, height):
    # Centre-crop to the target ratio, then resample to exactly W x H.
    img_w, img_h = image.size
    target, src = width / float(height), img_w / float(img_h)
    if src > target:
        new_w = int(round(img_h * target))
        left = (img_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, img_h))
    elif src < target:
        new_h = int(round(img_w / target))
        top = (img_h - new_h) // 2
        image = image.crop((0, top, img_w, top + new_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def vae_tile_size(width, height):
    # Whole-frame decode up to 1024 px (tiles leave seams); above that a full-frame decode does not fit a T4.
    longest = max(int(width), int(height))
    return 0 if longest <= 1024 else (512 if longest <= 1536 else 256)


@torch.no_grad()
def save_mp4(video, audio, sample_rate, path, fps=FPS):
    """Streams the decoded uint8 [3, T, H, W] clip and its audio to ffmpeg in one pass."""
    import soundfile as sf

    if not bool(torch.isfinite(video).all()):
        raise RuntimeError("Decoded video contains NaN/Inf; export refused")
    if audio is not None and not np.isfinite(np.asarray(audio)).all():
        raise RuntimeError("Decoded audio contains NaN/Inf; export refused")
    _, t, h, w = video.shape
    wav = path[:-4] + ".wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    if audio is not None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
            audio = audio.T  # soundfile wants [samples, channels]
        sf.write(wav, audio, int(sample_rate))
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # no preexec_fn: a real fork() would copy the pinned RAM
    try:
        frames = video if video.dtype == torch.uint8 else video.clamp(0, 255).to(torch.uint8)
        for start in range(0, t, 16):
            proc.stdin.write(frames[:, start:start + 16].permute(1, 2, 3, 0).contiguous().numpy().tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    if os.path.exists(wav):
        os.remove(wav)


# ==== Tensor parallel transformer: every block computed by both T4s at once ====
# The INT8 transformer (19.5 GB) does not fit one T4, and streaming the missing half from RAM at every step costs
# about as much as the compute. Split in two, it fits both GPUs with nothing left to stream: every attention keeps
# half of its heads on each GPU and every feed-forward half of its hidden units. Q/K/V, the gates and the FFN input
# projection are split by output rows; the attention output and FFN output projections by input columns, on 256-wide
# ConvRot rotation groups, so each INT8 GEMM computes exactly its share of the product. Each GPU runs an unchanged
# copy of the LTX block in its own thread; the two copies meet only to add their partial outputs (and the sums of
# squares of the q/k RMS norms, which span all heads), so both hold the same residual stream throughout.
import copy
import dataclasses
import queue
import types

from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
from shared.attention import bfloat16_supported, pay_attention
from shared.qtypes.int8_convrot import _rotate_activation
from models.ltx2.ltx_core.model.transformer import rope as ltx_rope
from models.ltx2.ltx_core.model.transformer.rope import apply_rotary_emb_inplace

_TP_ATTENTIONS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
_TP_FEEDFORWARDS = ("ff", "audio_ff")
# Overlapped exchange: the big partial sums (attention outputs, feed-forward outputs) are computed in row pieces, and
# each piece travels to the other GPU (via pinned host buffers, on side streams) while the next one is computed.
# LTX25_TP_OVERLAP=0 turns it off (one blocking exchange per layer, as both GPUs wait for the copy).
TP_OVERLAP = os.environ.get("LTX25_TP_OVERLAP", "1") != "0"
_PIPE_SLOT_BYTES = 32 << 20     # one pinned host buffer per piece in flight
_PIPE_SLOTS = 8                 # per GPU (ring)
_PIPE_MIN_BYTES = 4 << 20       # smaller partials (the audio stream) go through the plain exchange
_PIPE_MIN_ROWS = 512


def _piece_rows(tokens, width, element_size):
    """Rows per piece: about a quarter of the tokens (so 3/4 of the transfer hides behind compute), at least 512 rows
    for efficient GEMMs, at most what one host buffer holds."""
    rows = max(_PIPE_MIN_ROWS, -(-tokens // 4))
    return max(1, min(rows, _PIPE_SLOT_BYTES // (width * element_size), tokens))


def _local_split_groups(cache, head0, heads, half_dim):
    """The cache's own split groups restricted to the heads head0 .. head0 + heads - 1 (indexed from 0 in the local
    tensor). A group covers the heads head_start + m * axes (m < head_count) at the same local positions, with frequency
    freq_start + m * half_dim: the local part of it is a shorter group of the same shape, so the rotation is the same
    as the cache's, head by head (including the cache's handling of the padded frequencies)."""
    axes = len(cache.axes)
    groups, max_heads, max_width = [], 0, 0
    for axis_groups in cache.split_groups:
        local = []
        for g in axis_groups:
            first = max(0, -(-(head0 - g.head_start) // axes))          # first m with head >= head0
            last = min(g.head_count - 1, (head0 + heads - 1 - g.head_start) // axes)  # last m with head < head0 + heads
            if last < first:
                continue
            local.append(ltx_rope.RopeSplitGroup(head_start=g.head_start + first * axes - head0, head_count=last - first + 1,
                                                 local_start=g.local_start, freq_start=g.freq_start + first * half_dim,
                                                 freq_width=g.freq_width))
            max_heads, max_width = max(max_heads, last - first + 1), max(max_width, g.freq_width)
        groups.append(tuple(local))
    return tuple(groups), max_heads, max_width


def _rope_split_local(x, cache, head0, heads, dim_head):
    """Split RoPE on this GPU's heads only: the rotation loop of ltx_rope._apply_split_rope_layout_inplace with the
    groups of the local head range (the original rotates all heads, so each GPU would rotate a zero-padded full-width
    copy). Returns False for anything it does not handle; the caller then uses the full-width path."""
    if (not isinstance(cache, ltx_rope.RopeCache) or cache.rope_type != ltx_rope.LTXRopeType.SPLIT
            or not cache.layouts or not cache.split_groups or not cache.rope_axes or dim_head % 2
            or cache.num_attention_heads is None or torch.compiler.is_compiling()):
        return False
    half_dim, axes = dim_head // 2, len(cache.axes)
    axis_width = cache.axes[0].cos.shape[-1]
    if axis_width <= 0 or axis_width * axes != cache.num_attention_heads * half_dim - cache.pad_size:
        return False
    if x.shape[1] != cache.token_count:  # the original leaves such a tensor unrotated
        return True
    if not x.is_contiguous() or (x.shape[0] > 1 and len(cache.layouts) > 1):  # token slices would not be views
        return False
    groups, max_heads, max_width = _local_split_groups(cache, head0, heads, half_dim)
    work = torch.float32 if cache.use_fp32_freqs else x.dtype
    for layout in cache.layouts:
        segment = x[:, layout.token_start:layout.token_stop]
        if math.prod(layout.grid_sizes) != segment.shape[1]:
            continue
        view = segment.view(segment.shape[0], *layout.grid_sizes, heads, dim_head)
        x0, x1 = view[..., :half_dim], view[..., half_dim:]
        buffers = ltx_rope._make_axis_buffers_for_shape((*x0.shape[:-2], max_heads, max_width), x.dtype, x.device, work)
        for axis_index, axis in enumerate(cache.rope_axes):
            cos_axis = layout.axis_cos[axis_index].to(device=x.device, dtype=work)
            sin_axis = layout.axis_sin[axis_index].to(device=x.device, dtype=work)
            for g in groups[axis_index]:
                x0g = x0[..., g.head_start::axes, g.local_start::axes][..., :g.head_count, :g.freq_width]
                x1g = x1[..., g.head_start::axes, g.local_start::axes][..., :g.head_count, :g.freq_width]
                cos_g = ltx_rope._broadcast_group_freqs(ltx_rope._split_group_freq_view(cos_axis, g, half_dim), axis,
                                                        len(layout.grid_sizes))
                sin_g = ltx_rope._broadcast_group_freqs(ltx_rope._split_group_freq_view(sin_axis, g, half_dim), axis,
                                                        len(layout.grid_sizes))
                ltx_rope._apply_axis_rotation(x0g, x1g, cos_g, sin_g, work,
                                              ltx_rope._slice_axis_buffers(buffers, g.head_count, g.freq_width))
    return True


class TPExchange:
    """The meeting point of the per-GPU threads: each hands in tensors and gets the other's, copied to its own GPU."""

    def __init__(self, world):
        if world != 2:
            raise ValueError("The tensor-parallel split is written for two GPUs")
        self.world = world
        self.barrier = threading.Barrier(world, timeout=900)
        self.slots = [None] * world
        self.flag = [False]
        self.overlap = False  # set by setup_overlap once the GPU self-test passes

    def setup_overlap(self, devices):
        self.host = [[torch.empty(_PIPE_SLOT_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(_PIPE_SLOTS)]
                     for _ in devices]
        self.send_streams = [torch.cuda.Stream(d) for d in devices]
        self.recv_streams = [torch.cuda.Stream(d) for d in devices]
        self.d2h_done = [[None] * _PIPE_SLOTS for _ in devices]  # rank r's piece is in host[r][slot]
        self.h2d_done = [[None] * _PIPE_SLOTS for _ in devices]  # the other GPU has finished reading host[r][slot]
        self.seq = [0] * len(devices)

    def pipeline(self, rank, piece_bytes, pieces):
        """A partial sum computed in pieces: overlapped when the pieces are large enough to be worth it."""
        overlap = self.overlap and piece_bytes * pieces >= _PIPE_MIN_BYTES and piece_bytes <= _PIPE_SLOT_BYTES
        return _OverlapPipe(self, rank) if overlap else _SyncPipe(self, rank)

    def swap(self, rank, tensors):
        self.slots[rank] = tensors
        self.barrier.wait()
        device = tensors[0].device
        others = [[t.to(device, non_blocking=t.is_cuda and device.type == "cuda", copy=True) for t in self.slots[r]]
                  for r in range(self.world) if r != rank]
        # A cross-GPU copy runs on the source GPU's stream: once both copies are queued, either side may reuse its buffers.
        self.barrier.wait()
        return others

    def add_others(self, rank, tensors):
        """tensors[i] += the other GPU's tensors[i], in place. With two GPUs both get bit-identical sums (a + b == b + a)."""
        for i, other in enumerate(self.swap(rank, tensors)[0]):
            tensors[i].add_(other)
        return tensors


class _SyncPipe:
    """Pieces kept until finish(), then summed with one blocking exchange (CPU tests, small tensors, overlap off)."""

    def __init__(self, exchange, rank):
        self.exchange, self.rank, self.pieces = exchange, rank, []

    def send(self, piece):
        self.pieces.append(piece)

    def finish(self):
        return self.exchange.add_others(self.rank, self.pieces)


class _OverlapPipe:
    """Each piece goes GPU -> pinned host buffer (send stream) and the other GPU's piece comes host -> GPU (receive
    stream) while the compute stream goes on with the next piece; finish() waits for the copies and adds. Both GPUs
    send the same number of pieces of the same shapes, in the same order (slot ring indices stay in step)."""

    def __init__(self, exchange, rank):
        self.ex, self.rank, self.items = exchange, rank, []

    def send(self, piece):
        ex, rank = self.ex, self.rank
        other = 1 - rank
        compute = torch.cuda.current_stream(piece.device)
        nbytes = piece.numel() * piece.element_size()
        slot = ex.seq[rank] % _PIPE_SLOTS
        ex.seq[rank] += 1
        send, recv = ex.send_streams[rank], ex.recv_streams[rank]
        send.wait_stream(compute)  # the piece is computed
        if ex.h2d_done[rank][slot] is not None:
            send.wait_event(ex.h2d_done[rank][slot])  # the other GPU has read what this buffer held before
        mine = ex.host[rank][slot][:nbytes].view(piece.dtype).view(piece.shape)
        with torch.cuda.stream(send):
            mine.copy_(piece, non_blocking=True)
        piece.record_stream(send)
        sent = torch.cuda.Event()
        sent.record(send)
        ex.d2h_done[rank][slot] = sent
        ex.barrier.wait()  # both pieces are on their way to host memory
        theirs = ex.host[other][slot][:nbytes].view(piece.dtype).view(piece.shape)
        recv.wait_event(ex.d2h_done[other][slot])
        with torch.cuda.stream(recv):
            received = torch.empty_like(piece)
            received.copy_(theirs, non_blocking=True)
        received.record_stream(compute)
        got = torch.cuda.Event()
        got.record(recv)
        ex.h2d_done[other][slot] = got
        self.items.append((piece, received, sent, got))

    def finish(self):
        compute = torch.cuda.current_stream(self.items[0][0].device)
        for piece, received, sent, got in self.items:
            compute.wait_event(sent)  # this GPU's copy of the piece is out before the piece is summed in place
            compute.wait_event(got)
            piece.add_(received)      # a + b on one GPU, b + a on the other: bit-identical sums
        return [item[0] for item in self.items]


class TPRunner:
    """One thread per GPU, kept for the whole session (the current CUDA device and grad mode are per thread)."""

    def __init__(self, devices):
        self.devices = devices
        self.exchange = TPExchange(len(devices))
        self.jobs = [queue.Queue() for _ in devices]
        self.results = queue.Queue()
        for rank in range(len(devices)):
            threading.Thread(target=self._loop, args=(rank,), daemon=True, name=f"tp-gpu{rank}").start()

    def _loop(self, rank):
        if self.devices[rank].type == "cuda":
            torch.cuda.set_device(self.devices[rank])
        while True:
            fn, inference = self.jobs[rank].get()
            try:
                # inference / grad mode are per thread: use the caller's (Wan2GP generates under inference_mode, and
                # in-place updates of its inference tensors are refused outside it)
                with torch.inference_mode(inference), torch.no_grad():
                    self.results.put((rank, True, fn(rank)))
            except BaseException as e:
                self.exchange.barrier.abort()  # release the other thread if it waits at the exchange
                self.results.put((rank, False, e))

    def run(self, fn):
        inference = torch.is_inference_mode_enabled()
        for q in self.jobs:
            q.put((fn, inference))
        outputs, errors = [None] * len(self.devices), []
        for _ in self.devices:
            rank, ok, value = self.results.get()
            if ok:
                outputs[rank] = value
            else:
                errors.append(value)
        if errors:
            self.exchange.barrier.reset()
            seq = getattr(self.exchange, "seq", None)
            if seq:  # a GPU that failed mid-layer sent fewer pieces: both continue from the same host buffer
                self.exchange.seq = [max(seq)] * len(seq)
            real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
            raise (real or errors)[0]
        return outputs


_LORA_LAYERS = []  # ShardLinear layers that carry a LoRA


def park_loras():
    """Moves the live LoRA weights (~1.45 GB per GPU for MSR V2) to pinned RAM while the VAE decodes, so the decoder gets
    whole-frame chunks. Pinned copies are made once and kept; each layer pulls its weights back on its next forward."""
    parked = 0
    for layer in _LORA_LAYERS:
        if layer.lora_a is None:
            continue
        if layer.lora_parked is None:
            pin = layer.lora_a.is_cuda  # pinned host memory needs CUDA
            layer.lora_parked = tuple(t.cpu().pin_memory() if pin else t.cpu().clone() for t in (layer.lora_a, layer.lora_b))
        parked += layer.lora_a.numel() + layer.lora_b.numel()
        layer.lora_a = layer.lora_b = None
    if parked and torch.cuda.is_available():
        for device in {MAIN_DEV, AUX_DEV}:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return parked * 2


class ShardLinear(torch.nn.Module):
    """One GPU's share of a linear layer: a range of output rows, or a range of input columns (no bias: it is added once,
    after the partial outputs are summed). INT8 ConvRot weights stay INT8 and run on Kitchen's INT8 GEMM."""

    def __init__(self, linear, device, rows=None, cols=None, lora=None):
        super().__init__()
        weight = getattr(linear, "qweight", None)
        weight = weight if isinstance(weight, WeightQBytesTensor) else linear.weight
        rows = slice(None) if rows is None else slice(*rows)
        cols = slice(None) if cols is None else slice(*cols)
        self.group = int(getattr(linear, "_convrot_group_size", 0) or 0)
        if isinstance(weight, WeightQBytesTensor):
            data, scale = weight._data, weight._scale.reshape(-1)
            self.register_buffer("qdata", data[rows, cols].contiguous().to(device, copy=True), persistent=False)
            self.register_buffer("qscale", scale[rows].float().contiguous().to(device, copy=True), persistent=False)
            self.dense = None
        else:
            self.qdata = self.qscale = None
            self.register_buffer("dense", weight.detach()[rows, cols].contiguous().to(device, copy=True), persistent=False)
        bias = linear.bias if cols == slice(None) else None
        self.register_buffer("bias", None if bias is None else bias.detach()[rows].contiguous().to(device, copy=True),
                             persistent=False)
        source = self.dense if self.qdata is None else self.qdata
        self.out_features, self.in_features = source.shape
        if self.group and self.in_features % self.group:
            raise ValueError(f"ConvRot shard of {self.in_features} columns splits a {self.group}-wide rotation group")
        # LoRA (A: rank x in, B: out x rank), applied live: merging it into INT8 weights would round most of it away.
        # Row shards keep all of A and their rows of B; column shards their columns of A and all of B (their partial
        # products are summed with the base output). The LoRA's input basis is the original one: Kitchen rotates
        # (ConvRot) only inside the INT8 GEMM.
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)
        self.lora_parked = None  # pinned RAM copies while the LoRA is parked (see park_loras)
        self.device = torch.device(device)
        if lora is not None:
            lora_a, lora_b = lora
            self.lora_a = lora_a[:, cols].to(device=device, dtype=PIPELINE_DTYPE).contiguous()
            self.lora_b = lora_b[rows].to(device=device, dtype=PIPELINE_DTYPE).contiguous()
            _LORA_LAYERS.append(self)
            if self.lora_a.shape[1] != self.in_features or self.lora_b.shape[0] != self.out_features:
                raise ValueError(f"LoRA shapes {tuple(lora_a.shape)} / {tuple(lora_b.shape)} do not match the layer")

    def forward(self, x):
        x2 = x.reshape(-1, self.in_features).contiguous()
        if self.qdata is None:
            bias = None if self.bias is None else self.bias.to(x2.dtype)
            out = torch.nn.functional.linear(x2, self.dense.to(x2.dtype), bias)
        elif int8_backend.kitchen_enabled() and x2.is_cuda and self.group in (0, 256):
            out = int8_backend._linear_impl(x2, self.qdata, self.qscale, self.bias, self.group == 256)
        else:  # exact reference path: rotate, dequantize, matmul
            weight = self.qdata.to(x2.dtype) * self.qscale.to(x2.dtype)[:, None]
            bias = None if self.bias is None else self.bias.to(x2.dtype)
            out = torch.nn.functional.linear(_rotate_activation(x2, self.group), weight, bias)
        if self.lora_a is None and self.lora_parked is not None:  # back from pinned RAM after a VAE decode
            self.lora_a = self.lora_parked[0].to(self.device, non_blocking=True)
            self.lora_b = self.lora_parked[1].to(self.device, non_blocking=True)
        if self.lora_a is not None:
            # FP16 tensor cores; the input is scaled down by 2^4 first so x @ A cannot overflow FP16.
            low = torch.mm(x2.to(self.lora_a.dtype) * 0.0625, self.lora_a.t())
            out.addmm_(low.to(out.dtype), self.lora_b.t().to(out.dtype), alpha=16.0)
        return out.reshape(*x.shape[:-1], self.out_features)


class TPAttention(torch.nn.Module):
    """Attention.forward of LTX on this GPU's heads: q/k/v/gate shards, q/k RMS norm over all heads, RoPE, SDPA, output
    projection shard; the partial outputs of the GPUs are summed before the bias."""

    def __init__(self, attn, rank, device, exchange, lora=None):
        super().__init__()
        lora = lora or {}
        world = exchange.world
        if attn.heads % world:
            raise ValueError(f"{attn.heads} heads cannot be split over {world} GPUs")
        self.rank, self.exchange = rank, exchange
        self.dim_head, self.inner = attn.dim_head, attn.heads * attn.dim_head
        self.heads = attn.heads // world
        h0 = rank * self.heads
        self.cols = (h0 * self.dim_head, (h0 + self.heads) * self.dim_head)
        self.to_q = ShardLinear(attn.to_q, device, rows=self.cols, lora=lora.get("to_q"))
        self.to_k = ShardLinear(attn.to_k, device, rows=self.cols, lora=lora.get("to_k"))
        self.to_v = ShardLinear(attn.to_v, device, rows=self.cols, lora=lora.get("to_v"))
        self.to_gate_logits = (None if attn.to_gate_logits is None
                               else ShardLinear(attn.to_gate_logits, device, rows=(h0, h0 + self.heads)))
        out = attn.to_out[0]
        self.to_out = ShardLinear(out, device, cols=self.cols, lora=lora.get("to_out.0"))
        self.register_buffer("out_bias", None if out.bias is None else out.bias.detach().to(device, copy=True), persistent=False)
        self.register_buffer("q_weight", attn.q_norm.weight.detach()[self.cols[0]:self.cols[1]].to(device, copy=True), persistent=False)
        self.register_buffer("k_weight", attn.k_norm.weight.detach()[self.cols[0]:self.cols[1]].to(device, copy=True), persistent=False)
        self.eps = attn.q_norm.eps
        self.rope_type = attn.rope_type

    def _rope(self, t, freqs):
        # Split RoPE caches: rotate this GPU's heads in place. Anything else: rotate a full-width tensor holding this
        # GPU's heads (rotation pairs never cross heads).
        if _rope_split_local(t, freqs, self.cols[0] // self.dim_head, self.heads, self.dim_head):
            return t
        full = t.new_zeros((*t.shape[:-1], self.inner))
        full[..., self.cols[0]:self.cols[1]] = t
        apply_rotary_emb_inplace(full, freqs, self.rope_type)
        return full[..., self.cols[0]:self.cols[1]].contiguous()

    def forward(self, x_list, context_list=None, mask=None, pe=None, k_pe=None, NAG=None):
        x = x_list[0]
        x_list.clear()
        context = None
        if context_list is not None:
            context = context_list[0]
            context_list.clear()
        if NAG is not None and context is not None:
            raise NotImplementedError("NAG is not supported by the tensor-parallel transformer")
        gate = None if self.to_gate_logits is None else self.to_gate_logits(x)
        source = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(source), self.to_v(source)
        x = source = context = None
        sq = torch.linalg.vector_norm(q, ord=2, dim=-1, keepdim=True, dtype=torch.float32).square_()
        sk = torch.linalg.vector_norm(k, ord=2, dim=-1, keepdim=True, dtype=torch.float32).square_()
        self.exchange.add_others(self.rank, [sq, sk])
        q.mul_(sq.div_(self.inner).add_(self.eps).rsqrt_()).mul_(self.q_weight)
        k.mul_(sk.div_(self.inner).add_(self.eps).rsqrt_()).mul_(self.k_weight)
        sq = sk = None
        if pe is not None:
            q = self._rope(q, pe)
            k = self._rope(k, pe if k_pe is None else k_pe)
        out_dtype = q.dtype
        q, k, v = (t.view(t.shape[0], t.shape[1], self.heads, self.dim_head) for t in (q, k, v))
        if q.dtype == torch.bfloat16 and q.is_cuda and not bfloat16_supported:
            q, k, v = q.half(), k.half(), v.half()  # once, not once per piece (Turing attends in FP16)
        # Query pieces: the attention of piece i + 1 runs while the output projection of piece i is on its way to the
        # other GPU. Keys and values are shared by all pieces (same result as one call). Masks are laid out
        # [batch, queries or 1, heads or 1, keys]: sliced per piece, or shared when they broadcast over the queries.
        tokens, width = q.shape[1], self.to_out.out_features
        sliceable = mask is None or (mask.dim() == 4 and mask.shape[1] in (1, tokens))
        rows = _piece_rows(tokens, width, torch.finfo(out_dtype).bits // 8) if sliceable else tokens
        pipe = self.exchange.pipeline(self.rank, q.shape[0] * rows * width * (torch.finfo(out_dtype).bits // 8),
                                      -(-tokens // rows))
        for start in range(0, tokens, rows):
            piece_mask = mask if mask is None or mask.shape[1] == 1 else mask[:, start:start + rows]
            out = pay_attention([q[:, start:start + rows], k, v], attention_mask=piece_mask).to(out_dtype)
            if gate is not None:
                out.mul_((2.0 * torch.sigmoid(gate[:, start:start + rows])).to(dtype=out.dtype).unsqueeze(-1))
            pipe.send(self.to_out(out.flatten(2, 3)))
            out = None
        q = k = v = gate = None
        pieces = pipe.finish()
        partial = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        pieces = None
        if self.out_bias is not None:
            partial.add_(self.out_bias.to(partial.dtype))
        return partial


class TPFeedForward(torch.nn.Module):
    """FeedForward of LTX on this GPU's half of the hidden units (GELU-tanh in between)."""

    def __init__(self, ff, rank, device, exchange, lora=None):
        super().__init__()
        lora = lora or {}
        up, down = ff.net[0].proj, ff.net[2]
        world = exchange.world
        hidden = up.out_features if hasattr(up, "out_features") else up.weight.shape[0]
        part = hidden // world
        self.rank, self.exchange, self.mult = rank, exchange, ff.mult
        span = (rank * part, (rank + 1) * part)
        self.up = ShardLinear(up, device, rows=span, lora=lora.get("net.0.proj"))
        self.down = ShardLinear(down, device, cols=span, lora=lora.get("net.2"))
        self.register_buffer("down_bias", None if down.bias is None else down.bias.detach().to(device, copy=True), persistent=False)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        tokens, width = flat.shape[0], self.down.out_features
        rows = _piece_rows(tokens, width, flat.element_size())
        pipe = self.exchange.pipeline(self.rank, rows * width * flat.element_size(), -(-tokens // rows))
        for start in range(0, tokens, rows):  # also bounds the hidden activations to one piece
            h = self.up(flat[start:start + rows])
            for chunk in torch.split(h, max(1, (64 << 20) // (h.shape[-1] * h.element_size()))):
                chunk.copy_(torch.nn.functional.gelu(chunk, approximate="tanh"))
            pipe.send(self.down(h))
            h = None
        pieces = pipe.finish()
        partial = (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)).view(*x.shape[:-1], width)
        pieces = None
        if self.down_bias is not None:
            partial.add_(self.down_bias.to(partial.dtype))
        return partial


def _replicate_block(block, rank, device, exchange, lora=None):
    """A copy of an LTX block for one GPU, with its attentions / feed-forwards split. `lora` maps a module path inside
    the block ("attn1.to_q", "ff.net.2", ...) to its (A, B) LoRA weights."""
    lora = lora or {}

    def module_lora(name):
        return {key[len(name) + 1:]: value for key, value in lora.items() if key.startswith(name + ".")}

    replica = type(block).__new__(type(block))
    torch.nn.Module.__init__(replica)
    for key, value in block.__dict__.items():  # idx, norm_eps, cross_attention_adaln, ff_chunk_min_tokens, ...
        if not key.startswith("_") and key != "training":
            replica.__dict__[key] = value
    for name, p in block._parameters.items():
        replica._parameters[name] = None if p is None else torch.nn.Parameter(p.detach().to(device, copy=True), requires_grad=False)
    for name, b in block._buffers.items():
        replica._buffers[name] = None if b is None else b.to(device, copy=True)
    for name, module in block._modules.items():
        if module is None:
            replica._modules[name] = None
        elif name in _TP_ATTENTIONS:
            replica._modules[name] = TPAttention(module, rank, device, exchange, lora=module_lora(name))
        elif name in _TP_FEEDFORWARDS:
            replica._modules[name] = TPFeedForward(module, rank, device, exchange, lora=module_lora(name))
        else:
            raise RuntimeError(f"Unexpected module '{name}' in an LTX block: the tensor-parallel split does not know it")
    replica.ff_chunk_min_tokens = 1 << 62  # TPFeedForward splits the rows itself (and overlaps the exchange)
    replica.eval()
    return replica


def _to_device(value, device):
    """A copy of the block inputs (TransformerArgs, RoPE caches, masks) on another GPU."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=value.is_cuda and device.type == "cuda", copy=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _to_device(getattr(value, f.name), device)
                                             for f in dataclasses.fields(value) if f.init})
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _tp_process_blocks(self, video, audio, perturbations):
    runner, replicas, before = self._tp_runner, self._tp_replicas, self._tp_before_blocks
    if before is not None:
        before()
    inputs = [(video, audio)] + [(_to_device(video, d), _to_device(audio, d)) for d in runner.devices[1:]]
    flag = runner.exchange.flag
    flag[0] = False

    def work(rank):
        v, a = inputs[rank]
        for blocks in replicas[rank]:
            if rank == 0:
                flag[0] = bool(self.interrupt_check is not None and self.interrupt_check())
            runner.exchange.barrier.wait()  # every GPU sees rank 0's decision before the block
            if flag[0]:
                return None, None
            v, a = blocks(video=v, audio=a, perturbations=perturbations)
        return v, a

    outputs = runner.run(work)
    if flag[0]:
        self.interrupted = True
        return None, None
    return outputs[0]


def _tp_joint_blocks(self, video_list, audio_list, perturbations):
    """Joint pass (modality guidance: the prompt pass + a pass without audio <-> video cross-attention): the
    conditions run one after the other through the tensor-parallel blocks."""
    per_condition = perturbations if isinstance(perturbations, list) else [perturbations] * len(video_list)
    videos, audios = [], []
    for v, a, p in zip(video_list, audio_list, per_condition):
        v_out, a_out = _tp_process_blocks(self, v, a, p)
        if self.interrupted:
            return [None] * len(video_list), [None] * len(audio_list)
        videos.append(v_out)
        audios.append(a_out)
    return videos, audios


def _enable_overlap(runner):
    """Sets up the overlapped exchange and checks it on the GPUs before use: random pieces, each produced behind a
    long matmul (a missing stream wait shows up as wrong sums), summed by the pipeline and compared with the CPU sum.
    On any mismatch or error the blocking exchange is kept. Also reports the GPU <-> GPU transfer speed."""
    ex, devices = runner.exchange, runner.devices
    try:
        ex.setup_overlap(devices)
        rows, width, count = 1024, 4096, 2 * _PIPE_SLOTS + 3  # 8 MB pieces, every host buffer reused twice
        sources = []
        for rank in range(len(devices)):
            g = torch.Generator().manual_seed(1234 + rank)
            sources.append([torch.randn(rows, width, generator=g).half() for _ in range(count)])
        expected = [(a.float() + b.float()).half() for a, b in zip(*sources)]

        def work(rank, overlap):
            dev = devices[rank]
            local = [t.to(dev) for t in sources[rank]]
            busy = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            pipe = _OverlapPipe(ex, rank) if overlap else _SyncPipe(ex, rank)
            for t in local:
                for _ in range(4):
                    busy = busy @ busy.T * 1e-3  # keep the compute stream busy before the piece is ready
                pipe.send(t.clone())
            out = pipe.finish()
            torch.cuda.synchronize(dev)
            return time.time() - t0, [t.cpu() for t in out]

        ex.overlap = True
        runner.run(lambda rank: work(rank, False))  # warm-up: cuBLAS, streams, pinned buffers
        _, outputs = zip(*runner.run(lambda rank: work(rank, True)))
        bad = sum(int(not torch.equal(o, e)) for out in outputs for o, e in zip(out, expected))
        if bad:
            raise RuntimeError(f"{bad} of {2 * count} pieces differ from the CPU sum")
        sync_seconds, _ = zip(*runner.run(lambda rank: work(rank, False)))
        seconds, _ = zip(*runner.run(lambda rank: work(rank, True)))
        gb = rows * width * 2 * count / 1024**3
        peer = torch.cuda.can_device_access_peer(devices[0].index, devices[1].index)
        print(f"  ⚡ Overlapped GPU exchange on: {gb:.1f} GB each way in {max(seconds):.2f}s with compute running "
              f"(blocking exchange {max(sync_seconds):.2f}s) | direct GPU-to-GPU access: {'yes' if peer else 'no'}")
    except Exception as e:
        ex.overlap = False
        runner.exchange.barrier.reset()
        print(f"  ⚠️ Overlapped GPU exchange disabled ({e}): using the blocking exchange")


class BlockLoras:
    """LoRA weights of a safetensors file grouped by transformer block, read block by block (the file is memory-mapped,
    so only one block's tensors are in RAM at a time). Keys: [diffusion_model.]transformer_blocks.N.<module>.lora_A/B."""

    def __init__(self, path, multiplier=1.0):
        from safetensors import safe_open
        self.file = safe_open(path, framework="pt", device="cpu")
        self.multiplier = float(multiplier)
        self.blocks = {}
        pattern = re.compile(r"^(?:diffusion_model\.)?transformer_blocks\.(\d+)\.(.+)\.lora_([AB])\.weight$")
        for key in self.file.keys():
            match = pattern.match(key)
            if match:
                self.blocks.setdefault(int(match.group(1)), {}).setdefault(match.group(2), {})[match.group(3)] = key
        alphas = [key for key in self.file.keys() if key.endswith(".alpha")]
        if alphas:
            raise ValueError("LoRA alpha tensors are not supported by the tensor-parallel loader")
        self.modules = sum(len(mods) for mods in self.blocks.values())

    def block(self, index):
        result = {}
        for module, keys in self.blocks.get(index, {}).items():
            if set(keys) != {"A", "B"}:
                raise ValueError(f"Incomplete LoRA pair for block {index} {module}")
            a = self.file.get_tensor(keys["A"]).float()
            b = self.file.get_tensor(keys["B"]).float() * self.multiplier
            result[module] = (a, b)
        return result


def enable_tensor_parallel(velocity_model, devices, before_blocks=None, progress=None, loras=None):
    """Splits every block of the LTX model over the devices; everything outside the blocks goes to devices[0].
    `loras`: an optional BlockLoras, applied live in each split layer."""
    runner = TPRunner(devices)
    if TP_OVERLAP and all(d.type == "cuda" for d in devices):
        _enable_overlap(runner)
    blocks = velocity_model.transformer_blocks
    replicas = [[] for _ in devices]
    for i in range(len(blocks)):
        block = blocks[i]
        block_lora = loras.block(i) if loras is not None else None
        if block_lora:
            known = {f"{name}.{sub}" for name in _TP_ATTENTIONS for sub in ("to_q", "to_k", "to_v", "to_out.0")}
            known |= {f"{name}.{sub}" for name in _TP_FEEDFORWARDS for sub in ("net.0.proj", "net.2")}
            unknown = sorted(set(block_lora) - known)
            if unknown:
                raise ValueError(f"LoRA targets the split layers cannot carry: block {i} {unknown[:4]}")
        for rank, device in enumerate(devices):
            replicas[rank].append(_replicate_block(block, rank, device, runner.exchange, lora=block_lora))
        block_lora = None
        blocks[i] = replicas[0][i]  # the original (CPU) block is released
        del block
        if progress is not None:
            progress(i + 1, len(blocks))
    velocity_model.to(devices[0])
    object.__setattr__(velocity_model, "_tp_runner", runner)          # plain attributes: the other GPUs' replicas
    object.__setattr__(velocity_model, "_tp_replicas", replicas)      # must not be moved by velocity_model.to()
    object.__setattr__(velocity_model, "_tp_before_blocks", before_blocks)
    velocity_model._process_transformer_blocks = types.MethodType(_tp_process_blocks, velocity_model)
    velocity_model._process_transformer_blocks_joint = types.MethodType(_tp_joint_blocks, velocity_model)
    return runner


# ==== Load the model (once; it stays in memory) ====
print(f"\n📦 Loading LTX-2.5 22B Distilled (INT8 ConvRot) + Gemma-4 12B (INT8 ConvRot) | compute "
      f"{'BF16' if PIPELINE_DTYPE == torch.bfloat16 else 'FP16'}...")
sys.stdout.flush()
t_load = time.time()


def extract_msr_slots(lora_path, slots_path):
    """The MSR V2 LoRA carries its learned reference-slot embedding (a small Fourier MLP, `reference_slot_embedding.*`)
    next to the LoRA weights. Wan2GP's MSR conditioning reads it from a file of its own: copy the five tensors out,
    unchanged. (The audio-slot embedding, for voice references, is not used here.)"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    prefix = "diffusion_model.reference_slot_embedding."
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        slots = {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
    expected = {"frequencies", "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    if set(slots) != expected:
        raise SystemExit(f"❌ {os.path.basename(lora_path)} has no complete reference_slot_embedding: is it an MSR LoRA?")
    save_file(slots, slots_path + ".tmp")
    os.replace(slots_path + ".tmp", slots_path)


BASE_MODEL_TYPE = "ltx2_25_22B"
model_def = {"ltx2_pipeline": "distilled", "name": "LTX-2 2.5 Distilled 22B", "architecture": BASE_MODEL_TYPE}
if APP_MODE == "msr":
    if not TENSOR_PARALLEL:
        raise SystemExit("❌ The multi-subject reference LoRA runs on the tensor-parallel transformer: "
                         "Settings → Accelerator → GPU T4 x2 (and leave LTX25_TENSOR_PARALLEL unset).")
    if not os.path.exists(MSR_SLOTS_FILE):
        extract_msr_slots(MSR_LORA_FILE, MSR_SLOTS_FILE)
    # Wan2GP's LTX-2.5 MSR model type: reference images are encoded one by one (33 or 25 repeated frames), given their
    # learned slot embedding and appended as clean tokens at negative time positions (subjects first, background last).
    BASE_MODEL_TYPE = "ltx2_25_22B_msr"
    model_def = {"ltx2_pipeline": "distilled", "name": "LTX-2 2.5 MSR Ref Distilled 22B", "architecture": BASE_MODEL_TYPE,
                 "ltx2_msr": True, "ltx2_msr_frame_count": 33}
model_def.update(family_handler.query_model_def(BASE_MODEL_TYPE, model_def))
if APP_MODE == "msr":
    model_def["ltx2_msr_slot_embeddings_file"] = MSR_SLOTS_FILE

# Loaded under no_grad, not inference_mode: inference tensors fail when touched from the generation thread.
with torch.no_grad():
    wan_model, pipe = family_handler.load_model(
        model_filename=TRANSFORMER_FILE,
        model_type="ltx2_25_22B_distilled",
        base_model_type=BASE_MODEL_TYPE,
        model_def=model_def,
        dtype=PIPELINE_DTYPE,
        VAE_dtype=torch.float16,
        text_encoder_filename=GEMMA_FILE,
    )
mmgp_kwargs = {}
if isinstance(pipe, dict) and "pipe" in pipe:
    mmgp_kwargs = pipe
    pipe = mmgp_kwargs.pop("pipe")
loras = mmgp_kwargs.pop("loras", [])
if "transformer" in pipe and "transformer" not in loras:
    loras.append("transformer")

# Single-stage generation never uses the x2 spatial upsampler (it is only loaded because the checkpoint set lists it):
# drop it, which frees its 1 GB FP16 copy in RAM.
pipe.pop("spatial_upsampler", None)
wan_model.spatial_upsampler = None
if getattr(wan_model.pipeline, "models", None) is not None:
    wan_model.pipeline.models.spatial_upsampler = None

# FP32 video encoder (Image-to-Video, one frame per generation) and audio encoder (lip-sync audio, a few seconds of mel):
# both are small and run once per video, and FP16 overflows there poison the whole clip. _lock_dtype keeps MMGP from
# folding them back to FP16.
for _key in ("video_encoder", "audio_encoder"):
    _encoder = pipe.get(_key)
    if _encoder is not None:
        _encoder.float()
        for _m in _encoder.modules():
            _m._lock_dtype = torch.float32

# Video decoder, audio decoder and vocoder in FP16 whatever the compute dtype (Turing has no fast BF16 convolutions).
for _key in ("vae", "audio_decoder", "vocoder"):
    _decoder = pipe.get(_key)
    if _decoder is not None:
        _decoder.half()

text_encoder = wan_model.text_encoder
if hasattr(text_encoder.model, "lm_head"):  # LTX only reads Gemma's hidden states
    text_encoder.model.lm_head = torch.nn.Identity()
OFFLOAD = None


def _free_gpu0_for_denoise():
    """Before the blocks run: unload what MMGP still holds on GPU 0 (text encoder / connectors / image encoder)."""
    _unload_mmgp()


if TENSOR_PARALLEL:
    # Both T4s hold half of every transformer block (~9.3 GB each): nothing is streamed during denoising. Gemma-4 then
    # lives in pinned RAM and MMGP streams it through GPU 0 once per new prompt (repeated prompts are cached).
    transformer = pipe.pop("transformer")
    if "transformer" in loras:
        loras.remove("transformer")
    velocity_model = transformer.velocity_model
    block_loras = BlockLoras(MSR_LORA_FILE) if APP_MODE == "msr" else None
    BAR.start("split", "Splitting the transformer over both GPUs", len(velocity_model.transformer_blocks), "blocks")
    try:
        enable_tensor_parallel(velocity_model, (MAIN_DEV, AUX_DEV), before_blocks=_free_gpu0_for_denoise,
                               progress=lambda done, total: BAR.update(done, total), loras=block_loras)
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    if block_loras is not None:
        print(f"  ✅ MSR V2 LoRA (LiconStudio, rank 128): {block_loras.modules} layers, applied live in FP16 on both GPUs")
    block_loras = None
    gc.collect()
    print(f"  ✅ Transformer split over both GPUs: GPU 0 {_gb(torch.cuda.memory_allocated(MAIN_DEV))}, "
          f"GPU 1 {_gb(torch.cuda.memory_allocated(AUX_DEV))} (tensor parallel, nothing streamed per step)")
elif TWO_GPUS:
    # Gemma-4 on GPU 1 for the whole session; the transformer is streamed on GPU 0.
    try:
        text_encoder.to(TE_DEV)
        pipe.pop("text_encoder", None)
        with torch.cuda.device(TE_DEV):
            torch.cuda.empty_cache()
        print(f"  ✅ Gemma-4 text encoder resident on GPU 1 ({_gb(torch.cuda.memory_allocated(TE_DEV))}, "
              f"{_gb(torch.cuda.mem_get_info(TE_DEV)[0])} free)")
    except Exception as e:
        print(f"  ⚠️ Gemma-4 stays on GPU 0 with offloading ({e})")
        text_encoder.to("cpu")
        TE_DEV = MAIN_DEV
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
gc.collect()

# Second video decoder on GPU 1 (0.8 GB, next to its transformer half): VAE chunks are decoded on both GPUs at once.
if TENSOR_PARALLEL and pipe.get("vae") is not None:
    try:
        VAE_REPLICA = copy.deepcopy(pipe["vae"]).to(AUX_DEV).eval().requires_grad_(False)  # before MMGP hooks the original
        pipe["vae"].tiled_decode = types.MethodType(_parallel_tiled_decode, pipe["vae"])
        print(f"  ✅ Video decoder copy on GPU 1: VAE chunks decoded on both GPUs ({_gb(torch.cuda.mem_get_info(AUX_DEV)[0])} free on GPU 1)")
    except Exception as e:
        VAE_REPLICA = None
        print(f"  ⚠️ VAE decodes on GPU 0 only ({e})")
        with torch.cuda.device(AUX_DEV):
            torch.cuda.empty_cache()

# MMGP for everything else on GPU 0. Without tensor parallelism, the INT8 transformer (19.5 GB) does not fit a 15 GB
# T4: MMGP keeps as many blocks in VRAM as the budget allows and streams the others from pinned RAM at every step.
# The budget leaves ~4.5 GB for activations (enough for 1088p).
_free0_mb = int(torch.cuda.mem_get_info(MAIN_DEV)[0] / 1024**2)
budgets = {"video_encoder": 3000, "vae": 3000, "*": 1000}
if "transformer" in pipe:
    budgets["transformer"] = int(max(6000, min(10500, _free0_mb - 4500)))
    print(f"  ⚡ MMGP: transformer budget {budgets['transformer']} MB in VRAM, the rest streamed from pinned RAM")
if "text_encoder" in pipe:
    budgets["text_encoder"] = int(max(1000, min(8000, _free0_mb - (2500 if TENSOR_PARALLEL else 6000))))
pinned = "transformer" if "transformer" in pipe else ("text_encoder" if "text_encoder" in pipe else False)
sys.stdout.flush()
OFFLOAD = offload.profile(
    pipe,
    profile_no=4,
    pinnedMemory=pinned,
    partialPinning=True,
    # reserved (pinned) RAM: the transformer's streamed half, or Gemma-4 (12.9 GB) when the transformer is split
    perc_reserved_mem_max=0.45 if TENSOR_PARALLEL else (0.40 if TWO_GPUS else 0.35),
    asyncTransfers=True,
    quantizeTransformer=False,
    convertWeightsFloatTo=PIPELINE_DTYPE,
    loras=loras,
    budgets=budgets,
    **mmgp_kwargs,
)
offload.shared_state["_attention"] = "sdpa"
offload.shared_state["_radial"] = False
torch.cuda.empty_cache()
print(f"✅ Model ready in {time.time() - t_load:.0f}s | {INT8_LABEL} | RAM {_ram_in_use()}")
if os.environ.get("LTX25_PROFILE", "0") == "1":
    print("🔬 Profiling on: denoising step 2 of the first video is traced, and a summary is printed after it")
sys.stdout.flush()


# ==== Generation ====
PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": ""}

# LTX25_PROFILE=1: the 2nd denoising step of the first video is traced on both GPUs and summarised (GPU busy / idle
# time and where the busy time goes). Costs a few seconds once; off by default.
PROFILE = {"wanted": os.environ.get("LTX25_PROFILE", "0") == "1", "prof": None, "t0": 0.0, "samples": [], "sampling": False}
_SMI_FIELDS = "index,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu,clocks_event_reasons.active"


def _sample_gpus():
    """nvidia-smi every 0.5 s while the profiled step runs: clocks, temperature, power, throttle reasons per GPU."""
    while PROFILE["sampling"]:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            PROFILE["samples"].append([line.split(", ") for line in out.strip().splitlines()])
        except Exception:
            pass
        time.sleep(0.5)


def _gpu_telemetry():
    rows = {}
    for sample in PROFILE["samples"]:
        for fields in sample:
            if len(fields) >= 8:
                rows.setdefault(fields[0], []).append(fields)
    lines = []
    for index, samples in sorted(rows.items()):
        def avg(i):
            values = [float(f[i]) for f in samples if f[i].replace(".", "", 1).isdigit()]
            return sum(values) / len(values) if values else float("nan")
        reasons = sorted({f[7] for f in samples})
        lines.append(f"GPU {index}: {avg(1):.0f}°C | SM {avg(2):.0f} of {avg(3):.0f} MHz | {avg(4):.0f} of {avg(5):.0f} W | "
                     f"util {avg(6):.0f}% | throttle reasons {','.join(reasons)}")
    return lines
_KERNEL_GROUPS = (("attention", ("fmha", "attention", "flash", "efficient", "softmax")),
                  ("INT8 GEMM", ("gemm", "xmma", "imma", "cutlass", "i8i8", "s8")),
                  ("INT8 quantize / rotate", ("quant", "convrot", "hadamard", "rotate", "int8")),
                  ("norm / scale / gate / add", ("norm", "elementwise", "vectorized", "reduce", "add", "mul", "fill",
                                                 "copy", "cat", "gelu", "sigmoid", "rope", "index")))


def _profile_begin():
    from torch.profiler import ProfilerActivity, profile
    for dev in {MAIN_DEV, AUX_DEV}:
        torch.cuda.synchronize(dev)
    PROFILE["prof"] = profile(activities=[ProfilerActivity.CUDA])
    PROFILE["prof"].start()
    PROFILE["t0"] = time.time()
    PROFILE["samples"], PROFILE["sampling"] = [], True
    threading.Thread(target=_sample_gpus, daemon=True).start()


def _union(intervals):
    total, end = 0.0, -1.0
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def _profile_end():
    import json
    prof, PROFILE["prof"], PROFILE["wanted"] = PROFILE["prof"], None, False
    for dev in {MAIN_DEV, AUX_DEV}:
        torch.cuda.synchronize(dev)
    wall = time.time() - PROFILE["t0"]
    PROFILE["sampling"] = False
    prof.stop()
    path = os.path.join(WORK_DIR, "ltx25_step_trace.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)
    events = events.get("traceEvents", events) if isinstance(events, dict) else events
    per_dev, compact = {}, []
    for e in events:
        cat, args = e.get("cat", ""), e.get("args") or {}
        if e.get("ph") != "X" or cat not in ("kernel", "gpu_memcpy", "gpu_memset") or "device" not in args:
            continue
        d = per_dev.setdefault(int(args["device"]), {"busy": [], "groups": {}, "copies": {}, "names": {}})
        span = (e["ts"] / 1e6, (e["ts"] + e.get("dur", 0)) / 1e6)
        name = e.get("name", "")
        compact.append([int(args["device"]), args.get("stream"), cat, name, e["ts"], e.get("dur", 0),
                        args.get("grid"), args.get("block"), args.get("bytes")])
        if cat == "gpu_memcpy":
            d["copies"][name] = d["copies"].get(name, 0.0) + span[1] - span[0]
            continue
        d["busy"].append(span)
        low = name.lower()
        group = next((g for g, keys in _KERNEL_GROUPS if any(k in low for k in keys)), "other kernels")
        d["groups"][group] = d["groups"].get(group, 0.0) + span[1] - span[0]
        d["names"][name] = d["names"].get(name, 0.0) + span[1] - span[0]
    import gzip
    compact_path = os.path.join(OUTPUT_DIR, "ltx25_step_kernels.json.gz")
    with gzip.open(compact_path, "wt") as f:
        json.dump({"wall": wall, "telemetry": PROFILE["samples"], "fields": _SMI_FIELDS, "events": compact}, f)
    print(f"\n  🔬 Profile of one denoising step: {wall:.1f}s wall (kernel list: {compact_path})")
    for line in _gpu_telemetry():
        print(f"     {line}")
    for dev in sorted(per_dev):
        d = per_dev[dev]
        busy = _union(d["busy"])
        groups = " | ".join(f"{g} {t:.1f}s" for g, t in sorted(d["groups"].items(), key=lambda kv: -kv[1]))
        copies = " | ".join(f"{n.split('(')[0].strip()} {t:.1f}s" for n, t in sorted(d["copies"].items(), key=lambda kv: -kv[1]))
        print(f"     GPU {dev}: kernels busy {busy:.1f}s ({100 * busy / max(wall, 1e-6):.0f}%), idle {max(0.0, wall - busy):.1f}s"
              f"\n        {groups}\n        copies (run beside kernels): {copies or 'none'}")
        for name, t in sorted(d["names"].items(), key=lambda kv: -kv[1])[:10]:
            print(f"        {t:6.2f}s  {name[:110]}")
    sys.stdout.flush()


def report(fraction, text):
    PROGRESS.update(active=True, fraction=float(fraction), text=text)


def run_generation(prompt, image_start, image_end, width, height, frames, steps, seed, extra=None):
    TIMES.clear()
    t0 = time.time()
    state = {"t": None, "total": steps, "last": None}

    def callback(step_idx=-1, latent=None, force_refresh=True, read_state=False, override_num_inference_steps=-1,
                 pass_no=-1, progress_unit=None, progress_title=None, **kwargs):
        if torch.is_tensor(latent) and not bool(torch.isfinite(latent).all()):
            raise RuntimeError("Denoising produced NaN/Inf; refusing invalid video")
        if read_state:  # interrupt polling, once per transformer block
            return
        if progress_unit is not None:  # layer / tile progress of other phases
            return
        now = time.time()
        if PROFILE["wanted"] and step_idx == 0 and PROFILE["prof"] is None:
            _profile_begin()  # after step 1: step 2 is traced
        elif PROFILE["prof"] is not None and step_idx == 1:
            _profile_end()
        if step_idx < 0:  # denoising starts
            state["total"] = override_num_inference_steps if override_num_inference_steps and override_num_inference_steps > 0 else steps
            state["t"] = state["last"] = now
            BAR.start("denoise", "Denoising", state["total"], "steps")
            report(0.12, f"🎬 Denoising step 0/{state['total']}")
            return
        done = step_idx + 1
        dt = now - (state["last"] or now)
        state["last"] = now
        TIMES["denoise"] = now - (state["t"] or now)
        BAR.update(done, state["total"])
        report(0.12 + 0.76 * done / state["total"], f"🎬 Denoising step {done}/{state['total']} ({dt:.1f}s/step)")
        if done >= state["total"]:
            BAR.finish()
            report(0.9, "🎞️ Decoding video + audio...")

    def set_progress_status(status):
        if status and "decod" not in status.lower():
            report(PROGRESS["fraction"], f"⏳ {status}...")

    kwargs = dict(input_prompt=prompt, image_start=image_start, height=height, width=width, frame_num=frames, fps=float(FPS),
                  seed=seed, callback=callback, set_progress_status=set_progress_status,
                  VAE_tile_size=vae_tile_size(width, height), input_video_strength=1.0, denoising_strength=1.0,
                  guide_scale=1.0, audio_cfg_scale=1.0, alt_guide_scale=1.0, sampling_steps=int(steps), guide_phases=1,
                  sample_solver="distilled_8_steps", self_refiner_setting=0, n_prompt="")
    if image_end is not None:
        kwargs["image_end"] = image_end
    kwargs.update(extra or {})
    wan_model._interrupt = False
    report(0.02, "📝 Encoding the prompt...")
    try:
        with torch.no_grad():  # grad mode is per thread and ON in a new thread
            result = wan_model.generate(**kwargs)
    except BaseException:
        BAR.abort()
        raise
    BAR.abort() if wan_model._interrupt else BAR.finish()
    TIMES["total"] = time.time() - t0
    return result


def generate_video(prompt, start_img, end_img, duration, resolution, aspect, steps, seed):
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    try:
        prompt = (prompt or "").strip()
        if not prompt:
            return None, "❌ Please write a prompt."
        frames = DURATIONS.get(duration, 121)
        width, height = size_for(resolution, aspect)
        image_start = start_img.convert("RGB") if start_img is not None else None
        image_end = end_img.convert("RGB") if end_img is not None else None
        ref = image_start if image_start is not None else image_end
        if ref is not None:  # Image-to-Video: the output follows the frame's aspect ratio
            width, height = fit_to_image(ref, width, height)
            image_start = prepare_conditioning_image(image_start, width, height) if image_start is not None else None
            image_end = prepare_conditioning_image(image_end, width, height) if image_end is not None else None
        steps = set_denoising_steps(steps)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        mode = ("First + Last Frame → Video" if image_start is not None and image_end is not None
                else "Image → Video" if image_start is not None else "Last Frame → Video" if image_end is not None
                else "Text → Video")
        print(f"\n{'=' * 70}\n🎬 {mode} | {width}x{height} | {frames} frames ({frames / FPS:.1f}s) | {steps} steps | seed {seed}")
        print(f"📝 {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()
        torch.cuda.reset_peak_memory_stats(MAIN_DEV)

        result = run_generation(prompt, image_start, image_end, width, height, frames, steps, seed)
        if wan_model._interrupt:
            return None, "⏹️ Generation stopped."
        if not isinstance(result, dict) or not torch.is_tensor(result.get("x")):
            return None, "❌ The pipeline returned no video."
        report(0.97, "💾 Saving MP4...")
        t_save = time.time()
        out_path = os.path.join(OUTPUT_DIR, f"ltx25_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        BAR.start("save", "Saving MP4")
        save_mp4(result["x"], result.get("audio"), result.get("audio_sampling_rate", 24000), out_path)
        BAR.finish()
        t_text, t_denoise, t_decode = TIMES.get("text", 0.0), TIMES.get("denoise", 0.0), TIMES.get("decode", 0.0)
        per_step = t_denoise / max(1, steps)
        total = TIMES.get("total", 0.0) + time.time() - t_save
        status = "\n".join([
            f"✅ Done in {total:.0f}s ({mode})",
            f"⚡ Prompt {t_text:.0f}s | Denoising {t_denoise:.0f}s ({per_step:.1f}s/step) | VAE + audio {t_decode:.0f}s",
            f"📐 {width}x{height} | {frames} frames @ {FPS} fps ({frames / FPS:.1f}s) | {steps} steps | seed {seed}",
            f"📁 {out_path}",
        ])
        print(f"✅ Saved {out_path} | total {total:.0f}s = prompt {t_text:.0f}s + denoise {t_denoise:.0f}s "
              f"({per_step:.1f}s/step) + VAE/audio {t_decode:.0f}s + MP4 {time.time() - t_save:.0f}s | "
              f"peak VRAM GPU0 {_gb(torch.cuda.max_memory_allocated(MAIN_DEV))} | RAM {_ram_in_use()}\n")
        sys.stdout.flush()
        return out_path, status
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        return None, "❌ Out of GPU memory. Pick a lower resolution or a shorter duration and try again."
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):  # e.g. an illegal memory access: the GPU context is unusable from here on
            print("❌ The GPU is in an error state. Re-run Step 2 to restart the engine.")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine."
        return None, f"❌ Error: {e}"
    finally:
        BAR.abort()
        PROGRESS["active"] = False
        gc.collect()
        torch.cuda.empty_cache()


# ==== Lip-sync avatar: reference image + audio → video ====
# Wan2GP's audio-to-video path: the audio goes through LTX's audio VAE encoder and its latent is held fixed (strength 1)
# while the video latent is denoised, so lips, face and body move to the sound. The reference image is the first
# frame. The MP4 gets the original audio (exactly what was uploaded), not the VAE's re-synthesis of it.
AVATAR_MAX_SECONDS = 20
AUDIO_RATE = 48000


def load_audio(path, start=0.0):
    """Any audio (or video) file → float32 [samples, 2] at 48 kHz, decoded by ffmpeg, from `start` seconds."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(start)):.3f}", "-i", path, "-vn", "-ac", "2", "-ar",
           str(AUDIO_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout  # no preexec_fn: vfork, not a copy of pinned RAM
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def avatar_frames(seconds):
    """Frames for `seconds` of audio: LTX needs 8k + 1 frames; rounded up so the clip covers the whole audio."""
    return max(25, min(8 * math.ceil((seconds * FPS - 1) / 8) + 1, AVATAR_MAX_SECONDS * FPS + 1))


def generate_avatar(prompt, ref_img, audio_path, resolution, max_seconds, start_s, steps, seed, original_audio,
                    sync_strength=1.0):
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    try:
        prompt = (prompt or "").strip()
        if ref_img is None:
            return None, "❌ Please upload a reference image (the person / character that should speak or sing)."
        if not audio_path:
            return None, "❌ Please upload an audio file (speech or song)."
        if not prompt:
            return None, "❌ Please write a prompt (who is in the image and what they do while speaking / singing)."
        try:
            audio = load_audio(audio_path, start_s or 0.0)
        except subprocess.CalledProcessError as e:
            return None, f"❌ Could not read the audio file: {(e.stderr or b'').decode(errors='ignore')[-300:]}"
        available = len(audio) / AUDIO_RATE
        if available < 1.0:
            return None, f"❌ Only {available:.1f}s of audio after the start time: at least 1 s is needed."
        seconds = min(available, float(max_seconds or AVATAR_MAX_SECONDS), AVATAR_MAX_SECONDS)
        frames = avatar_frames(seconds)
        samples = int(round(frames / FPS * AUDIO_RATE))
        audio = audio[:samples]
        if len(audio) < samples:  # the last partial frame: silence (mouth closed), so the whole clip is conditioned
            audio = np.concatenate([audio, np.zeros((samples - len(audio), 2), dtype=np.float32)])

        image = ref_img.convert("RGB")
        width, height = fit_to_image(image, *size_for(resolution, "16:9 Landscape"))
        image = prepare_conditioning_image(image, width, height)
        steps = set_denoising_steps(steps)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        print(f"\n{'=' * 70}\n🎤 Lip-sync avatar | {width}x{height} | {frames} frames ({frames / FPS:.1f}s, audio "
              f"{seconds:.1f}s from {float(start_s or 0):.1f}s) | {steps} steps | seed {seed}")
        print(f"📝 {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()
        torch.cuda.reset_peak_memory_stats(MAIN_DEV)

        sync_strength = max(1.0, float(sync_strength or 1.0))
        print(f"🔊 Lip-sync strength (modality guidance) {sync_strength:g}"
              + (" | 2 transformer passes per step" if sync_strength > 1.0 else ""))
        result = run_generation(prompt, image, None, width, height, frames, steps, seed,
                                extra=dict(input_waveform=audio, input_waveform_sample_rate=AUDIO_RATE, audio_scale=1.0,
                                           alt_guide_scale=sync_strength))
        if wan_model._interrupt:
            return None, "⏹️ Generation stopped."
        if not isinstance(result, dict) or not torch.is_tensor(result.get("x")):
            return None, "❌ The pipeline returned no video."
        report(0.97, "💾 Saving MP4...")
        t_save = time.time()
        out_path = os.path.join(OUTPUT_DIR, f"ltx25_avatar_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        BAR.start("save", "Saving MP4")
        if original_audio:
            save_mp4(result["x"], audio, AUDIO_RATE, out_path)
        else:
            save_mp4(result["x"], result.get("audio"), result.get("audio_sampling_rate", 24000), out_path)
        BAR.finish()
        t_text, t_denoise, t_decode = TIMES.get("text", 0.0), TIMES.get("denoise", 0.0), TIMES.get("decode", 0.0)
        per_step = t_denoise / max(1, steps)
        total = TIMES.get("total", 0.0) + time.time() - t_save
        status = "\n".join([
            f"✅ Done in {total:.0f}s (lip-sync avatar)",
            f"⚡ Prompt {t_text:.0f}s | Denoising {t_denoise:.0f}s ({per_step:.1f}s/step) | VAE {t_decode:.0f}s",
            f"📐 {width}x{height} | {frames} frames @ {FPS} fps ({frames / FPS:.1f}s) | {steps} steps | seed {seed}",
            f"🔊 {'Your original audio' if original_audio else 'Audio decoded by LTX'} ({seconds:.1f}s from {float(start_s or 0):.1f}s)"
            f" | lip-sync strength {sync_strength:g}",
            f"📁 {out_path}",
        ])
        print(f"✅ Saved {out_path} | total {total:.0f}s = prompt {t_text:.0f}s + denoise {t_denoise:.0f}s "
              f"({per_step:.1f}s/step) + VAE/audio {t_decode:.0f}s + MP4 {time.time() - t_save:.0f}s | "
              f"peak VRAM GPU0 {_gb(torch.cuda.max_memory_allocated(MAIN_DEV))} | RAM {_ram_in_use()}\n")
        sys.stdout.flush()
        return out_path, status
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        return None, "❌ Out of GPU memory. Pick a lower resolution or a shorter maximum length and try again."
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):
            print("❌ The GPU is in an error state. Re-run Step 2 to restart the engine.")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine."
        return None, f"❌ Error: {e}"
    finally:
        BAR.abort()
        PROGRESS["active"] = False
        gc.collect()
        torch.cuda.empty_cache()


# ==== Multi-subject reference (MSR V2): up to 4 subjects + an optional background → one video ====
MSR_REFERENCE_FRAMES = {"33 frames (best identity)": 33, "25 frames (faster)": 25}
_REMBG = {"session": None, "error": None}


def _aspect_family(width, height):
    ratio = width / height
    return "landscape" if ratio >= 1.25 else ("portrait" if ratio <= 0.8 else "square")


def msr_prepare_reference(image, width, height, is_background):
    """LiconStudio's reference preparation: the background is scaled to cover the frame and center-cropped. A subject is
    never stretched: with a similar aspect ratio and a larger image it covers the frame (center crop), otherwise the whole
    image is fitted and padded with white (the MSR LoRA was trained on subjects on white)."""
    image = image.convert("RGB")
    src_w, src_h = image.size
    if is_background or (_aspect_family(src_w, src_h) == _aspect_family(width, height)
                         and not (src_w <= width and src_h <= height)):
        return prepare_conditioning_image(image, width, height)
    scale = min(width / src_w, height / src_h)
    new_w, new_h = max(1, min(width, round(src_w * scale))), max(1, min(height, round(src_h * scale)))
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.LANCZOS), ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"  # rembg's default model


def _u2net_session():
    if _REMBG["session"] is None:
        path = os.path.join(MODELS_DIR, "u2net.onnx")
        if not os.path.exists(path):
            print("  📥 Downloading the background-removal model (U2-Net, 176 MB, once)...")
            urllib.request.urlretrieve(U2NET_URL, path + ".part")
            os.replace(path + ".part", path)
        import onnxruntime as ort
        _REMBG["session"] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _REMBG["session"]


def remove_background(image):
    """Subject on white: rembg's U2-Net run with onnxruntime on the CPU, with rembg's own pre/post-processing (320 x 320,
    ImageNet normalisation, min-max mask). rembg itself is not installed: it pins NumPy / Pillow / scikit-image versions
    newer than Kaggle's. Returns the image unchanged if the model is unavailable."""
    if _REMBG["error"] is not None:
        return image
    try:
        session = _u2net_session()
        image = image.convert("RGB")
        small = np.asarray(image.resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
        small = small / max(float(small.max()), 1e-6)
        small = (small - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        pred = session.run(None, {session.get_inputs()[0].name: small.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        pred = pred[:, 0, :, :].squeeze()
        pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
        mask = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize(image.size, Image.Resampling.LANCZOS)
        return Image.composite(image, Image.new("RGB", image.size, (255, 255, 255)), mask)
    except Exception as e:
        _REMBG["error"] = e
        print(f"  ⚠️ Background removal unavailable ({e}): subjects are used as uploaded")
        return image


def generate_msr(prompt, subject1, subject2, subject3, subject4, background, remove_bg, duration, resolution, aspect,
                 reference_frames, steps, seed):
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    try:
        prompt = (prompt or "").strip()
        subjects = [img for img in (subject1, subject2, subject3, subject4) if img is not None]
        if not subjects:
            return None, "❌ Please upload at least one subject / object image (Subject 1 to 4)."
        if not prompt:
            return None, "❌ Please write a prompt: describe each subject, what they do, and the scene."
        frames = DURATIONS.get(duration, 121)
        width, height = size_for(resolution, aspect)
        if background is not None:  # the background sets the frame's aspect ratio
            width, height = fit_to_image(background.convert("RGB"), width, height)
        ref_frames = MSR_REFERENCE_FRAMES.get(reference_frames, 33)
        steps = set_denoising_steps(steps)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        report(0.01, "🧩 Preparing the reference images...")
        refs = []
        for index, image in enumerate(subjects, start=1):
            image = image.convert("RGB")
            if remove_bg:
                t0 = time.time()
                image = remove_background(image)
                print(f"  🪄 Subject {index}: background removed in {time.time() - t0:.1f}s")
            refs.append(msr_prepare_reference(image, width, height, False))
        if background is not None:
            refs = [msr_prepare_reference(background, width, height, True)] + refs  # Wan2GP order: background first
        video_prompt_type = "KI" if background is not None else "I"
        wan_model.model_def["ltx2_msr_frame_count"] = ref_frames
        latent_frames = (ref_frames - 1) // 8 + 1
        ref_tokens = len(refs) * latent_frames * (height // 32) * (width // 32)
        video_tokens = ((frames - 1) // 8 + 1) * (height // 32) * (width // 32)
        print(f"\n{'=' * 70}\n🧩 Multi-subject reference | {len(subjects)} subject(s)"
              f"{' + background' if background is not None else ''} | {width}x{height} | {frames} frames "
              f"({frames / FPS:.1f}s) | {steps} steps | seed {seed}")
        print(f"   tokens: video {video_tokens} + references {ref_tokens} ({ref_frames} frames each)")
        print(f"📝 {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()
        torch.cuda.reset_peak_memory_stats(MAIN_DEV)

        result = run_generation(prompt, None, None, width, height, frames, steps, seed,
                                extra=dict(input_ref_images=refs, video_prompt_type=video_prompt_type))
        if wan_model._interrupt:
            return None, "⏹️ Generation stopped."
        if not isinstance(result, dict) or not torch.is_tensor(result.get("x")):
            return None, "❌ The pipeline returned no video."
        report(0.97, "💾 Saving MP4...")
        t_save = time.time()
        out_path = os.path.join(OUTPUT_DIR, f"ltx25_msr_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        BAR.start("save", "Saving MP4")
        save_mp4(result["x"], result.get("audio"), result.get("audio_sampling_rate", 24000), out_path)
        BAR.finish()
        t_text, t_denoise, t_decode = TIMES.get("text", 0.0), TIMES.get("denoise", 0.0), TIMES.get("decode", 0.0)
        per_step = t_denoise / max(1, steps)
        total = TIMES.get("total", 0.0) + time.time() - t_save
        status = "\n".join([
            f"✅ Done in {total:.0f}s (multi-subject reference)",
            f"⚡ Prompt {t_text:.0f}s | Denoising {t_denoise:.0f}s ({per_step:.1f}s/step) | VAE + audio {t_decode:.0f}s",
            f"📐 {width}x{height} | {frames} frames @ {FPS} fps ({frames / FPS:.1f}s) | {steps} steps | seed {seed}",
            f"🧩 {len(subjects)} subject(s){' + background' if background is not None else ''} | {ref_frames}-frame references",
            f"📁 {out_path}",
        ])
        print(f"✅ Saved {out_path} | total {total:.0f}s = prompt {t_text:.0f}s + denoise {t_denoise:.0f}s "
              f"({per_step:.1f}s/step) + VAE/audio {t_decode:.0f}s + MP4 {time.time() - t_save:.0f}s | "
              f"peak VRAM GPU0 {_gb(torch.cuda.max_memory_allocated(MAIN_DEV))} | RAM {_ram_in_use()}\n")
        sys.stdout.flush()
        return out_path, status
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        return None, ("❌ Out of GPU memory. Fewer references, 25-frame references, a lower resolution or a shorter "
                      "duration all help.")
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):
            print("❌ The GPU is in an error state. Re-run Step 2 to restart the engine.")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine."
        return None, f"❌ Error: {e}"
    finally:
        BAR.abort()
        PROGRESS["active"] = False
        gc.collect()
        torch.cuda.empty_cache()


GENERATE = {"a2v": generate_avatar, "msr": generate_msr}.get(APP_MODE, generate_video)


# ==== Generation as a background job ====
# The Generate button only starts a job and returns: the page asks for its state every 2 s with short, independent
# requests (queue=False: plain HTTP, no event stream), so a dropped tunnel or a reloaded page never loses the video.
JOB = {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready."}


def _job_worker(job_id, args):
    try:
        torch.cuda.set_device(MAIN_DEV)  # the current CUDA device is per thread
        video, status = GENERATE(*args)
    except BaseException as e:  # generate_video reports its own errors; this is a last resort
        video, status = None, f"❌ Error: {e}"
    JOB.update(running=False, done=job_id, video=video, status=status)
    PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                    message=status.splitlines()[0] if status else "")


def start_generation(*args):
    if JOB["running"]:
        return gr.skip(), "⏳ A video is already being generated. It appears here as soon as it is ready."
    JOB.update(id=JOB["id"] + 1, running=True)
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    threading.Thread(target=_job_worker, args=(JOB["id"], args), daemon=True).start()
    return None, ("🚀 Generation started. Progress is shown at the top of the page and the video appears here when it "
                  "is ready. Reloading the page or a dropped connection does not stop the job.")


def poll_job(seen):
    """Every 2 s: live progress while a job runs, the result once (per page) when it is done."""
    if JOB["running"]:
        elapsed = time.time() - PROGRESS["started"] if PROGRESS["started"] else 0.0
        return gr.skip(), f"⏳ {PROGRESS['text'] or 'Working...'} · {_fmt_secs(elapsed)}", seen
    if JOB["done"] and JOB["done"] != seen:
        return JOB["video"], JOB["status"], JOB["done"]
    return gr.skip(), gr.skip(), seen


def stop_generation():
    if not JOB["running"]:
        return "Nothing is running."
    wan_model._interrupt = True
    return "⏹️ Stopping..."


def load_latest_video():
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return None, f"No videos in {OUTPUT_DIR} yet."
    return videos[-1], f"📁 Loaded latest video: {videos[-1]}"


# ==== Gradio UI (AIQUEST Academy branding) ====
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.btn-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.social-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 150px; padding: 10px 18px; border-radius: 10px; font-weight: 700; font-size: 13px; text-decoration: none; color: white; white-space: nowrap; }
.yt-btn  { background: #FF0000; box-shadow: 0 4px 12px rgba(255,0,0,0.3); }
.x-btn   { background: #000000; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
.sup-btn { background: linear-gradient(135deg,#f6d365,#fda085); box-shadow: 0 4px 12px rgba(253,160,133,0.35); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.status-line { text-align: center; color: #6b7280; font-size: 13px; margin: -8px 0 12px 0; }
.brand-footer { text-align: center; color: #6b7280; font-size: 12px; margin-top: 24px; }
"""

BRAND_TITLE = {"a2v": "🎤 LTX-2.5 Lip-Sync Avatar - Audio → Video",
               "msr": "🧩 LTX-2.5 Multi-Subject Reference - MSR V2"}.get(APP_MODE, "🎬 LTX-2.5 22B Distilled - Video + Audio")
BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">{BRAND_TITLE}</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
""".replace("{BRAND_TITLE}", BRAND_TITLE)
STATUS_HTML = (f'<div class="status-line">⚡ {INT8_LABEL} &nbsp;|&nbsp; '
               + ("Transformer split over both T4s (tensor parallel)" if TENSOR_PARALLEL
                  else "Transformer on GPU 0, Gemma-4 text encoder on GPU 1" if TE_DEV != MAIN_DEV else "Single GPU")
               + {"a2v": ' &nbsp;|&nbsp; lips, face and body driven by your audio</div>',
                  "msr": ' &nbsp;|&nbsp; LiconStudio MSR V2 LoRA: your subjects, one scene, with sound</div>'}.get(
                   APP_MODE, ' &nbsp;|&nbsp; 24 fps video + synced audio</div>'))
FOOTER_HTML = '<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>'

PROGRESS_HEAD = """
<script>
(function () {
  // Progress + result overlay fed by plain HTTP polling: it keeps working when a tunnel drops Gradio's event stream.
  var box = null, fill = null, label = null, hideAt = 0;
  function build() {
    box = document.createElement("div");
    box.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:99999;width:min(560px,92vw);" +
      "background:#1f2937;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:10px 14px;display:none;" +
      "font-family:Inter,sans-serif;color:#fff;font-size:13px";
    label = document.createElement("div");
    label.style.cssText = "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    var track = document.createElement("div");
    track.style.cssText = "height:8px;background:rgba(255,255,255,.18);border-radius:6px;overflow:hidden";
    fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);transition:width .6s ease";
    track.appendChild(fill); box.appendChild(label); box.appendChild(track); document.body.appendChild(box);
  }
  async function tick() {
    try {
      var r = await fetch("aiq_progress?t=" + Date.now(), {cache: "no-store"});
      var p = await r.json();
      if (!box && document.body) build();
      if (box) {
        if (p.active) {
          box.style.display = "block"; hideAt = 0;
          fill.style.width = Math.round(p.fraction * 100) + "%";
          label.textContent = "⏳ " + p.text + " · " + Math.round(p.elapsed) + "s";
        } else if (box.style.display === "block") {
          if (!hideAt) {
            fill.style.width = "100%";
            label.textContent = (p.video ? "✅ Done in " : "⚠️ Finished after ") + Math.round(p.elapsed) + "s ";
            if (p.video) {
              var link = document.createElement("a");
              link.href = "aiq_video?t=" + Date.now(); link.target = "_blank";
              link.style.cssText = "color:#c7d2fe;font-weight:600";
              link.textContent = "· ▶ open / download the video";
              label.appendChild(link);
            } else if (p.message) { label.textContent += "· " + p.message; }
            hideAt = Date.now() + 60000;
          } else if (Date.now() > hideAt) { box.style.display = "none"; }
        }
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
})();
</script>
"""

EXAMPLES = [
    ["A cinematic close-up of a snow leopard resting on a rocky ledge in the Himalayas during light snowfall. It slowly "
     "turns its head toward the camera and blinks, breath fogging in the cold air. Soft wind howls across the peaks and "
     "snowflakes tick against the rocks."],
    ["A street musician plays a violin on a rainy neon-lit Tokyo street at night. The camera slowly circles him as puddles "
     "reflect pink and blue signs. The violin melody echoes between the buildings over the hiss of rain and distant traffic."],
    ["A young woman in a cozy cafe looks up from her laptop, smiles warmly at the camera and says: \"You made it! I saved "
     "you the best seat.\" Quiet chatter, an espresso machine hissing and cups clinking in the background."],
]

AVATAR_EXAMPLES = [
    ["A young woman sits in a warm, softly lit home studio and sings straight to the camera, her lips, jaw and cheeks "
     "moving precisely with every word, eyebrows lifting on the long notes, gentle head movement. Static medium close-up, "
     "shallow depth of field."],
    ["A man in a dark suit speaks confidently to the camera in a modern office, clear lip movements matching his words, "
     "natural blinking and small hand gestures. Steady medium shot, soft daylight from a window."],
    ["An animated cartoon character talks excitedly to the viewer, exaggerated mouth shapes in sync with the speech, "
     "expressive eyes and eyebrows, bouncing slightly. Bright, colorful background."],
]


MSR_EXAMPLES = [
    ["The man from Subject 1 and the woman from Subject 2 walk side by side along the street in the background, talking "
     "and laughing. He gestures with his hands; she glances at him and smiles. Handheld tracking shot, warm evening "
     "light, city ambience and their quiet conversation."],
    ["The woman from Subject 1 sits at the cafe table from the background and picks up the red mug from Subject 2, "
     "takes a sip and looks out of the window. Static medium shot, soft daylight, cafe chatter and clinking cups."],
    ["The knight from Subject 1 and the dragon from Subject 2 face each other in the canyon from the background. The "
     "dragon lowers its head and the knight slowly raises his shield. Slow push-in, dramatic light, wind and a low growl."],
]


def _msr_ui():
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)
    gr.Markdown("💡 **How to use:** upload 1 to 4 **subjects or objects** (people, characters, products, pets) and, "
                "optionally, a **background** (the scene). In the prompt, refer to them as *Subject 1*, *Subject 2*, "
                "... and describe what they do, where, the camera and the sound. Character sheets (one character from "
                "several angles) keep identities best. With a background, the video follows its aspect ratio.")
    with gr.Row():
        subject1 = gr.Image(type="pil", label="👤 Subject 1", height=200)
        subject2 = gr.Image(type="pil", label="👤 Subject 2 (optional)", height=200)
        subject3 = gr.Image(type="pil", label="👤 Subject 3 (optional)", height=200)
        subject4 = gr.Image(type="pil", label="👤 Subject 4 (optional)", height=200)
    with gr.Row():
        with gr.Column(scale=3):
            background = gr.Image(type="pil", label="🏞️ Background / scene (optional)", height=220)
            prompt = gr.Textbox(label="🎬 Prompt", lines=4,
                                placeholder="The man from Subject 1 and the woman from Subject 2 walk through the "
                                            "street from the background, talking ... (camera, light, sound)")
        with gr.Column(scale=2):
            remove_bg = gr.Checkbox(value=True, label="🪄 Remove the background behind subjects (puts them on white)",
                                    info="Turn off if your subjects are already on a plain white background")
            resolution = gr.Dropdown([r for r in RESOLUTIONS if not r.startswith("1088")], value="480p (recommended)",
                                     label="📐 Resolution")
            aspect = gr.Dropdown(list(ASPECTS), value="16:9 Landscape", label="📏 Aspect ratio",
                                 info="Used when there is no background image")
            duration = gr.Dropdown(list(DURATIONS), value="5 s (121 frames)", label="⏱️ Duration")
            reference_frames = gr.Radio(list(MSR_REFERENCE_FRAMES), value="33 frames (best identity)",
                                        label="🧩 Reference strength",
                                        info="Every reference adds tokens: 25 frames is faster and lighter")
            steps = gr.Slider(4, 8, value=8, step=1, label="⚡ Denoising steps",
                              info="8 = full quality; fewer steps are faster but softer")
            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")

    with gr.Row():
        gen_btn = gr.Button("🧩 Generate Video", variant="primary", size="lg")
        stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
        clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
    video_out = gr.Video(label="🎥 Generated video (with audio)")
    latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
    status_out = gr.Textbox(label="ℹ️ Status", lines=5, interactive=False)
    seen_job = gr.State(0)
    poll_timer = gr.Timer(2.0)

    gr.Examples(MSR_EXAMPLES, inputs=[prompt], label="✨ Example prompts")
    gr.HTML(FOOTER_HTML)

    gen_btn.click(start_generation, inputs=[prompt, subject1, subject2, subject3, subject4, background, remove_bg, duration,
                                            resolution, aspect, reference_frames, steps, seed],
                  outputs=[video_out, status_out], queue=False)
    poll_timer.tick(poll_job, inputs=seen_job, outputs=[video_out, status_out, seen_job], queue=False,
                    show_progress="hidden")
    stop_btn.click(stop_generation, outputs=status_out, queue=False)
    latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
    clear_btn.click(lambda: (None, "Ready.", "", None, None, None, None, None, -1),
                    outputs=[video_out, status_out, prompt, subject1, subject2, subject3, subject4, background, seed],
                    queue=False)


def _avatar_ui():
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)
    gr.Markdown("💡 **How to use:** upload a clear, front-facing image of the person / character and the speech or song, "
                "then describe who it is and how they deliver it (singing, talking, emotion, camera). The video length "
                "follows the audio (up to the maximum below). 480p is the sweet spot on T4; long clips at 704p are slow.")
    with gr.Row():
        with gr.Column(scale=3):
            ref_img = gr.Image(type="pil", label="🖼️ Reference image (first frame)", height=260)
            audio_in = gr.Audio(type="filepath", label="🎵 Audio: speech or song (any format)")
            prompt = gr.Textbox(label="🎬 Prompt", lines=4,
                                placeholder="A woman sings to the camera, lips moving in sync with the lyrics ...")
        with gr.Column(scale=2):
            resolution = gr.Dropdown([r for r in RESOLUTIONS if not r.startswith("1088")], value="480p (recommended)",
                                     label="📐 Resolution", info="The video follows the image's aspect ratio")
            max_seconds = gr.Slider(2, AVATAR_MAX_SECONDS, value=10, step=1, label="⏱️ Maximum length (s)",
                                    info="Shorter audio sets the length itself")
            start_s = gr.Number(value=0, label="⏩ Start at (seconds into the audio)")
            steps = gr.Slider(4, 8, value=8, step=1, label="⚡ Denoising steps",
                              info="8 = full quality; fewer steps are faster but softer")
            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")
            sync_strength = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="👄 Extra lip-sync guidance",
                                      info="1 = off (normal speed). Above 1 adds LTX's modality guidance: a second "
                                           "pass per step (2x time) that strengthens the audio's pull on the face")
            original_audio = gr.Checkbox(value=True, label="Use my original audio in the MP4 (recommended)")

    with gr.Row():
        gen_btn = gr.Button("🎤 Generate Lip-Sync Video", variant="primary", size="lg")
        stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
        clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
    video_out = gr.Video(label="🎥 Generated lip-sync video")
    latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
    status_out = gr.Textbox(label="ℹ️ Status", lines=5, interactive=False)
    seen_job = gr.State(0)
    poll_timer = gr.Timer(2.0)

    gr.Examples(AVATAR_EXAMPLES, inputs=[prompt], label="✨ Example prompts")
    gr.HTML(FOOTER_HTML)

    gen_btn.click(start_generation, inputs=[prompt, ref_img, audio_in, resolution, max_seconds, start_s, steps, seed,
                                            original_audio, sync_strength],
                  outputs=[video_out, status_out], queue=False)
    poll_timer.tick(poll_job, inputs=seen_job, outputs=[video_out, status_out, seen_job], queue=False,
                    show_progress="hidden")
    stop_btn.click(stop_generation, outputs=status_out, queue=False)
    latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
    clear_btn.click(lambda: (None, "Ready.", "", None, None, -1),
                    outputs=[video_out, status_out, prompt, ref_img, audio_in, seed], queue=False)


def _t2v_ui():
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)
    gr.Markdown("💡 **Tips:** describe the subject, the action, the camera, the light, and the sounds or speech you want "
                "(the model makes the audio too). 480p + 5 s is the sweet spot on T4; 384p is the fastest preview. "
                "Time per step grows with resolution x duration.")
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="🎬 Prompt", lines=5,
                                placeholder="A cinematic shot of ... (subject, action, camera, light, sound)")
            with gr.Accordion("🖼️ Image → Video (start / end frame, optional)", open=False):
                with gr.Row():
                    start_img = gr.Image(type="pil", label="Start frame", height=200)
                    end_img = gr.Image(type="pil", label="End frame", height=200)
                gr.Markdown("*Start only = animate the image. Start + end = first/last-frame interpolation. With a frame, "
                            "the video follows the frame's aspect ratio (the Aspect ratio box is ignored).*")
        with gr.Column(scale=2):
            resolution = gr.Dropdown(list(RESOLUTIONS), value="480p (recommended)", label="📐 Resolution")
            aspect = gr.Dropdown(list(ASPECTS), value="16:9 Landscape", label="📏 Aspect ratio")
            duration = gr.Dropdown(list(DURATIONS), value="5 s (121 frames)", label="⏱️ Duration")
            steps = gr.Slider(4, 8, value=8, step=1, interactive=False, label="⚡ Denoising steps",
                              info="Official eight-step distilled schedule; no reduced-step experiments")
            gr.Markdown("Preset labels are rounded to the model grid. Default landscape: 832×512. Image inputs set their own rounded aspect ratio; exact dimensions appear in the job log.")
            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")

    with gr.Row():
        gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
        clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
    video_out = gr.Video(label="🎥 Generated video (with audio)")
    latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
    status_out = gr.Textbox(label="ℹ️ Status", lines=5, interactive=False)
    seen_job = gr.State(0)
    poll_timer = gr.Timer(2.0)

    gr.Examples(EXAMPLES, inputs=[prompt], label="✨ Example prompts")
    gr.HTML(FOOTER_HTML)

    gen_btn.click(start_generation, inputs=[prompt, start_img, end_img, duration, resolution, aspect, steps, seed],
                  outputs=[video_out, status_out], queue=False)
    poll_timer.tick(poll_job, inputs=seen_job, outputs=[video_out, status_out, seen_job], queue=False,
                    show_progress="hidden")
    stop_btn.click(stop_generation, outputs=status_out, queue=False)
    latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
    clear_btn.click(lambda: (None, "Ready.", "", None, None, -1),
                    outputs=[video_out, status_out, prompt, start_img, end_img, seed], queue=False)




print("🖥️ Building the UI...")
sys.stdout.flush()
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, head=PROGRESS_HEAD, title=f"{APP_NAME} | AIQUEST Academy") as demo:
    {"a2v": _avatar_ui, "msr": _msr_ui}.get(APP_MODE, _t2v_ui)()


# ==== Cloudflare quick tunnel: a second public link in case the Gradio share link fails ====
# Started without preexec_fn: that would force a real fork() of this process, which has to copy every pinned page.
# The link only works once cloudflared has registered with Cloudflare's edge and the random *.trycloudflare.com name
# is in DNS; a browser that asks too early caches the "not found" answer for minutes. cloudflared tries QUIC (UDP)
# first, which Kaggle often blocks: hence --protocol http2.
def _dns_published(host):
    """Cloudflare's DNS-over-HTTPS: True / False, or None when the lookup itself is unavailable."""
    try:
        import json
        req = urllib.request.Request(f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
                                     headers={"accept": "application/dns-json"})
        return bool(json.loads(urllib.request.urlopen(req, timeout=5).read()).get("Answer"))
    except Exception:
        return None


def start_cloudflare_tunnel(port):
    import stat
    found = {"url": None, "registered": False, "ready": False}
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate",
                                   "--protocol", "http2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        import atexit
        atexit.register(tunnel.kill)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return found

    def _watch():
        for line in tunnel.stdout:  # keep draining so cloudflared never blocks on a full pipe
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)  # skip cloudflared's API host
            if match and not found["url"]:
                found["url"] = match.group(0)
            if "Registered tunnel connection" in line:
                found["registered"] = True

    def _wait_ready():
        deadline, registered_at = time.time() + 300, None
        while time.time() < deadline:
            if found["registered"] and registered_at is None:
                registered_at = time.time()
            published = _dns_published(found["url"].split("//", 1)[1]) if found["url"] and registered_at else False
            if published or (published is None and time.time() - registered_at > 15):
                found["ready"] = True
                print(f"\n✅ Cloudflare link is ready: {found['url']}\n")
                sys.stdout.flush()
                return
            time.sleep(3)
        print("⚠️ The Cloudflare link did not come up within 5 minutes; use the Gradio share link.")

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_wait_ready, daemon=True).start()
    return found


print("🌐 Starting the public links (Cloudflare + Gradio share)...")
sys.stdout.flush()
cf = start_cloudflare_tunnel(SERVER_PORT)
demo.queue(max_size=20)
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, show_error=True,
            prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8, allowed_paths=[OUTPUT_DIR])
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute


def _progress_route():
    end = PROGRESS["finished"] or time.time()
    elapsed = end - PROGRESS["started"] if PROGRESS["started"] else 0.0
    return JSONResponse({**PROGRESS, "elapsed": elapsed}, headers={"Cache-Control": "no-store"})


def _video_route():
    """The newest video as a plain file (plays in the browser, with a download option): no Gradio involved."""
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return JSONResponse({"error": "no video yet"}, status_code=404)
    name = os.path.basename(videos[-1])
    return FileResponse(videos[-1], media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                                      "Cache-Control": "no-store"})


for _path, _route in (("/aiq_progress", _progress_route), ("/aiq_video", _video_route)):
    demo.app.router.routes.insert(0, APIRoute(_path, _route, methods=["GET"]))  # ahead of Gradio's routes
for _ in range(60):  # give the tunnel up to ~30 s to come up before printing the links
    if cf["ready"]:
        break
    time.sleep(0.5)

print("\n" + "=" * 70)
print(f"🎬 {APP_NAME} is LIVE!")
print("=" * 70)
if cf["ready"]:
    print(f"🌐 Cloudflare tunnel: {cf['url']}")
else:
    print(f"🌐 Cloudflare tunnel: ⏳ warming up{' (' + cf['url'] + ')' if cf['url'] else ''} - wait for the "
          f"'✅ Cloudflare link is ready' line before opening it; opening it too early can keep it unreachable for minutes")
print(f"🔗 Gradio share:      {getattr(demo, 'share_url', None) or '(unavailable, use the Cloudflare link)'}")
print(f"💾 Videos are saved to {OUTPUT_DIR} (Kaggle Output panel)")
print(f"🎞️ Latest video, without the UI: <link>/aiq_video (e.g. {cf['url'] or 'https://...trycloudflare.com'}/aiq_video)")
print("=" * 70 + "\n")
sys.stdout.flush()
while True:  # keep the server process alive; generation logs stream to the notebook cell
    time.sleep(3600)
