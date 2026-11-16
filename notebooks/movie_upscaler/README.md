# Movie Upscaler v2

A separate Kaggle upscaling studio for existing H3 clips and other SDR/CFR video.
It does not change the four H3 generation notebooks or Image Studio.

- [Movie Upscaler notebook](../../Movie_Upscaler_Kaggle_v1.ipynb)
- [CPU-only model packs](../../Movie_Upscaler_Model_Packs.ipynb)
- [Full operating guide](intro.md)

## Start

Run the model-pack notebook with **Accelerator: None**, Internet enabled and
private Kaggle API authentication (`KAGGLE_API_TOKEN` in Kaggle Secrets).
`AUTO_CREATE_DATASET=True` is now the default: after verification it automatically
creates/uploads a **private named Kaggle Dataset** and prints its link. Set
`DATASET_OWNER` to your Kaggle username if authentication cannot detect it.
Wait for processing, then attach it through **Add Input → Your Datasets**.
No manual Save Version is required to submit the Dataset. The model payload is
approximately 3.94 GiB. Repeated runs reuse a recorded submission; existing
Datasets are never overwritten or made public. Changed model catalogs use a new
content-derived Dataset name. This matches the automatic private saver in
`notebooks/image_studio/asset_builder.py`.

`AUTO_CREATE_DATASET=False` retains the optional Save Version with output files →
attach Notebook Outputs approach. Dataset uploads use flat hardlinks to the
verified nested model files, avoiding another multi-gigabyte working-storage copy.
Credentials never enter the pack metadata, upload receipts or generated notebook.
The upload-staging hardlinks remain locally for retries; saving notebook outputs
as well may store those model paths twice. Live Dataset upload is unverified here.

Import [Movie_Upscaler_Kaggle_v2.ipynb](../../Movie_Upscaler_Kaggle_v2.ipynb), enable T4 GPUs and Internet, then run its one startup cell. Click **Upload clip** in the leftmost node, wait for **Ready**, and press **Run once**. No separate Kaggle Dataset for the clip is required. The same node accepts an existing attached-video path. Uploads use 8 MiB requests so large files do not require one oversized tunnel request.

Both connected workflows contain upload → source/chunk controls → native model nodes → save. SeedVR2 defaults to temporal 2× restoration; Real-ESRGAN uses 4× inference followed by automatically connected Lanczos resizing to 2× or 4×. Saved outputs default to **FFV1 lossless RGB MKV**. Select `high_quality_mp4` for H.264 CRF 14 / slow. The format, CRF and encoder preset are visible in the save node. Export changes reuse the restored chunks, including after completion.

Keep frontend Auto Queue OFF and queue count 1. The commit node queues one next chunk. Interrupt then Clear Queue pauses. Run again to resume. Uploads live under `/kaggle/working/movie-studio/input/movie-clips/`; outputs under `/kaggle/working/movie-studio/movie-jobs/`. Save output files before ending Kaggle. Existing model inputs and the model-pack builder are unchanged. v1 remains an identical renderer alias; use v2 to avoid importing a stale copy.

Read [the complete settings rationale](QUALITY_SETTINGS.md), including exactly which defaults changed and the limits of the local validation.

## Implementation and limits

**SeedVR2:** upstream 3B Q8 GGUF, FP16 VAE, block swapping to CPU, SDPA, five-frame
batches and internal temporal overlap. Seventeen core frames plus up to four
context frames on either side are loaded per execution. Context is cropped before
commit, retaining exactly the original number of frames. Outer chunk boundaries
are not crossfaded; context and upstream temporal blending reduce seam risk but
cannot guarantee invisible seams or prevent all scene-cut artifacts.

**Real-ESRGAN:** the native ComfyUI model loader and tiled upscale node, followed
by optional 4×→2× downsampling. It has no temporal model and can flicker. No automatic
face enhancement, creative SDXL redraw or frame interpolation is applied.

DiT and Real-ESRGAN use GPU 0. SeedVR2 VAE uses GPU 1 when available, falling back to GPU 0 on single-GPU sessions. This is phase placement, not pooled memory or simultaneous tensor-parallel inference. T4 hardware does not support BF16/FP8 arithmetic like newer GPUs;
the upstream compatibility path, Q8 weights and FP16 fallback still need real
Kaggle inference validation. VAE tiling does not spatially tile DiT attention.
The default longest-edge guard is 1920; 4K requires deliberately raising it and
is not certified to fit a T4.

The source must be square-pixel SDR with one video stream and a constant frame
rate. Input frame timestamps are scanned incrementally before inference. Unsupported
HDR, VFR and display rotation fail early. Video is decoded to RGB8. Default FFV1/BGR0 MKV preserves those processed pixels exactly; optional H.264/YUV420 MP4 uses visible CRF/preset controls (14/slow by default). High-bit-depth SDR is reduced to 8-bit. Known source
colour matrix/transfer/primary tags are retained with explicit output conversion.

Lossless FFV1 checkpoints, atomically written JSON receipts and SHA-256 protect
resume. Source content and submitted graph settings identify a job; queue ticks,
source path spelling, auto-continuation and export-container choices do not change
its restoration identity. All other settings do. A verified file may be reused
from a different source path. Same-session file hashes are cached against device,
inode, size, mtime and ctime; restarting verifies them again.

Chunk loading seeks with preroll and selects by original frame timestamps; frame
counts must match before any checkpoint is committed. Export verifies geometry,
frame count and audio-stream count. Audio is stream-copied; MP4 retries with AAC
192 kbps if stream-copy fails. MKV also copies subtitle streams. Existing audio
starts are offset relative to the source video start. Metadata and chapters are
copied; arbitrary attachments/data tracks and full HDR semantics are not retained.

The job retains its lossless chunks and final movie. Measured disk preflight uses
an uncompressed RGB estimate and reserve; Kaggle film-length disk/time limits still
apply. No automatic deletion. Across sessions, restore the saved `movie-jobs`
directory to the configured working output path before resuming. Model packs alone
do not save movies/checkpoints. Software and tunnel startup still need Internet.

## Frozen sources

- ComfyUI: `1332fabfcc507aa7c548cd5152071ed494c207e8` (0.44.0).
- [SeedVR2 extension](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler): `4490bd1f482e026674543386bb2a4d176da245b9` (2.5.24).
- [Upstream SeedVR](https://github.com/ByteDance-Seed/SeedVR).
- [Q8 weights](https://huggingface.co/AInVFX/SeedVR2_comfyUI): `ac66d6d98fa49975d893b58c55bff7677191c862`.
- [VAE weights](https://huggingface.co/numz/SeedVR2_comfyUI): `09ced71023636e9bc8cdf9cdecfb2625d1e691e8`.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): official v0.1.0 x4plus release checkpoint.
- Cloudflared: 2026.10.7, size and SHA-256 pinned.

All model sizes and SHA-256 values are embedded in `assets.py` and both notebooks.
The VAE and Real-ESRGAN weights can be reused from an attached Image Studio pack;
its native INT8 SeedVR2 weights are different from this extension's GGUF weights.
The startup constrains Kaggle's Torch/TorchVision/TorchAudio/NumPy/SciPy instead
of replacing them, and saves installed package versions and source/model provenance.
Base-image compatibility can still change. Source licenses remain applicable:
SeedVR2 extension uses Apache-2.0 and Real-ESRGAN code uses BSD-3-Clause; consult
linked model cards for weight terms. No paid creator workflow was copied.

## Validation

Completed locally:

- **11 media tests:** exact frame content across overlapping chunk boundaries,
  pause/reopen, corruption preservation, geometry/disk preflight, fractional frame
  rates, silent files, VFR rejection, nonzero source timestamps, audio timing,
  byte-identical copied audio packets, multiple audio tracks and MKV subtitles.
- **4 real HTTP upload tests:** exact video bytes across 8 MiB chunks, the browser
  body's small read-limit case, invalid/oversized/incomplete uploads, disk checks,
  safe cancellation and filename handling. No GPU/model execution occurs on upload.
- **v2 browser tests:** actual mouse clicks on the visible Upload clip button,
  real file chooser selection in both graphs, server byte integrity, connected
  source/resize paths, lossless defaults and every exported SeedVR2 setting.
- **v2 actual Real-ESRGAN CPU integration:** upload → 2× inference → pause/resume
  → MP4 with audio; completed-job lossless re-export without repeating inference;
  automatic continuation across three chunks; a single scale edit produces exact
  4× dimensions and a lossless movie. FFV1 output RGB bytes match saved chunks.
- Both exported v2 API graphs pass the actual pinned ComfyUI validator. The CPU
  SeedVR2 schema checkout exposes CUDA dropdown labels only; it never executes
  SeedVR2 inference, and this schema-only modification is not shipped.
- **8 model-pack/Dataset tests:** HTTP Range resume, server restart fallback, corrupt-file
  preservation, copying actual attached files into a completed pack, failed-pack
  handling; private Dataset creation, flat upload contents, repeat-submission reuse, and failed/corrupt upload rejection. These use small synthetic model files and mocked HTTP/API responses.
- Both notebooks compile and pass notebook-v4 schema validation; embedded source
  and graph payloads match their checked-in sources.
- Both real connected graphs loaded in Chromium; the exported API confirms actual
  widget values/links, temporal batch/overlap, tiled VAE and 2× downsampling.
- Actual official Real-ESRGAN CPU inference through the real graph on a tiny video;
  24×16 → 48×32, all three frames and audio retained. Verified pause/resume without
  recomputing the first checkpoint, then automatic continuation through three
  one-frame chunks. This verifies execution and media handling, not visual quality.

The upstream SeedVR2 schema assumes a GPU exists. For CPU-only graph inspection,
the development checkout's empty device list was locally changed to expose `cpu`;
that test-only adjustment is **not shipped**. No SeedVR2 model inference was run.

Not validated: Kaggle GPU startup, real T4 memory use/speed, SeedVR2 visual output,
full-size or film-length jobs, full 3.94 GiB pack build/attachment, live public
Kaggle tunnel access, comparative restoration quality. No GPU benchmark numbers
or claim of best visual quality are inferred from these checks.

Rebuild and tests from repository root:

```sh
python notebooks/movie_upscaler/build.py
python notebooks/movie_upscaler/test_media.py
python notebooks/movie_upscaler/test_assets.py
python notebooks/movie_upscaler/check_notebooks.py
COMFY_URL=http://127.0.0.1:8188 python notebooks/movie_upscaler/browser_check.py
COMFY_URL=http://127.0.0.1:8188 python notebooks/movie_upscaler/integration_check.py
```

Browser/inference checks require a matching running ComfyUI, the custom IO nodes,
SeedVR2 schemas, Playwright/Chromium and verified Real-ESRGAN weights. They are
explicit developer checks and are never run during the user's notebook startup.
