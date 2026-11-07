# Movie Upscaler v1

A separate Kaggle upscaling studio for existing H3 clips and other SDR/CFR video.
It does not change the four H3 generation notebooks or Image Studio.

- [Movie Upscaler notebook](../../Movie_Upscaler_Kaggle_v1.ipynb)
- [CPU-only model packs](../../Movie_Upscaler_Model_Packs.ipynb)
- [Full operating guide](intro.md)

## Start

Run the model-pack notebook with **Accelerator: None** and Internet enabled.
**Save Version with output files**. Attach that saved version in the upscaler via
**Add Input → Notebook Outputs**. This uses saved notebook outputs, not the Dataset API;
no Kaggle API token is needed. The pack is approximately 3.94 GiB.

Import a fresh upscaler notebook, enable **T4 GPU** and Internet, and run its one
startup cell. The temporary URL opens the connected SeedVR2 graph. Attach your
source movie and enter its full Kaggle path in `source_path`. Press **Run once**;
frontend Auto Queue must be OFF. The commit node schedules the following chunk.
To pause, Interrupt and Clear Queue; Run again with the same settings to resume.

Select **Movie Studio - RealESRGAN** from Workflows for faster, frame-independent
2× enlargement. For 4× change the source scale to 4 AND the downsample factor to 1.
The size guard rejects inconsistent target dimensions before saving a checkpoint.

The leftmost source node, real native model/upscale nodes, and checkpoint/export
node remain visible and connected. Custom nodes handle only media IO, provenance,
checkpoints and queue continuation; they do not conceal the restoration model.
No inference starts during notebook setup. No notebook preview or download prompt
is emitted. Outputs are saved under `/kaggle/working/movie-studio/movie-jobs/`.

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

GPU 0 is used. A second T4 is not pooled and is not used in this release. This
keeps one memory path to validate first; it is not a measured claim that one GPU
is faster. T4 hardware does not support BF16/FP8 arithmetic like newer GPUs;
the upstream compatibility path, Q8 weights and FP16 fallback still need real
Kaggle inference validation. VAE tiling does not spatially tile DiT attention.
The default longest-edge guard is 1920; 4K requires deliberately raising it and
is not certified to fit a T4.

The source must be square-pixel SDR with one video stream and a constant frame
rate. Input frame timestamps are scanned incrementally before inference. Unsupported
HDR, VFR and display rotation fail early. Video is decoded to RGB8 and exported
as H.264 YUV420 at CRF 18; high-bit-depth SDR is reduced to 8-bit. Known source
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

- **10 media tests:** exact frame content across overlapping chunk boundaries,
  pause/reopen, corruption preservation, geometry/disk preflight, fractional frame
  rates, silent files, VFR rejection, nonzero source timestamps, audio timing,
  byte-identical copied audio packets, multiple audio tracks and MKV subtitles.
- **5 model-pack tests:** HTTP Range resume, server restart fallback, corrupt-file
  preservation, copying actual attached files into a completed pack, failed-pack
  handling. These use small synthetic model files and mocked HTTP responses.
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
