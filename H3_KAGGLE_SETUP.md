# MiniMax H3 on Kaggle: storage and rendering

The uploaded Kaggle run passed FP16 computation and native ConvRot INT8 kernels on **both T4s**. It stopped because `/kaggle/working` had 11.57 GiB free while the selected weights total 41.38 GiB. No pretrained render completed in that run.

## Use scratch storage first

Import `minimax_h3_kaggle_dual_t4.ipynb`, enable Internet, select **T4 ×2**, and run in order. Leave `MODEL_STORAGE='auto'`.

Before package installation, the notebook prints measured free space for `/kaggle/temp`, `/tmp`, and working storage. It prefers scratch when enough space is available. A separate 50 GiB scratch volume would fit the 41.38 GiB payload and its 1 GiB margin; the notebook also reserves 5 GiB on working storage for output, plus a 6 GiB allowance before package installation unless the pinned stack is already present. Available capacity depends on the actual session. Directories on the same filesystem share capacity; their free space is never added together.

Weights and their download staging directories stay on the selected volume. ComfyUI receives symlinks instead of extra copies. Set `MODEL_STORAGE` to an explicit writable directory to override automatic selection. Scratch files disappear when the session ends and must be downloaded again next session. Saved notebook output should contain videos and reports, not the scratch weights.

## Reuse weights through attached inputs

If scratch is insufficient, or to avoid repeated weight downloads, run these three preparation notebooks separately with **Accelerator: None** and **Internet ON**:

| Notebook | Files | Payload |
| --- | --- | --- |
| [Diffusion](notebooks/minimax_h3_prepare_diffusion.ipynb) | Ref2VA INT8 ConvRot checkpoint | 19.53 GiB |
| [Encoder](notebooks/minimax_h3_prepare_encoder.ipynb) | Qwen3VL NVFP4 AWQ encoder | 14.61 GiB |
| [Support](notebooks/minimax_h3_prepare_support.ipynb) | Matching four-step LoRA, video and audio VAEs | 7.24 GiB |

Each preparation notebook streams downloads without installing packages, resumes partial files, and checks pinned sizes and SHA-256 digests. Each group is below 20 GiB; the preparation notebook also checks actual free space, including room already occupied by partial downloads. The diffusion group needs almost 20 GiB, so a preparation session with less free space cannot hold it.

1. Import each preparation notebook into Kaggle with a distinct title, run it, and **Save Version** including its output files.
2. In the rendering notebook, use **Add Input → Your Work** and attach all three saved notebook outputs. The exact panel labels may vary. Alternatively attach datasets containing those same verified files.
3. Leave `MODEL_DATASET=''`. The renderer searches nested `/kaggle/input` folders automatically and links matching files without copying them into working storage.
4. Run the rendering notebook in order. It verifies hashes before starting ComfyUI. If an earlier setup cell fails, later cells refuse to submit a render.

These instructions prepare reusable model storage. This GitHub editing environment cannot attach inputs to your Kaggle session. Keep prepared outputs private unless you have checked the redistribution terms for every model.

For FL2VA, set `MODEL_FAMILY='fl2va'` consistently in the renderer and preparation notebooks. Semantic Bridge remains optional; its author excludes Ref2VA. The preparation notebooks cover the five core assets; optional bridge files are small and download separately.

## Validation

The uploaded run establishes that CUDA 13, FP16 computation, host-staged GPU transfer, and Kitchen's native INT8 ConvRot kernel worked on its two T4s. CPU regression checks cover storage selection, nested input discovery, missing-model guards and resumable downloads. ComfyUI graph schema validation covers the supported graph variants. Full pretrained rendering, peak memory and timing still require a successful Kaggle render.

## Recover the quantization metadata error without downloading again

The pinned Comfy-Org INT8 checkpoint stores quantization configuration in per-layer `.comfy_quant` tensors. The original two-GPU loader expected the legacy `_quantization_metadata` global field. The notebook now installs a reader supporting both layouts and checks the actual checkpoint metadata before GPU model loading. Model bytes and hashes remain unchanged.

If the previous notebook failed in `h3_tensor_parallel.py` while reading that metadata, keep the same Kaggle session. Paste [the recovery cell](patches/resume_h3_after_metadata_error.py) into a new cell and run it. It stops only the notebook's own ComfyUI process and applies the compatibility patch. Then rerun steps **5, 6 and 7**. The already verified weights remain in scratch storage; step 4 can be skipped in this recovery path.

The uploaded run's step 4 took 10 minutes 44 seconds for downloading and verifying 41.38 GiB. GPUs do not accelerate network transfer or the notebook's CPU SHA-256 checks. The updated notebook hashes each newly downloaded file once instead of twice, prints download durations, and preserves complete failure details in `failed-history.json`. These changes do not establish a new download-speed or full-render benchmark.

## Decoder failure audit and safeguards

The later uploaded run reached `VAEDecodeH3DualGPU` with sampled video/audio latents and failed after about 4 minutes 13 seconds. Its older runner retained only the last 14,000 characters of the error, dominated by tensor contents. The exception type, message and decoder traceback are absent. An out-of-memory diagnosis cannot be established from that upload. Run [the read-only diagnostic cell](patches/diagnose_h3_failure.py) in the existing session before restarting it; it prints the full exception without another render.

The current notebook defaults to the built-in video VAE decoder and one render. Dual-T4 tensor-parallel sampling remains enabled. Disabling the optional dual VAE avoids that node, but does not establish why that node failed. Before decoding, the notebook saves both sampled latent tensors to `ComfyUI/output/h3_latents/<run-id>.safetensors`, checks for non-finite values, clears both TP shard references and unloads sampler models. This gives decoding more GPU memory. Future full renders reload shards, so the timing report no longer calls a second render a warm-resident benchmark.

If decoding or export fails after the checkpoint is saved, set `RESUME_CHECKPOINT` to its run ID and rerun the render cell. That submitted graph contains only latent loading, VAEs and export; it contains no sampler or text encoder. Keep the same canvas and duration settings. Checkpoints are written to working output, so they can also survive a saved notebook version. This safeguard cannot retroactively recover a latent that an older notebook never saved.

Audit checks cover checkpoint round-trip with H3 video/audio shapes, both TP cache references, non-finite latent rejection, native temporal chunk planning for 5/22/124 frames, actual H.264 MP4 writing/reading with stereo audio, and ComfyUI graph validation. The decoder geometry tests use synthetic decoded pixels; they do not run pretrained VAE weights. Full T4 inference remains unverified in this CPU workspace.

Run the standard tests with `python -m unittest discover -s tests -v`. With the pinned ComfyUI dependencies installed, run the additional CPU runtime checks using `COMFY_ROOT=/path/to/ComfyUI python tests/test_h3_runtime.py`.

### Confirmed follow-up: helper VAE CUDA OOM

The diagnostic supplied afterward confirms `torch.OutOfMemoryError` in `VAEDecodeH3DualGPU.decode`, specifically `load_models_gpu([helper.patcher], force_full_load=True)`. GPU 1 had 14.36 GiB allocated and 10.81 MiB free. It failed while loading helper weights, before decoding pixels. The retained sampler allocations and a fully resident helper decoder exhausted the card's capacity.

[The one-cell OOM recovery](patches/recover_h3_decoder_oom.py) upgrades the existing Kaggle kernel without reinstalling packages or downloading models. It preserves failure history, installs the checkpoint node, restarts only ComfyUI, and runs one clip using built-in video decoding after sampler-memory cleanup. If a checkpoint from the updated runner already exists, rerunning the recovery cell uses it for decode-only recovery. The initial recovery from an older notebook must resample once because that notebook never persisted its latents. This is a targeted fix with CPU validation; it still needs the user's T4 run to verify full pretrained rendering.
