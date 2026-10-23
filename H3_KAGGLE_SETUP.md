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
