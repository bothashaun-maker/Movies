# Native two-T4 FlashVSR replacement (local; not published)

The uploaded SeedVR2 log records 70 VAE out-of-memory recoveries and 731.22 seconds of encoding for the first 21-frame chunk. ComfyUI's force-FP16 flag did not prevent SeedVR2 from converting its VAE to BF16. The old pipeline also swapped all DiT blocks through CPU and used the GPUs in sequential phases.

This replacement uses TeamAIQ's actual dedicated FlashVSR+ upscaler by pinned source reference, rather than porting H3 tensor-parallel code into SeedVR2. It requires a new FlashVSR model Dataset. Existing H3/Qwen, SeedVR2 and Real-ESRGAN notebooks and Datasets are unchanged.

## Files

- `JJP_FlashVSR_Models.ipynb`: one CPU cell, private authentication, working native streaming Dataset submission. Turbo + Base + compact decoder, 13,843,927,661 bytes total (12.89 GiB). Peak missing-model download is the largest single checkpoint plus reserve. Each model has a pinned revision, exact size and SHA256. No GPU needed.
- `JJP_FlashVSR_T4.ipynb`: one startup cell, requires the ONE completed new Dataset and two available GPUs. Stages and hashes weights sequentially to measured scratch before model loading. No online model download during GPU inference. Fetches and verifies the exact upstream notebook Git blob before extracting the engine.

TeamAIQ source: https://github.com/TeamAIQ/Colab-notebooks/blob/237383fb084b671c1eb5224db822d88fca6fc8fa/kaggle_notebooks/flashvsr-video-upscaler.ipynb

FlashVSR+ code: https://github.com/OpenImagingLab/FlashVSR_Plus/tree/b548a2cc2c929d3d4676f2973abe27876ec4dee7

Models: https://huggingface.co/ruchangyao/FlashVSR_plus/tree/46b79520946ac67dec94d1566490690953c7debc

## Preserved native engine

- All top-level inference functions/classes pass exact AST identity against TeamAIQ: native attention backends, FP16/FP32 precision rules, model loading, resident GPU partitioning, pipeline/rebalancing, decoder, media IO and progress reporting.
- Both GPUs work on consecutive chunks in a pipeline. This is not a claim of pooled VRAM.
- T4 startup intentionally skips slow Triton tensor-core paths and benchmarks native PyTorch candidates. No speculative Sage/Flash patch was added.
- Compact decoder and streaming output replace the SeedVR2 VAE bottleneck.
- Original Gradio upload, output video, Stop control and AIQUEST branding remain.

Only UI defaults change: 2× first, CRF 14, comparison copy off. Turbo / normal motion / veryfast remain native initial choices. The inference code is fetched from its original source, not redistributed inside this package. Backend and model licences remain applicable.

## Spending/process controls

The launcher never kills processes by broad name. Only its recorded PID/create-time/exact script and descendants may be stopped. Other H3/Qwen/SeedVR2 engines are preserved. Low free GPU memory causes a clear early failure. The parent uses nvidia-smi rather than creating extra CUDA contexts.

Startup budget: 360 seconds after engine launch. Per-clip budget: 480 seconds, editable at the top of the notebook. A budget breach terminates the owned engine and children, retaining completed outputs. These are stop limits, not measured speed promises. Parent GPU/download staging time is separate. A failed/partial encode is not a completed output.

## Local validation (no user GPU session)

- Verified raw TeamAIQ Git blob identity against the actual source URL and rejected altered bytes.
- Native inference AST unchanged after UI adaptation.
- All notebook schemas, single-cell entries and embedded source payloads compile and match source.
- Native sparse SDPA, matmul and variable-length attention matched selected-key dense attention on CPU.
- Native geometry verified 960×544 → 1920×1088, with 1920×1152 model padding and final cropping.
- Native video writer produced a real three-frame MP4 at correct dimensions/frame count.
- Source preflight, complete Dataset detection, time-budget decisions and exact owned-process cleanup passed CPU checks.
- Real Kaggle SDK/file-token tests passed with network operations stubbed, including explicit private creation and server-rejection/no-success-receipt behaviour. They are not a live authenticated upload.
- Native Gradio upload checked in Chromium on CPU: a real 960×544 MP4 uploaded and displayed the 1920×1088 output plan without browser errors. UI configuration confirmed 2×, CRF 14, veryfast, and comparison off. No model inference was run in this interface check.

The CPU builder calls Kaggle's Dataset creation API automatically after uploading all five verified model files and the manifest. It prints the private Dataset URL and writes a success receipt only after a non-error creation response. Saving a notebook version is not the Dataset creation mechanism. Kaggle processing must finish before attaching the Dataset to the GPU notebook.

Full-size T4 rendering, actual model-set staging/upload, memory fit and comparative visual quality are not measured here. No under-eight-minute performance guarantee is claimed. This avoids the recorded SeedVR2 execution path by using a different native engine. It is not evidence that the old SeedVR2 implementation was repaired.
