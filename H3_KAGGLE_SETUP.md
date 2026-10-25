# MiniMax H3: complete Kaggle T4 ×2 notebook

Use [minimax_h3_kaggle_complete.ipynb](minimax_h3_kaggle_complete.ipynb). This replaces the earlier notebook and its separate repair cells. All runtime and recovery code is embedded; no helper script is fetched from this repository during execution.

1. Import the complete notebook into Kaggle.
2. Select **GPU T4 ×2** and **Internet ON**.
3. Edit the prompt and, optionally, an attached first-frame image in step 1.
4. **Run All**. Step 7 previews the result and places a downloadable MP4 at the top of `/kaggle/working`.

The default is one 512 × 768 clip, four Turbo LoRA steps, 124 frames at 24 fps, and native audio. It does not run a second benchmark clip. CUDA 13 requires driver 580 or newer; the notebook checks the driver and tests both T4s before downloading missing weights.

## How the rewrite prevents the observed decoder conflict

The supplied diagnostic confirms that `VAEDecodeH3DualGPU` failed while fully loading its helper VAE on GPU 1: 14.36 GiB was already allocated and only 10.81 MiB remained free.

The rewritten notebook runs two separate ComfyUI processes:

| Stage | Work | Completion condition |
| --- | --- | --- |
| Sampling | Two-GPU INT8 ConvRot model, matching four-step LoRA, conditioning | Joint video/audio latents saved atomically and checked |
| Process boundary | Terminate and wait for the sampler process | Sampler process has exited before decoding starts |
| Decoding | Load saved latents; native tiled video VAE on GPU 0, audio VAE on GPU 1; H.264 MP4 | Decoder exits; the entire MP4 is read and frame count, dimensions and audio are verified |

The decode graph contains no diffusion model, text encoder, LoRA or sampler. The native H3 VAE handles 256-pixel spatial tiles and temporal chunks internally. The dual-helper VAE node is not used.

The controller closes its processes on success, failure and keyboard interruption. It can also identify and stop an older server only when its Python executable and ComfyUI directory match this notebook's own paths. It does not kill unrelated services occupying the port.

## Retry without spending GPU time on sampling again

After successful sampling, step 6 saves `minimax_h3_dual_t4/ComfyUI/output/h3_latents/<run-id>.safetensors`. The run ID depends on the sampling graph, prompt, seed, model hashes, source revisions and prepared reference image.

If decoding fails, rerun **step 6** with the same settings. It validates and reuses that checkpoint automatically. A completed identical job reuses its verified MP4. Change the seed or prompt to generate another clip. Initial sampling failures cannot be resumed from an unsaved checkpoint.

Graphs, per-stage logs, full execution histories and concise exceptions are in `minimax_h3_dual_t4/runs/<run-id>/`. Exceptions are retained separately so large tensor dumps cannot hide the cause.

Keep the same session to reuse scratch models. **Save Version** retains working outputs, including checkpoints and video; it does not preserve temporary model storage.

## Storage and downloads

The default model payload is **41.38 GiB**. The notebook searches attached `/kaggle/input` files and existing model paths, then measures free space on `/kaggle/temp`, `/tmp` and working storage. It never assumes a 50 GiB quota or adds together directories on the same filesystem.

Downloads and staging stay on the selected model filesystem. Symlinks expose files to ComfyUI without copying the weights. New downloads receive one SHA-256 check; existing files are verified before use. Interrupted Hugging Face downloads can resume. Re-running only step 6 skips model setup and hashing.

Downloads and hash verification use network, disk and CPU, so low GPU utilization during step 4 is expected. Their duration depends on the session and connection. The notebook makes no one-minute rendering or download promise.

To avoid downloading weights during GPU sessions, prepare these inputs with **Accelerator: None** and **Internet ON**, save their outputs, and attach them through Kaggle's **Add Input → Your Work**:

| Preparation notebook | Files | Payload |
| --- | --- | --- |
| [Diffusion](notebooks/minimax_h3_prepare_diffusion.ipynb) | Ref2VA INT8 ConvRot checkpoint | 19.53 GiB |
| [Encoder](notebooks/minimax_h3_prepare_encoder.ipynb) | Qwen3VL NVFP4 AWQ encoder | 14.61 GiB |
| [Support](notebooks/minimax_h3_prepare_support.ipynb) | Four-step LoRA, video and audio VAEs | 7.24 GiB |

Each checks its actual free space and verifies pinned hashes. Leave `MODEL_DATASET=''` in the renderer to auto-discover nested attached files. For FL2VA, use `MODEL_FAMILY='fl2va'` consistently in preparation and rendering.

## Acceleration and optional Semantic Bridge

The pinned stack uses Python 3.12.12, PyTorch 2.10 / CUDA 13, ComfyUI v0.37.0, and the `kaggle-ok-64` T4 tensor-parallel fork. It includes the per-layer `.comfy_quant` compatibility reader needed by the pinned Comfy-Org checkpoint. Model files are not modified.

Four-step Turbo LoRA, native INT8 ConvRot kernels, protected FP16 and Comfy Kitchen attention are enabled. The FastVideo VSA sparse-attention kernel path is not a validated SM75 implementation in this stack and is excluded. Semantic Bridge is available with `MODEL_FAMILY='fl2va'` and `USE_BRIDGE=True`; its author excludes Ref2VA, so that unsupported combination is rejected before downloads.

## Validation and limits

The uploaded Kaggle runs already establish that the pinned FP16 and ConvRot INT8 probes passed on both T4s. The rewrite additionally checks those probes on each actual session.

Development checks exercise actual pinned ComfyUI graph validation for Ref2VA/FL2VA text and image workflows and the supported bridge workflow; terminal checkpoint execution and round-trip; non-finite latent rejection; process reaping after interruption; decoder failure followed by retry without resampling; complete error preservation; native temporal output geometry; and real H.264 MP4 export/readback with audio. Negative graph validation is also tested.

**A full pretrained render of this rewrite on Kaggle T4 ×2 has not been executed in the CPU development environment.** Graph and controller tests do not establish final rendering time, visual quality or measured peak VRAM.

The standalone notebook is generated from [reviewable Python sources](notebooks/h3_complete). With `nbformat` installed:

```bash
python notebooks/build_h3_complete.py
python -m unittest discover -s tests -v
```

For additional checks, install the dependencies of the pinned ComfyUI source and set `COMFY_ROOT` and `H3_MGPU_ROOT` to those checkouts. Set `H3_BRIDGE_ROOT` to the extracted pinned bridge package to include its optional schema case:

```bash
COMFY_ROOT=/path/to/ComfyUI H3_MGPU_ROOT=/path/to/ComfyUI-MultiGPU-Kaggle \
H3_BRIDGE_ROOT=/path/to/MiniMax_H3_Semantic_Bridge \
python -m unittest discover -s tests -v
```

The older notebook and repair scripts remain for reference. Use the complete notebook above for new runs.
