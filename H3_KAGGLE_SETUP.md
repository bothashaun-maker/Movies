# Open ComfyUI through Cloudflare

Import `minimax_h3_kaggle_complete.ipynb`, select T4 ×2 and Internet ON, and press its single Play button. It sets up the runtime and models, then displays **Your ComfyUI Cloudflare link**. Open that link, select **Workflows → MiniMax H3 - Full T4 Workflow**, edit the prompt and press **Run**. The notebook installs this adapted workflow automatically; no manual JSON import is required. No render, video preview or download copy runs automatically.

Keep Kaggle running. The URL is temporary and also saved to `minimax_h3_dual_t4/comfyui-cloudflare-url.txt`. Logs: `comfyui-ui.log` and `cloudflared.log`. Call `stop_comfyui_access()` to close the tunnel and owned UI server. Cloudflared 2026.10.3 is downloaded from its official release and SHA-256 verified.

The latest supplied Kaggle run completed the isolated renderer successfully: 124 frames, 512 × 768, native audio. Cloudflare access must be established from the running Kaggle session; it is not a static GitHub URL.

## Installed workflow

Select **MiniMax H3 - Full T4 Workflow**, not an already-open tab from the earlier one-node version. It is installed automatically under `ComfyUI/user/default/workflows/`. The old filename also receives the connected graph, but an existing unsaved browser tab may retain its previous contents.

The canvas has **17 nodes and 19 links**: diffusion and CLIP loaders, video/audio VAE loaders, Turbo LoRA, sigma shift, conditioning, noise, guider, sampler selection, scheduler, advanced sampler, video/audio decode, CreateVideo and isolated Save MP4. An optional Load Image node is included; upload an image and connect IMAGE to `first_frame` on the conditioning node to use it. Text generation works with that input disconnected.

All visible settings and connections are used. The final isolated save node receives the submitted graph without executing the model nodes in the browser server. It partitions that graph into sampling and decoding worker processes, saving joint latents between them. This preserves the process boundary that resolved the original GPU-memory conflict. Results appear in ComfyUI, not in the Kaggle notebook.

[Actual browser screenshot](docs/h3_full_workflow.png) · [Workflow JSON](workflows/minimax_h3_kaggle_t4.json)

The canvas was loaded in the pinned ComfyUI browser interface with no page errors. Its actual exported API graph and both resulting worker graphs passed ComfyUI validation for text and image input. An executor test verifies that native model dependencies do not run in the UI process. These are browser/CPU checks; the revised full-canvas path has not yet completed a T4 render in this development environment.

## Storage and downloads

The default model payload is **41.38 GiB**. The notebook searches attached `/kaggle/input` files and existing model paths, then measures free space on `/kaggle/temp`, `/tmp` and working storage. It never assumes a 50 GiB quota or adds together directories on the same filesystem.

Downloads and staging stay on the selected model filesystem. Symlinks expose files to ComfyUI without copying the weights. New downloads receive one SHA-256 check; existing files are verified before use. Interrupted Hugging Face downloads can resume. Rerunning the cell verifies and reuses existing files before reopening browser access.

Downloads and hash verification use network, disk and CPU, so low GPU utilization during weight preparation is expected. Their duration depends on the session and connection. The notebook makes no one-minute rendering or download promise.

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

## Validation

Tests cover the one-cell controller, real metadata subprocess, ComfyUI graphs, checkpoint handling, MP4/audio export, and tunnel link extraction/lifetime using a simulated tunnel process. Cloudflared was downloaded from the official release, its digest recorded and its version command executed. A live Cloudflare connection from Kaggle is not verified here.

Run `python notebooks/build_h3_complete.py` to regenerate the notebook, then `python -m unittest discover -s tests -v`. Runtime tests require the pinned ComfyUI dependencies and COMFY_ROOT/H3_MGPU_ROOT; H3_BRIDGE_ROOT enables bridge schema validation.
