> **Superseded for commercial image production.** Read [the v7 source review](../image_studio_commercial/RESEARCH.md) and [complete license audit](../image_studio_commercial/LICENSE_AUDIT.md). This historical page selected models before the commercial-use correction.

# Image Studio v2 — research, decisions and validation

Researched 7 November 2026 for Kaggle NVIDIA T4. This is a targeted source review, not an exhaustive scrape of GitHub and not a visual benchmark. No repository's stars, tutorial title or creator benchmark is treated as proof of best quality on a T4.

## What to use

For existing images, start with **Upscale 2x** using RealESRGAN x4plus. Use **Upscale 4x** when the additional pixels are needed. Compare **Restore SeedVR2** when the input needs reconstruction. Use **Upscale Creative** when you deliberately want the SDXL model to generate additional texture. Keep the original for side-by-side comparison: none can reconstruct unknowable ground truth or guarantee faithful text and faces.

For creation, SDXL/Juggernaut remains the no-token baseline with straightforward masked editing and compatible tiled enhancement. **FLUX 3 Klein** is the newer generation option added after this review. Its required prompt upsampler is retained. It is optional because of model-access terms, another approximately 17.75 GB of weights, and unverified T4 inference here. The default configuration does not silently download it. This does not establish a winner between generators on the user's subjects.

## Creator review

| Source | Evidence reviewed | Consequence |
| --- | --- | --- |
| [Pixaroma: upscaling Ep05](https://www.youtube.com/watch?v=BRHodz0_Uc4), [earlier Ep12](https://www.youtube.com/watch?v=i8v9RbNy4Zw), [ComfyUI-Pixaroma](https://github.com/pixaroma/ComfyUI-Pixaroma) | Indexed tutorial information, repository tree and resize documentation. | Keep enlargement, resizing, latent detail passes and generative enhancement as distinct workflows. A loader's resize option alone is not a neural restoration model. The whole Pixaroma extension is not needed for these graphs. |
| [AIQUEST / TeamAIQ notebooks](https://github.com/TeamAIQ/Colab-notebooks), [HYPIR tutorial](https://www.youtube.com/watch?v=WIfnA2PGVSY) | Cloned collection and inspected HYPIR notebook code, model identifiers, initialization and dependencies; reviewed its current Qwen image notebook as another generator candidate. | HYPIR is a real restoration candidate with a T4-oriented notebook. Its upstream restrictions and separate Diffusers/PEFT runtime matter; do not blindly install its Torch pins into ComfyUI. |
| [AI Late to Class channel](https://www.youtube.com/@ailatetoclass), [public upscale comparison post](https://www.patreon.com/ailatetoclass/posts/best-ai-upscale-143181594), [FLUX 3 video](https://www.youtube.com/watch?v=-lcjNIZaUwM) | Public indexed titles/pages; inspected the discoverable `ailatetoclass-1/aistuff` repository, which contains a minimal placeholder readme rather than a reusable upscaling implementation. | Investigated SeedVR2, SUPIR, ESRGAN and newer FLUX instead of assuming SDXL was the best current option. No paid workflow was obtained or copied. The sparse GitHub account alone does not verify ownership of a particular implementation. |

Direct YouTube retrieval from the execution environment returned a 403 tunnel failure; the web reader exposed limited video information. No full video playback or transcript review was completed. Creator titles and public descriptions were used to identify candidates; implementation decisions were checked against source repositories and model metadata.

## Candidate comparison

| Candidate / primary source | Fit and limitations | v2 decision |
| --- | --- | --- |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Compact established model; ComfyUI already implements tiled inference and safe loading through Spandrel. Code BSD-3-Clause. It can oversmooth or sharpen incorrectly. | Include official x4plus checkpoint, verified by SHA256; separate 2x and 4x workflows. No face-restoration pass is forced. |
| [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) and [original script](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) | Reuses SDXL to redraw overlapping tiles. Adjustable denoise and seam handling. Slower and can alter identity/text. GPL-3.0 extension. | Include pinned extension and pinned submodule, 2x default, 768px tiles, batch 1, denoise 0.20, tiled decode. |
| [SeedVR2 native ComfyUI template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/utility_seedvr2_3b_int8_upscale_image.json), [weights](https://huggingface.co/Comfy-Org/SeedVR2), [upstream custom implementation](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | Native 3B INT8 + FP16 VAE is approximately 4.0 GB. The separate custom extension has block swapping/GGUF and additional controls, but is not the same runtime as this native graph. VAE tiling does not spatially tile transformer attention. | Include native graph, tiled VAE, 2x default and one sampling step. Do not claim arbitrary 4K/8K targets fit a T4. |
| [HYPIR](https://github.com/XPixelGroup/HYPIR) | Single-step diffusion restoration. Official requirements pin a separate older Torch/Diffusers/PEFT stack. Repository contains a non-commercial-use declaration. | Research candidate; not bundled into the generator's runtime. AIQUEST's notebook remains a linked alternative for evaluation. |
| [SUPIR](https://github.com/Fanghua-Yu/SUPIR) | SDXL restoration with heavier dependencies and optional captioning. Upstream describes a slower reduced-memory mode, with 12 GB diffusion and 16 GB LLaVA figures. Non-commercial declaration. | Not a default T4 install; do not confuse upstream component memory figures with a measured full Kaggle pipeline. |
| [FLUX 3 Klein](https://github.com/black-forest-labs/flux-3-image-klein), [Comfy weights](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) | Newer generation, reference editing and layout capabilities. Upstream explicitly requires its prompt upsampler. Native reference inference is oriented toward newer accelerators; Comfy INT8 is the relevant older-GPU route. Weights use FLUX Kommunity terms, not the code's Apache-2.0 license. | Optional visible text-to-image graph with required prompt enhancement; no reference-editing or layout UI is claimed in this v2 graph. Requires accepted model terms and Kaggle Secret HF_TOKEN. |
| [FLUX.2](https://github.com/black-forest-labs/flux2), [Z-Image](https://github.com/Tongyi-MAI/Z-Image), Qwen image family in AIQUEST's collection | Other credible generation options with different speed, editing, license and memory tradeoffs. | Reviewed as alternatives; not added as redundant untested installs. No side-by-side visual ranking was performed. |

Paid hosted upscalers and desktop-only applications were not selected for a self-contained Kaggle ComfyUI deliverable. More models do not establish a better measured result, and desktop support does not establish Kaggle/T4 compatibility.

## Frozen implementation

- ComfyUI `1332fabfcc507aa7c548cd5152071ed494c207e8` (0.44.0).
- Ultimate SD Upscale `a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df`; its submodule resolves to `2322caa480535b1011a1f9c18126d85ea444f146`.
- SeedVR2 weight repository revision `df48879708206a403d2a61acd55578c2e80fd233`.
- FLUX 3 Comfy weight repository revision `9d36114cf7a2e03a45c28f5cc4987d20ebfdb6c1`.
- FLUX graph adapted from Comfy-Org's `image_flux3_klein_distilled_t2i.json`, inspected at workflow-template commit `21d7cc4fb9afd983957f2d2d6b1d583860ccb40b`. Its subgraph was expanded; prompt enhancement stays connected and enabled. SDXL and enlargement graphs are generated locally.
- Every model asset and the Cloudflare executable has a SHA256 check. The pinned asset catalog is embedded in the notebook. No custom-node manager or arbitrary extension search runs during startup.
- Existing v1 remains unchanged. v2 uses its own runtime, reuses verified v1 weights and attached inputs, and checks port ownership before downloading. Only the startup's own children are terminated on cell interruption.
- Kaggle's Torch/TorchVision/TorchAudio/Numpy/SciPy versions are constrained rather than replaced. Dependency compatibility can still change with Kaggle's base image. Actual installed versions and asset provenance are saved with outputs.

## Validation

Completed locally:

- Four regression checks: executable notebook compilation, graph edges/reachability, live native-node input schemas, and embedded graph consistency.
- Notebook v4 schema validation.
- Real ComfyUI 0.44.0 startup on CPU with the pinned Ultimate extension loaded.
- All nine graphs load in Chromium as registered nodes. Widget checks verify the creative upscale denoise/tile/batch values and FLUX3 step/CFG settings. Missing large models and unselected uploads are expected in this development browser; these are not successful inference validations.
- Real official RealESRGAN x4plus inference through ComfyUI on a synthetic 24×24 image. The 2x workflow produced a 48×48 PNG; 4x produced a 96×96 PNG. This validates wiring and output geometry, not photographic restoration quality or T4 speed.

Not validated: Kaggle GPU startup, T4 inference or memory peaks; full-size SDXL/SeedVR2/FLUX3 outputs; gated FLUX3 download; prompt-enhancer generation; comparative visual quality; live public Cloudflare access. No Kaggle GPU quota was used. There are no invented performance numbers or promises of the best possible result.

## Rebuild and test

From the repository root:

```sh
python notebooks/image_studio/build.py --output Image_Studio_Kaggle_v2.ipynb
STUDIO_NOTEBOOK=Image_Studio_Kaggle_v2.ipynb COMFY_URL=http://127.0.0.1:8189 python notebooks/image_studio/check.py
PLAYWRIGHT_BROWSERS_PATH=/path/to/browser/cache python notebooks/image_studio/browser_check.py
```

The checks expect a running matching ComfyUI server with Ultimate SD Upscale. The CPU inference check also needs Pillow and the official RealESRGAN checkpoint:

```sh
COMFY_ROOT=/path/to/ComfyUI REAL_ESRGAN_MODEL=/path/to/RealESRGAN_x4plus.pth python notebooks/image_studio/integration_check.py
```

For source checkout tests, set `STUDIO_NOTEBOOK` to the rebuilt notebook's path. Store outputs before stopping Kaggle; downloading the notebook alone does not back up its generated PNGs.
