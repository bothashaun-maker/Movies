# Reusable model packs — v3 startup

Use the same approach as your uploaded **H3_Model_Packs Data Sets.ipynb**:

1. Run **Image_Studio_Model_Packs.ipynb** once with **Internet ON** and **Accelerator: None**, leaving `PACK = 'standard'`.
2. Wait for **PACK COMPLETE**, then **Save Version with output files** and keep it private.
3. Here, choose **Add Input → Your Work / Notebook Output Files** and attach that saved version. Enable **Internet + T4 GPU**, then run the studio cell.

No named Dataset or Kaggle API token is needed for this saved-output approach. `REQUIRE_ATTACHED_MODELS = True` stops startup if weights are missing. Attached files are checksum-verified and read directly without another full model copy. Software setup and the studio tunnel still use Internet.

For FLUX3, build a separate **flux3** pack with accepted Hugging Face terms and Kaggle Secret HF_TOKEN, attach its saved output alongside the standard pack, and enable `ENABLE_FLUX3`. The renderer needs no Hugging Face token once the complete weights are attached.

The generation/upscaling workflows are unchanged from v2. Earlier notebooks remain available.

---

# Image Studio — Kaggle v3

Single-cell ComfyUI image studio, built for Shaun's Kaggle T4 workflow.

1. Enable **Internet** and a **T4 GPU** in Kaggle settings.
2. Run the cell below once; open its **OPEN IMAGE STUDIO** URL.
3. In ComfyUI's workflow menu, open **Image Studio - Create**, **Restyle**, **Inpaint**, or **Detail**, or an upscaling workflow. Change the prompt and press **Run**. No image starts automatically.
4. Images and logs are in `/kaggle/working/image-studio`. **Save Version with outputs** before ending your session. PNG metadata retains the generation graph and seed; drag a PNG into ComfyUI to restore it.

**Create:** 1024×1024, 30 steps, DPM++ 2M Karras, CFG 5.5. Try 832×1216 portrait or 1216×832 landscape, batch 1. These are starting settings, not a universal optimum.

**Restyle:** upload a source with dimensions around 1 megapixel; denoise 0.6 balances source structure and change. Lower preserves more structure. **Inpaint:** upload, right-click the image node → Open in MaskEditor, paint the area to change, save the mask and run. Uses SDXL latent-noise masking, not a dedicated inpainting checkpoint; seams and identity changes remain possible.

**Detail:** 1024px generation followed by 1.5× latent enlargement and a 0.3-denoise detail pass. This is generative detail, not faithful super-resolution; it can change faces/text. It is slower and uses more VRAM. If memory fails, use Create or reduce the selected resolution explicitly. No automatic quality reductions occur.

**Model:** [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9), about 7.1 GB, verified with the model file's published SHA256. Read its model card and underlying SDXL license before commercial use. This is a practical SDXL quality option for T4, not a claim to be the best model for every image. One T4 runs the graph; a second T4 is not pooled VRAM. No paid image API is used.

The checkpoint is reused during the session. Use the CPU model pack builder to save it once; attached inputs are discovered and checksum-verified. Model weights/runtime live in `/kaggle/temp`, not in saved image outputs. Dependencies require Internet even with an attached checkpoint. ComfyUI and cloudflared are pinned; Kaggle's Torch stack is preserved with constraints. Future Kaggle image changes may need a compatibility update.

**Validation limit:** see the research and validation report linked beside the notebook for completed local checks. Kaggle GPU generation, output quality, T4 speed, and public tunnel reliability remain unverified. If setup fails, retain `comfy.log` and `tunnel.log`. Keep the startup cell running; stopping it stops its own server and tunnel. Quick Tunnel URLs grant access to anyone holding the URL.

## Upscaling added in v2

**Upscale 2x / Upscale 4x:** upload an existing image in the Load Image node and press Run. Uses the official RealESRGAN x4plus checkpoint and ComfyUI's tiled model upscaling. The 2x option runs the 4x model and downsamples with Lanczos. This can still change small details; inspect faces and text. A 1024×1024 input becomes 2048×2048 or 4096×4096 respectively.

**Upscale Creative:** upload an image, replace the positive prompt with a description of that image, then Run. Uses Ultimate SD Upscale with 768px tiles, batch 1, 20 steps and denoise 0.20; starts at 2x. It reuses your SDXL checkpoint. Expect more compute and possible invented texture, face changes and seams. It is a connected graph with the upstream Ultimate SD Upscale node; the pipeline is not hidden inside a new wrapper.

**Restore SeedVR2:** native ComfyUI SeedVR2 3B INT8 workflow with FP16 compute, tiled VAE encoding/decoding, one sampling step and a 2x starting size. This is generative restoration and can change detail. The diffusion transformer is not spatially tiled: VAE tiling alone does not guarantee a large target fits in VRAM. Start with a source at 1024px or below. Increase the multiplier only after a successful run. No automatic render or hidden resolution reduction.

The standard model pack contains about 11.1 GB of model files. Set `LOAD_SDXL = False` to skip the 7.1 GB SDXL download and show only the enlargement/restoration workflows; set `ENABLE_SEEDVR2 = False` to skip its approximately 4.0 GB. RealESRGAN adds about 67 MB. First setup needs additional dependency space. Attached Kaggle inputs with matching filenames and SHA256 values are reused. An existing v1 checkpoint is also reused within the same session.

Stop the older studio cell before starting v2. The new notebook has a separate runtime, current pinned ComfyUI, and model-specific workflows. Earlier v1 remains available. Keep your original images; upscaling cannot recover unknowable ground truth.

## Optional FLUX 3 Klein generation

Set `ENABLE_FLUX3 = True` only if you want this model. Accept the [FLUX 3 Klein Comfy model terms](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), create a Hugging Face read token with access to that model, and store it in Kaggle **Secrets → HF_TOKEN**, enabled for this notebook. Never paste a token into a prompt or notebook code. You can set `LOAD_SDXL = False` to avoid downloading the older generator too.

The **Create FLUX3** graph expands ComfyUI's official template into visible ordinary nodes. It retains the required Qwen3.5 prompt upsampler, shows the enhanced prompt, uses the INT8 distilled generator/text encoder, a W4A8 prompt enhancer and tiled VAE decode. Starts at 1024×1024, 16 steps, Euler/simple, CFG 1. The extra models total about 17.75 GB. This profile is experimental here: graph loading is checked, but gated model download, T4 inference, speed and visual quality are unverified. An 8 GB RTX 3070 claim from upstream does not establish T4 performance. T4 uses FP16 compute and compatible INT8/dequantization paths; no FP8/NVFP4 hardware acceleration is assumed. The weights use the FLUX Kommunity License, distinct from the code's Apache-2.0 license.
