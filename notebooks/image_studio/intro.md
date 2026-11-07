# Image Studio — Kaggle v1

Single-cell ComfyUI image studio, built for Shaun's Kaggle T4 workflow.

1. Enable **Internet** and a **T4 GPU** in Kaggle settings.
2. Run the cell below once; open its **OPEN IMAGE STUDIO** URL.
3. In ComfyUI's workflow menu, open **Image Studio - Create**, **Restyle**, **Inpaint**, or **Detail**. Change the prompt and press **Run**. No image starts automatically.
4. Images and logs are in `/kaggle/working/image-studio`. **Save Version with outputs** before ending your session. PNG metadata retains the generation graph and seed; drag a PNG into ComfyUI to restore it.

**Create:** 1024×1024, 30 steps, DPM++ 2M Karras, CFG 5.5. Try 832×1216 portrait or 1216×832 landscape, batch 1. These are starting settings, not a universal optimum.

**Restyle:** upload a source with dimensions around 1 megapixel; denoise 0.6 balances source structure and change. Lower preserves more structure. **Inpaint:** upload, right-click the image node → Open in MaskEditor, paint the area to change, save the mask and run. Uses SDXL latent-noise masking, not a dedicated inpainting checkpoint; seams and identity changes remain possible.

**Detail:** 1024px generation followed by 1.5× latent enlargement and a 0.3-denoise detail pass. This is generative detail, not faithful super-resolution; it can change faces/text. It is slower and uses more VRAM. If memory fails, use Create or reduce the selected resolution explicitly. No automatic quality reductions occur.

**Model:** [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9), about 7.1 GB, verified with the model file's published SHA256. Read its model card and underlying SDXL license before commercial use. This is a practical SDXL quality option for T4, not a claim to be the best model for every image. One T4 runs the graph; a second T4 is not pooled VRAM. No paid image API is used.

The checkpoint is reused during the session. To avoid future downloads, save it separately as a private Kaggle model/input with its original filename; attached inputs are discovered and checksum-verified. Model weights/runtime live in `/kaggle/temp`, not in saved image outputs. Dependencies require Internet even with an attached checkpoint. ComfyUI and cloudflared are pinned; Kaggle's Torch stack is preserved with constraints. Future Kaggle image changes may need a compatibility update.

**Validation limit:** notebook/source and graph structure checked locally; no Kaggle GPU generation, output quality, T4 speed, or public tunnel reliability has been verified. If setup fails, retain `comfy.log` and `tunnel.log`. Keep the startup cell running; stopping it stops its own server and tunnel. Quick Tunnel URLs grant access to anyone holding the URL.
