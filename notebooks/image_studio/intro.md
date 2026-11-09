# JesusJuiceProductions — Image Studio v6

The supplied **JJP Launch System** is now available as an image-production layer across creation, restyling, inpainting, detail, face swap and every upscaling mode. Original general workflows remain available.

## Start

Attach your existing standard model pack, enable **Internet + T4 GPU**, and run the single cell. Optional FLUX3 still needs its existing model pack and `ENABLE_FLUX3=True`. No new generator weights are required by the JJP tools. The v5 fix remains: missing face models are prepared on CPU and private Dataset submission is attempted using existing authentication. Valid attached/cached models are reused. Software setup and the studio tunnel still need Internet.

Open the printed studio URL, then select a workflow and press **Run**. Startup never renders automatically. Stop an earlier studio cell before starting this one; v6 reuses the compatible existing runtime. GitHub updates do not change an already imported Kaggle copy—import this v6 notebook.

## The JJP menus

- **JJP Episode 01–08:** editable connected creation graphs for the eight worlds in your PDF, from Verdict of the Lamb through Love Your Enemies.
- **JJP - Create / Restyle / Inpaint / Detail:** cinematic realism, world/palette, cast prefix and shot-purpose controls, with source-art dimensions connected to generation where applicable.
- **JJP - Upscale 2x / 4x / Creative / Restore SeedVR2:** existing enhancement engines with the same shared brief and artwork export. The brief does not secretly change a RealESRGAN model or guarantee identity preservation.
- **JJP - Face Swap:** manual source and target photo inputs. **Face Swap from Cast** loads an approved identity. **Duet Face Swap from Cast** applies Lead One and Lead Two in two visible swap stages; choose target indices carefully (0 = leftmost).
- **JJP - Approve Cast:** upload and inspect a reference, set its role and reusable prompt prefix, set `approved=true`, then Run. The library starts empty. Save private outputs and reattach them in future sessions; refresh node options after adding references.

Your references stay in the private runtime/output library, not this public GitHub repository. The toolkit does not invent your cast's appearance or automatically approve a generated likeness. Text prompts are not multi-image identity conditioning. Use approved references, review the final faces, and reject drift.

## Artwork and records

The shared **JJP Episode / Shot Brief** has episode, format, scene, cast, character prefix, title and exact scripture-reference fields. A preset reference can be disabled or replaced. Missing references stay empty; no quotation or translation is generated.

The **JJP Artwork + Episode Export** node preserves an untouched `master.png` and writes `delivery.png` plus `episode.json` under `image-studio/images/JJP/episodes`. Titles and references are drawn after generation with consistent typography. Choose original dimensions, square, 16:9 or 9:16; aspect conversion pads rather than cropping the source. Delivery resizing is Lanczos interpolation, not a learned upscale. Run the existing upscaler first when you need learned detail. Disable lettering for clean reference artwork; face-swap delivery defaults to lettering off.

Keep your model packs, approved cast and finished images distinct. Model Datasets do not back up photos or generated outputs. Save private outputs before discarding Kaggle. Files are not posted to any social platform automatically. The full companion guide is `notebooks/image_studio/JJP_PLAYBOOK.md` on GitHub.

## Existing controls and limits

SDXL/Juggernaut starts around one megapixel, 30 steps, DPM++ 2M Karras, CFG 5.5. Restyle starts at denoise 0.6; inpaint uses SDXL latent-noise masking, not a dedicated inpaint checkpoint. Detail uses a 1.5x latent pass with denoise 0.3. These are editable starting settings, not universal quality claims.

RealESRGAN x4plus supplies 2x/4x enlargement (2x uses 4x then Lanczos downsampling). Ultimate SD Upscale starts at 2x, 768px tiles, 20 steps, denoise 0.20. SeedVR2 3B INT8 starts at 2x with tiled VAE and one sample step; VAE tiling alone does not guarantee large targets fit transformer VRAM. Start with a source around 1024px or smaller. These engines can alter detail and likeness.

Face swap runs on CPU through InsightFace/InSwapper; its internal aligned face crop is 128px. These pretrained face weights are **non-commercial research only**. The PDF's release plan does not change model licensing. Use photos you own or have permission to use. No commercial face model replacement is implied.

FLUX3 remains optional, with its required prompt upsampler, INT8 generator/text encoder and W4A8 enhancer. It requires accepted model terms/HF_TOKEN when building its pack, not when all weights are already attached. Its T4 performance remains unverified here. Separate T4s are not pooled VRAM.

The source PDF, private reference photos and personal biographies are not included in the public repository. Review likeness/dignity, antagonist design, spelling, scripture accuracy, model rights and relevant synthetic-content disclosure before a release. These are production aids, not automatic approval or guaranteed theological/visual accuracy.

Local tests cover presets, cast approval/checksums, exact reference strings, unchanged master pixels, metadata, export geometry and ComfyUI/browser wiring. Full Kaggle/T4 generation, real Dataset upload and user-photo likeness remain unverified. Keep the startup cell running while using the studio; stopping it closes its own server and tunnel. Anyone holding a live Quick Tunnel URL can access that session.
