# JJP Image Studio Commercial v9 — one saved dataset

**CPU builder → one ALL MODELS Dataset → studio.** The working v3 private Dataset upload method is retained. Automatic finishing-model preparation has been removed from studio startup.

1. Run **Image_Studio_All_Models_Dataset.ipynb** on CPU. No Dataset Inputs are required for the builder; any existing matching files are reused automatically. The default `PACK='all'` saves all eight models (~20.53 GB / 19.12 GiB) to one private Dataset. Allow sufficient disk and account quota.
2. Attach that one saved Dataset to **Image_Studio_Commercial_v9.ipynb**. Select a GPU and enable Internet for software setup/tunnel, then run the single startup cell.
3. Startup verifies saved models, opens persistent ComfyUI and installs all 24 workflows. It does not download models, prepare packs, publish Datasets or generate images. Choose a workflow and press Run.

Missing saved models stop startup with their filenames. Keep the current model files and Datasets until the combined Dataset is available. Use the completed ALL MODELS Dataset as the studio’s single input. The deleted faceswap pack is not needed.

Included: FLUX.2 Klein **4B distilled** generation, native single/multiple-reference editing, masked editing, two reference identity editing, CPU mesh face compositing, Real-ESRGAN 2×/4×, SeedVR2 restoration, approved cast, eight JJP episode worlds, custom world/palette/wing controls, deterministic branded square/landscape/vertical exports. All generation uses visible connected native ComfyUI nodes. No FaceID, ArcFace, InsightFace, GFPGAN or CodeFormer weights are silently downloaded.

Identity editing is generative: inspect facial likeness, hands, clothing and background changes. Describe image 1 as the target and image 2 as the identity reference; identify which person to change. For duets, map images 2 and 3 to Lead One and Lead Two explicitly. It is not a measured substitute for a neural face-swap identity metric. The mesh compositor is supplemental and showed mouth/edge artifacts in the CPU sample; it is not the recommended quality replacement. It transfers source pixels with landmarks; choose similar pose, expression and lighting. Indices are zero-based, left to right. Large pose changes, occlusion, hair and motion remain limitations. This notebook processes stills, not a temporally consistent video face-swap.

The distilled model uses four steps and CFG 1. Negative text in the JJP brief is retained for review but is not active negative guidance at CFG 1. Masked edit first generates a reference edit, then composites it within your painted LoadImage mask. Use MaskEditor and a white replacement mask. This preserves the unmasked target except for the selected feather band; it does not use a dedicated inpainting checkpoint. Export resizing is interpolation, not neural upscaling. Use the upscaling workflow for learned detail first.

Default weights need about 20.53 GB in one pack, plus runtime disk. Generation uses GPU 0; two T4s do not form a single shared VRAM pool. Low-VRAM/FP16 flags are supplied. Full FLUX inference on Kaggle/T4, peak memory, image quality and live Cloudflare remain unverified here. Do not treat model-author speed/memory claims as measured notebook results. See VALIDATION.md and LICENSE_AUDIT.md.
