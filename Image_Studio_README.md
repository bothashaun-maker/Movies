# Image Studio — normal image workflows

**[Open Image Studio Normal v10](Image_Studio_Normal_v10.ipynb)**

Attach your existing **ONE ALL MODELS Dataset**. Select Kaggle GPU T4, enable Internet for software/tunnel setup, run the single code cell and open ComfyUI. Choose **Image Studio - Create - Cinematic**, write your prompt and press Run. Your Dataset, models, credentials and uploader are unchanged: **do not rebuild the Dataset for this notebook**.

Normal workflows only:

- Create cinematic images with ordinary prompt, width, height and seed controls.
- Edit images or paint a mask for local edits.
- Face identity editing, masked face editing, BFS head replacement and a simple pixel-composite fallback.
- Upscale 2×, Upscale 4× and SeedVR2 restoration. These original upscaling graphs are unchanged.

No episode presets, scripture fields, cast-approval stages, required branding or lettered exports. PNGs save normally with graph/settings metadata. Old app-managed canvases are preserved in an output archive outside the workflow menu; existing images and personal workflows are not deleted.

[Normal workflow instructions](notebooks/image_studio_normal/README.md) · [Individual workflow JSONs](notebooks/image_studio_normal/workflow_files) · [Validation](notebooks/image_studio_normal/VALIDATION.md)

If you already have the ALL MODELS Dataset, keep it. The [existing CPU Dataset creator](Image_Studio_All_Models_Dataset_v2.ipynb) remains available separately and is not changed by this workflow correction. Startup verifies saved models and does not prepare model packs or create Datasets.

Full FLUX/BFS/SeedVR2 GPU inference and comparative cinematic quality remain unverified here. Existing commercially reviewed weights are retained; research-only InsightFace weights are not reintroduced.
