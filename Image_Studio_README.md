# JesusJuiceProductions Image Studio — ONE Dataset

**[1. NEW: ALL MODELS Dataset Creator (CPU)](Image_Studio_All_Models_Dataset.ipynb)**

**[2. NEW: Commercial Studio v9 (GPU)](Image_Studio_Commercial_v9.ipynb)**

Run the new CPU creator once. It puts **all eight reviewed models into one private Dataset** using the working v3 upload method. No Dataset Inputs are required for the creator; matching saved files are reused when available. The weights total **20.53 GB / 19.12 GiB**, plus small manifests and license notices. The builder checks available disk; your account quotas apply.

Attach **only that ONE completed Dataset** to the new studio. All 24 workflows are included in the notebook. Studio startup verifies saved weights and opens ComfyUI; it does not download models, prepare extra packs or create Datasets. Software installation and the tunnel still need Internet.

Models: Klein 4B, Qwen3-4B encoder, Flux2 VAE, Real-ESRGAN x4plus, SeedVR2 model and VAE, Google face landmarks, BFS 4B head-swap adapter. No InsightFace research-only weights are included. This is the complete reviewed image stack, not a collection of all historical models.

[Production playbook PDF](JJP_Image_Production_Playbook.pdf) · [Current setup](notebooks/image_studio_commercial/README.md) · [Dataset instructions](notebooks/image_studio_commercial/MODEL_PACKS.md) · [License audit](notebooks/image_studio_commercial/LICENSE_AUDIT.md) · [Validation](notebooks/image_studio_commercial/VALIDATION.md)

The earlier split-pack/automatic-finish startup approach is superseded for this workflow. Do not import an older copy expecting it to update automatically. Keep existing files until the complete Dataset is available. Full live Kaggle publication of the combined pack and GPU inference remain unverified here.
