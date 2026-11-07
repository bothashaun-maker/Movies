# Image Studio Model Packs — face swap added

Use **Accelerator: None** and **Internet ON**. No GPU or generation runs here.

1. Leave `PACK = 'faceswap'` for the new face models (**843 MB**). Your existing standard image pack remains usable; do not rebuild it just for face swap.
2. Run the single code cell. It downloads and verifies the files, then—with `AUTO_CREATE_DATASET = True`—uses your existing Kaggle authentication to submit a new **private Dataset** and print its link.
3. After Kaggle finishes processing, attach that Dataset to **Image Studio v4** via **Add Input → Your Datasets**, alongside your existing standard pack.

Available packs: **faceswap** ~843 MB, **standard** ~11.1 GB (SDXL, RealESRGAN, SeedVR2), and optional **flux3** ~17.75 GB. Prepare one pack per copy/session. Packs contain real files, verified SHA256 values and source URLs. No model files pass through your PC.

For the H3 saved-output approach, set `AUTO_CREATE_DATASET = False`, wait for **PACK COMPLETE**, then **Save Version with output files**. Attach that saved notebook output through **Add Input → Notebook Output Files**. A draft alone is not durable storage. Upload failures retain the verified files for retry or saving; existing Datasets are never changed or overwritten. Automatic upload detects your Kaggle username from authentication, or uses `DATASET_OWNER` if set. Credentials are not embedded in this notebook.

FLUX3 requires accepted Hugging Face terms and **HF_TOKEN** in the builder's Kaggle Secrets. The face/standard packs need no Hugging Face token. Complete attached weights let the renderer work without a model-host token. Software setup and the studio tunnel still need Internet.

**Face-model terms:** InsightFace pretrained weights are restricted to non-commercial research. The renderer uses the pinned buffalo_l archive and InSwapper 128 ONNX model. Use photos you own or have permission to use.

Local validation includes a real CPU face-swap test, ComfyUI graph/API checks, synthetic model-pack tests and notebook compilation. Full Kaggle pack building, real Dataset submission and T4 rendering remain unverified here.
