> **Historical v1–v6 guidance. For commercial music-video images use [Commercial v7](../../Image_Studio_Commercial_v7.ipynb), the [revised playbook](../image_studio_commercial/JJP_PLAYBOOK.md) and [license audit](../image_studio_commercial/LICENSE_AUDIT.md). The earlier InsightFace pretrained face stack is research-only.**

# JJP v6 cast workflows

The original manual photo-swap graph is retained. v6 adds approved-reference library loading, a two-character duet graph, and JJP artwork/provenance export. See JJP_PLAYBOOK.md for the full image-production system. Existing face-weight restrictions and CPU integration apply; no new face models or commercial authorization are implied.

# v5 startup repair

The reported v4 exception listed `inswapper_128.onnx` and `buffalo_l.zip` because no exact-size face-model files were found in the scanned attached/cached locations. That trace does not establish whether a Dataset was built, uploaded or attached elsewhere. It does establish that requiring the separate face pack prevented the studio opening.

v5 prepares only missing face models before GPU initialization, using the same verified CPU builder already in the project. It reuses attached/cached files, stores real pack files in working output, and tries private Dataset submission through the existing authentication. Upload failure is reported and does not block this session once model preparation passes. Missing core-generation packs remain strict. Set AUTO_PREPARE_FACE_PACK=False for the old strict behavior.

Nine local repair tests use small synthetic face-model files and a real local HTTP server: the exact missing-pair scenario, one-file reuse, attached-file reuse, no face downloads when face swap is disabled, strict missing-core behavior, disabled auto-preparation, repeated local-pack reuse, nonblocking Dataset failure and generated notebook embedding. Seven existing pack tests also pass. No face-inference or workflow code changed. Live Kaggle/public-host download/upload remain unverified for this repair.

# Face Swap — Image Studio v4

A dedicated photo face-swap workflow is now available in the normal ComfyUI workflow menu. It uses InsightFace buffalo_l detection/recognition and InSwapper 128, with separate visible source detection, target detection, swap and PNG-save stages. It does not use a prompt to approximate someone's identity.

1. Open **Image_Studio_Model_Packs.ipynb** on Kaggle. It now starts with `PACK = 'faceswap'` and `AUTO_CREATE_DATASET = True`, using your existing Kaggle authentication. Run it on **Accelerator: None**, Internet enabled. It downloads and verifies about **843 MB** of face models, then submits a new private Dataset and prints its link. Actual API submission remains unverified from this development environment.
2. Attach the face Dataset to **Image_Studio_Kaggle_v5.ipynb**, alongside your existing standard pack. Your existing 11.1 GB standard pack is unchanged; do not rebuild it just to add face swap. Saved notebook outputs also work if you prefer that route or Dataset submission fails.
3. Run the v4 startup cell. Open the studio link and select **Image Studio - Face Swap**.
4. Upload the identity photo under **SOURCE** and the picture to change under **TARGET**. Leave both indices at 0 for a one-face photo; face indices start at 0, ordered from left to right in each image.
5. Press **Run**. The result is saved as PNG under `image-studio/images/ImageStudio/FaceSwap`. The target's canvas size is preserved. For enlargement, open an existing upscaling workflow and upload the swapped PNG.

Swapping runs on CPU through ONNX Runtime deliberately, keeping generation's T4 VRAM available and avoiding CUDA-provider version conflicts. No facial-restoration model is forced after swapping because it can change identity. InSwapper works with a 128px aligned face crop: a large output canvas does not imply native high-resolution facial detail. Large side angles, occlusion, expression differences and small faces can reduce likeness. Missing faces and out-of-range indices stop with an explicit error. Photo batches and video are not supported by this graph.

Only use photos you own or have permission to use. The InsightFace pretrained model terms restrict these weights to non-commercial research; the code license is different from the weight license. [Upstream InsightFace terms](https://github.com/deepinsight/insightface/tree/master/python-package) and [model distribution](https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models).

The face pack is a separate pinned catalog: `Gourieff/ReActor` dataset revision `079e0e459cd4261bbcba8c71d47df54a4ded7642`; both the ONNX swapper and buffalo_l archive are SHA256-verified. No InsightFace first-use model download is needed: startup installs the verified archive in its expected model directory. Only detection and recognition models are initialized. InsightFace 0.7.3 and ONNX Runtime 1.23.2 are installed in the studio environment, with the existing scientific/Torch constraints preserved. InsightFace builds a native extension; startup selects GCC/G++ when available and probes the import before completing setup. The entire custom node source is embedded in the single-cell notebook.

If you already have the face pack, set `ENABLE_FACESWAP = False` only when you deliberately want to omit face swapping. `REQUIRE_ATTACHED_MODELS = True` stays the default: no silent model redownload. The model-pack builder's automatic Dataset upload is enabled by default as requested; private credentials are not stored in the generated notebooks.

A real CPU integration test used public-domain NASA and US Navy portraits: both faces were detected, swapping returned finite pixels at the target dimensions and a changed output, and blank-image/out-of-range-index failures passed. This is an integration test, not a quantitative likeness benchmark. Full Kaggle setup/GPU generation and user-photo fidelity have not been measured. Existing generation settings and older notebook versions are retained.

Local validation also passed the actual ComfyUI HTTP queue → face detection → swap → PNG-save chain, all ten graphs loaded in Chromium as registered nodes, four graph/notebook regression checks passed, and the existing seven model-pack checks passed. This confirms local CPU integration, not live Kaggle startup or a likeness score. The CPU photo fixtures were NASA's public-domain astronaut image (scikit-image v0.25.2 sample) and the US Navy Grace Hopper portrait (Matplotlib v3.10.1 sample). Test scripts expect those fixture files and the verified ONNX/archive files under the development work directory.
