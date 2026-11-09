# JJP v6 compatibility

Image Studio v6 adds the JJP creative production nodes and playbook with no new model weights. Reuse existing standard, face and optional FLUX3 packs. The v5 missing-face preparation fix and automatic private Dataset attempt are retained. See JJP_PLAYBOOK.md for approved-cast/output persistence; model packs do not back up your reference photos.

# v5 missing-face-pack repair

The renderer can now prepare the small face pack automatically if it is missing. Keep the existing standard pack attached and run v5 with Internet enabled. It verifies the face models, attempts private Dataset submission using existing authentication, and prints the link if submitted. If Dataset upload fails, verified face models remain usable for that session; retain the working outputs or retry saving them. Nothing is redownloaded when a valid attached/cached face pack is found.

AUTO_PREPARE_FACE_PACK=True enables this targeted repair. REQUIRE_ATTACHED_MODELS=True still protects the generation packs against unrequested downloads. The standalone CPU builder below remains available but is no longer a mandatory separate step for the missing face pack.

# Image Studio model packs

The builder now defaults to **PACK='faceswap'** (~843 MB), with **AUTO_CREATE_DATASET=True** using the user's existing Kaggle authentication. Run it on CPU with Internet enabled. After verification it requests a new private named Dataset and prints its link. Wait for Kaggle processing, then attach it to **Image Studio v4** alongside the existing standard pack. Existing Datasets are not updated or overwritten. Failed uploads retain verified files for retry; credentials are never included in notebook source.

The existing **standard** pack (~11.1 GB, SDXL/RealESRGAN/SeedVR2) is unchanged. Do not rebuild it just to add faces. Optional **flux3** is a separate ~17.75 GB pack and requires accepted Hugging Face terms and the HF_TOKEN secret during building. Build one pack per copy/session.

The uploaded H3 notebook's saved-output approach is also supported: set AUTO_CREATE_DATASET=False, run to PACK COMPLETE, then Save Version with output files. Add that saved output through Add Input → Your Work / Notebook Output Files. A draft alone is not durable model storage. No models need to pass through the user's computer.

The renderer detects matching attached weights, verifies SHA256 and uses symlinks to read the attached files rather than making another full copy. REQUIRE_ATTACHED_MODELS=True stops before model downloads if required files are absent. Internet remains necessary for software installation and Cloudflare. Picture uploads and generated-picture downloads still use the user's connection.

The face pack contains InSwapper 128 and the buffalo_l archive from pinned Gourieff/ReActor dataset revision 079e0e459cd4261bbcba8c71d47df54a4ded7642. See FACE_SWAP.md for architecture, model terms, tests and use. The InsightFace pretrained weights are restricted to non-commercial research. The builder itself neither loads these models for inference nor uses a GPU.

Seven synthetic/local pack checks cover repeat reuse, corruption preservation, partial-download restart, durable input copying, shared-catalog/notebook compilation, explicit private Dataset creation arguments and failed-upload reporting. These do not establish successful live Kaggle upload. The face workflow additionally passes real CPU model detection/swap checks and local ComfyUI/browser checks. Full Kaggle setup, Dataset creation/processing and T4 generation remain unverified.

Rebuild from repository root:

```sh
python notebooks/image_studio/build_assets.py
python notebooks/image_studio/build.py --output Image_Studio_Kaggle_v4.ipynb
python notebooks/image_studio/test_assets.py
```

The asset builder writes its notebook beside its source; copy it to the repository root for publication. The existing v1–v3 notebooks remain available. Save generated images independently: model packs do not back up output pictures.
