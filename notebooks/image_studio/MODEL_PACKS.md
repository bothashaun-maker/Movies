# Save weights once; reuse on Kaggle

Use **Image_Studio_Model_Packs.ipynb** to create reusable model files without a GPU. This saves repeated model downloads and startup time. Transfers go directly from the model hosts to Kaggle; the previous renderer also downloaded its weights directly to Kaggle, not through your home connection.

1. Import the builder into Kaggle, select **Accelerator: None**, enable Internet, leave `PACK = 'standard'`, and run its one code cell.
2. Wait for **PACK COMPLETE**, then **Save Version with output files**. Keep that version private. A draft alone does not preserve the weights.
3. Import **Image_Studio_Kaggle_v3.ipynb**. Choose **Add Input → Your Work / Notebook Output Files** and attach the saved builder version.
4. Enable Internet and T4 GPU, run the startup cell, and open the studio link.

The standard pack is approximately **11.1 GB**. This matches the user's uploaded **H3_Model_Packs Data Sets.ipynb**: model files are created and verified automatically, and saving a notebook version makes its outputs reusable. There is no need for a named Dataset, Kaggle API token, PC download or re-upload. A saved model pack does not back up generated images.

Optional named Dataset creation remains available with `AUTO_CREATE_DATASET = True`, requiring private Kaggle API authentication. New uploads explicitly request private visibility and never update existing datasets. It is disabled by default because the uploaded H3 notebook uses saved outputs instead.

For optional FLUX3, use a separate builder notebook with `PACK = 'flux3'`. That pack is approximately **17.75 GB**. Accept the Hugging Face model terms and enable Kaggle Secret **HF_TOKEN** in the builder. Attach its saved output to v3 and set `ENABLE_FLUX3 = True`. If you do not use SDXL, set `LOAD_SDXL = False`; set `ENABLE_SEEDVR2 = False` too if you are not attaching the standard pack. The FLUX3 pack contains its generator, text encoder, VAE and required prompt enhancer; RealESRGAN is in the standard pack and is always needed by the studio's enlargement workflows, so attach the standard pack as well. The v3 default keeps this straightforward by expecting the standard pack.

The renderer checks required filenames and sizes before installing packages or initializing its GPU. It verifies SHA256 before model use and links directly to `/kaggle/input`, avoiding another model copy. `REQUIRE_ATTACHED_MODELS = True` prevents missing weights from being silently downloaded. Set this to `False` only if you deliberately want the original online download behavior.

Only model weights are packaged. Software, ComfyUI source and the Cloudflare executable still install over Kaggle's connection, and opening the studio needs Internet. Source-image uploads and generated-image downloads still use your connection. This is not a fully offline software bundle.

The builder writes actual model files, not symlinks to another input. Its manifest records source URLs, expected sizes and SHA256 values and is marked complete only after all selected files pass verification. Existing valid files are reused. Public partial downloads support Range requests, restarting safely if the server ignores them. Existing corrupt completed files stop the build without being overwritten. The builder does not load a generation model or import Torch.

Seven local checks passed using small synthetic model files and a local HTTP server: completed packs and repeat reuse, partial-download restart, corruption preservation, copying attached files into durable outputs, shared-catalog/notebook compilation, explicit private Dataset creation with a mocked API, and failed-upload handling. Both generated notebooks pass v4 schema validation. The full 11.1 GB or 17.75 GB build, real Kaggle Dataset submission/processing/attachment, authenticated FLUX3 download, and GPU rendering remain unverified here. Earlier v2 graph/CPU-inference checks are unchanged; model weights and generation defaults have not changed.

Rebuild from repository root:

```sh
python notebooks/image_studio/build_assets.py
python notebooks/image_studio/build.py --output Image_Studio_Kaggle_v3.ipynb
python notebooks/image_studio/test_assets.py
```

The asset notebook builder writes beside its source by default; move the generated notebook to the repository root for publication. Keep generated images separately with **Save Version with outputs** or download them before ending a render session. A saved model pack does not back up your generated pictures.

The uploaded H3 model-pack notebook confirms the saved-output approach. The Music asset builder also has optional Dataset upload. This image builder defaults to the same H3 approach and includes optional private named Dataset creation through the official Kaggle API (`dataset_create_new(public=False)`). Authentication remains private and is never embedded into the generated notebook.
