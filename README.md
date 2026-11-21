# JesusJuiceProductions — Movies

Self-contained **Kaggle** notebooks for AI video generation, image creation, editing and upscaling. Every studio runs in the browser: import one notebook, attach one saved private model Dataset, run its single startup cell, and open the printed Cloudflare/Gradio link. No local software is required.

Weight preparation happens once, on CPU, into a private Kaggle Dataset. GPU studios then attach that same Dataset and never download models during a render.

## Start here

- **[Movie Creation Launch System](docs/movie_launch/JJP_MOVIE_LAUNCH_SYSTEM.md)** — the practical path from one 15-second scene to a finished episode: model packs, reference stills, upscaling, delivery, weekly rhythm. [Printable PDF](docs/movie_launch/JJP_MOVIE_LAUNCH_SYSTEM.pdf) · [Planning templates](docs/movie_launch/README.md)
- **[H3 Kaggle setup](H3_KAGGLE_SETUP.md)** — baseline runtime and step-by-step instructions.
- **[Notebook collections](docs/H3_NOTEBOOK_COLLECTIONS.md)** — the Current Working, Pixa, Director and Three-Step canvases.
- **[Image Studio](Image_Studio_README.md)** — normal image workflows.

## Studios

Each pair is a CPU model builder plus a GPU studio. Import the `*_Models` notebook first, keep its Dataset, then open the studio with **Accelerator: GPU T4 ×2** and **Internet ON**.

| Studio | Model builder (CPU) | Studio (T4 ×2) | Docs |
| --- | --- | --- | --- |
| AIQuest H3 — native camera | `JJP_H3_Camera_Models.ipynb` | `JJP_H3_Native_Camera.ipynb` | [notes](notebooks/h3_camera/README.md) |
| H3 Next | `H3_Next_Model_Packs.ipynb` | `H3_Next_T4.ipynb` | [notes](docs/H3_NEXT_T4.md) |
| H3 collections (Pixa, Director, Three-Step) | — | `Pixa_5_Workflows.ipynb`, `Director.ipynb`, `Three_Step_Workflow.ipynb`, `Current_Working_Workflow.ipynb` | [notes](docs/H3_NOTEBOOK_COLLECTIONS.md) |
| LTX-2.5 | `JJP_LTX25_Models.ipynb` | `JJP_LTX25_T4.ipynb` | [notes](notebooks/ltx25_native/README.md) |
| WanGP motion kit | `JJP_WanGP_Motion_Models.ipynb` | `JJP_WanGP_Motion_Test_Kit.ipynb` | [notes](notebooks/wan_gp_motion/README.md) |
| FlashVSR upscaler | `JJP_FlashVSR_Models.ipynb` | `JJP_FlashVSR_T4.ipynb` | [notes](notebooks/flashvsr_native/README.md) |
| Movie Upscaler v2 | `Movie_Upscaler_Model_Packs.ipynb` | `Movie_Upscaler_Kaggle_v2.ipynb` | [notes](notebooks/movie_upscaler/README.md) |
| Image Studio — normal | `Image_Studio_All_Models_Dataset_v2.ipynb` | `Image_Studio_Normal_v10.ipynb` | [notes](notebooks/image_studio_normal/README.md) |
| Image Studio — commercial | `Image_Studio_Commercial_Model_Packs_v3.ipynb` | `Image_Studio_Commercial_v9.ipynb` | [notes](notebooks/image_studio_commercial/README.md) |

The AIQuest-native editions keep AIQuest/TeamAIQ's own Gradio/Wan2GP engine and interface; they are not ComfyUI adaptations. The H3 collection notebooks install connected ComfyUI canvases and are the exception.

## A session, start to finish

1. **Build once.** Import a `*_Models` notebook, set **Accelerator: None** and **Internet ON**, enable the `KAGGLE_API_TOKEN` secret (or `KAGGLE_USERNAME` + `KAGGLE_KEY`), run the single cell and wait for **PRIVATE DATASET SUBMITTED**.
2. **Attach.** Open the studio notebook, attach that one Dataset, set **GPU T4 ×2** and **Internet ON**.
3. **Start.** Run the single startup cell and keep it running. Nothing renders automatically.
4. **Choose.** Open the link, select a canvas under **Workflows**, set the frame/duration controls and press **Run**.
5. **Save.** Downloads and generated clips stay in `/kaggle/working/…`; save them before the session ends. Dataset submission does not save renders.

GitHub changes do not update a notebook already imported into Kaggle — import the named notebook again to receive the current setup.

## Status and limits

- These are working research studios, not certified pipelines. T4 fit, speed and output quality are measured per notebook, not guaranteed, and every README states what was and was not verified.
- Defaults bound the intended workload (for the H3 collections: 430,000 pixels and 124 decoded frames). Exceeding a cap stops before model preparation.
- Dataset creation is private. No credentials appear in receipts, notebooks or manifests, and model files are hash-pinned to immutable revisions.
- GPU quota is real: read the notebook's own notes before repeating a known-failing acceleration test.

## Structure and rebuilds

`notebooks/<studio>/` holds the build sources, checks and documentation. The root `.ipynb` files are **generated** — edit the sources, then rebuild:

```sh
python notebooks/<studio>/build.py     # or notebooks/build_h3_collections.py
```

Generated notebooks, workflow canvases and bundled example media are marked in [`.gitattributes`](.gitattributes); they are committed complete and self-contained on purpose, so a notebook imports with no extra assets.

## Licence

MIT — see [LICENSE](LICENSE). Upstream engines, weights and canvases keep their own licences and credit.
