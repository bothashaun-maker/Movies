# Kaggle notebook collections

Select **GPU T4 ×2**, enable Internet, then run the notebook's single executable cell. Open its Cloudflare workflow link and select a canvas under **Workflows**. No rendering starts automatically. Keep Kaggle running; outputs appear in ComfyUI. Use one notebook setup at a time in a session.

| Notebook | Installed workflows |
| --- | --- |
| [Current Working Workflow](../Current_Working_Workflow.ipynb) | Existing connected four-step MiniMax H3 graph |
| [Pixa — 5 Workflows](../Pixa_5_Workflows.ipynb) | Pixa - Text to Video; Pixa - Image to Video; Pixa - First and Last Frame; Pixa - Speaking Audio; Pixa - Singing Audio |
| [Director](../Director.ipynb) | Director - Three Step; Director - PDD Eight Step |
| [Three-Step Workflow](../Three_Step_Workflow.ipynb) | Simpler TaoMate three-step first-frame graph with INT8 VAE |

GitHub changes do not update an already-imported Kaggle notebook. Import the named notebook again to receive the new setup.

## Included controls

Pixa preserves the creator's connected nodes, size/duration controls, image inputs and speech/singing audio synchronization. Upload your image in the visible image loader. Speaking Audio needs your recording in its audio loader. The uploaded archive's original speech WAV and BunnyMan JPEG were unreadable; intact portrait images and the singing example are included. The speech workflow is installed with an empty audio selection and reports a clear instruction before downloading weights if Run is pressed without audio.

Director preserves the real `MiniMaxH3DirectorCS` timeline, local prompts, reference toggle and image/audio/video controls. Both canvases default to five seconds at 512 × 768. Three Step uses TaoMate sampling; PDD uses the pinned author's Euler recipe and PDD sigmas. The separate Three-Step notebook retains the simpler existing graph. RTX upscaling is omitted as requested; upscale exported video in CapCut.

## T4 runtime and storage

The collection worker uses the existing pinned dual-T4 tensor-parallel runtime. Sampling saves joint latents and exits before decoding starts in a fresh process. Supplied audio is preserved in a separate waveform checkpoint, avoiding a second encoder/sampling run during export. The UI save node is lazy so it does not execute GPU dependencies in the browser server.

New collection preflight caps are 430,000 pixels and 124 decoded frames. The defaults are approximately 0.4 MP and five seconds. Increasing controls beyond those caps stops before model preparation. This bounds the intended workload; it does not guarantee a render will fit every memory condition.

The first canvas's weights are prepared during setup. Other canvases select their required pinned weights on Run, including Director's active reference branch. Switching FL2VA/Ref2VA can require a large new download. Files are reused and SHA-256 checked; attached Kaggle inputs are preferred. Space is measured rather than assuming a 50 GB temporary disk. If needed, only verified, notebook-owned inactive model caches are evicted. Inputs, rendered outputs and latent checkpoints are retained. A later switch may redownload evicted weights. Completed jobs and existing latents are reused without unnecessarily downloading sampling weights.

## Sources and validation

Regenerate with `python notebooks/build_h3_collections.py`. Original uploaded JSON lives in `workflows/sources/`; adapted canvases live in `workflows/Pixa_5_Workflows/` and `workflows/Director/`. The generated notebooks embed their code, canvases and available example media.

Additional packages are pinned to Pixaroma `88260de112af70a25f82edcd6ee2d31dccff8f13`, Director `5846ce8c3dac33cbdc629c2e01606e2a740f563c`, and PDD `311a65dd53832d8a5f8177a9d5fb923c09e35a90`. PDD's compatibility inspection reads the original core module because the T4 precision wrapper replaces the inspected `forward` method; the supported audio-scale core remains required. Model revisions, sizes and hashes are recorded in `notebooks/h3_suite/assets.py` and the baseline catalog.

All 47 automated tests passed, including exact supplied-audio roundtrips, lazy UI execution, graph partitioning and restricted cache eviction. All five Pixa and both Director canvases loaded in the real pinned ComfyUI browser with no missing nodes or browser errors. Their exported browser, sampling and decoding graphs passed ComfyUI validation. Screenshots: [Director](director_collection.png), [Pixa](pixa_collection.png).

These are browser and CPU checks. The new collections have not been rendered on Kaggle T4s in this development environment; speed, quality, and end-to-end GPU compatibility remain unverified. Existing successful baseline Kaggle renders do not establish those results for the new workflows.
