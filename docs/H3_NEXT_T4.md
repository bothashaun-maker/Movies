# H3 Next: new experimental T4 notebook

[Open the rendering notebook](../H3_Next_T4.ipynb) · [Open the model-pack notebook](../H3_Next_Model_Packs.ipynb)

This is a separate notebook. Your Current Working, Pixa, Director and Three-Step notebooks remain unchanged. H3 Next contains three full, editable ComfyUI workflows:

| ComfyUI workflow menu | Intended use |
| --- | --- |
| Singularity - Pika 4 Step Final | Start here; four-step generation |
| Singularity - Pika 2 Step Draft | Two-step comparison; expect a quality tradeoff |
| DenoMix - 8 Step Quality | Eight-step comparison using a different base model |

These names describe presets, not measured quality or speed guarantees. Actual T4 rendering of this new combination remains unverified.

## Start rendering

1. Open Kaggle and create a **New Notebook**.
2. Choose **File → Import Notebook → GitHub/URL**. Paste `https://github.com/bothashaun-maker/Movies/blob/main/H3_Next_T4.ipynb`.
3. Set **Accelerator: GPU T4 ×2** and **Internet: ON**. Stop your older ComfyUI session first.
4. If you have saved the new model packs below, use **Add Input → Your Datasets** to attach them. Otherwise, the required files download automatically.
5. Press the Play button on the notebook's single code cell. Wait for the ComfyUI Cloudflare link and open it. Keep Kaggle running while you use ComfyUI.
6. Choose a workflow from ComfyUI's **Workflows** menu. Edit the prompt in **Prompt / size / seconds / optional reference**, then press **Run**.

For an image reference, upload your image in **Upload reference image**. Connect its **IMAGE** output to **reference_image** on the prompt node. Leave that connection empty for text-to-video. Describe the reference as `<Picture 1>` in your prompt.

Change clip length using **seconds** in the prompt node. The default is five seconds at 512 × 768; the T4 guard limits size to 0.43 megapixels and duration to five seconds. Dimensions must be divisible by 32. Keep the motion dial at **0** initially; other values apply the optional motion adapter.

The resulting video appears at the Save MP4 node and in ComfyUI's output folder under `H3_Next`. There is no automatic generation or notebook video preview. A Cloudflare URL exists only while your notebook session runs.

## Save models once, reuse them

You need **four saved model packs shared by all three new workflows**, not one dataset per workflow. Together they use about **62 GiB**. You may skip DenoMix's pack if you only use Singularity; selecting DenoMix later will download its missing checkpoint.

1. Create a new Kaggle notebook. Import `https://github.com/bothashaun-maker/Movies/blob/main/H3_Next_Model_Packs.ipynb`.
2. Name it **H3 Next Models - Singularity**. Set **Accelerator: None**, **Internet: ON**, and keep it private.
3. Leave `PACK = 'singularity'` and `AUTO_CREATE_DATASET = True` near the top of the code cell.
4. Open **Add-ons → Secrets**. Add your Kaggle API token under **KAGGLE_API_TOKEN** and enable it for this notebook. Do not paste the token into code. Set `DATASET_OWNER` to your Kaggle username if automatic detection does not work.
5. Run the single cell. It checks authentication, downloads and verifies the models, then automatically submits a **private Kaggle Dataset** and prints its link. Wait for Kaggle processing to finish. **No manual Save Version is required for this upload.**
6. Repeat with three more copies of the same preparation notebook:

| Notebook name | Set PACK to | Size |
| --- | --- | --- |
| H3 Next Models - DenoMix | `'denomix'` | 19.53 GiB |
| H3 Next Models - Molmo | `'molmo'` | 8.99 GiB |
| H3 Next Models - Support | `'support'` | 13.93 GiB |

You can start these preparation notebooks separately while earlier uploads are running, subject to Kaggle's available concurrent sessions. There is no GPU requirement for preparation.

7. In **H3 Next T4**, select **Add Input → Your Datasets** and attach those private Datasets. Run the rendering notebook normally.

Successful submission is recorded for same-session retries; existing remote Datasets are never overwritten. If a fresh session reports a duplicate name, check Your Datasets for the existing pack. Failed uploads retain verified local files. Optional fallback: set `AUTO_CREATE_DATASET=False`, then Save Version with output files and attach Notebook Outputs. A draft alone does not preserve local models. Live authenticated Kaggle submission remains unverified here. These are new packs: the older H3 packs contain shared reference/audio VAEs but do not contain Singularity, DenoMix, Molmo2, Pika or Sol. Runtime setup and hash verification still run on startup; reusing packs avoids downloading the weights again.

## Engineering details and validation

- Pinned ComfyUI v0.44.0 and the existing T4 tensor-parallel fork; Python 3.12.12, PyTorch 2.10 with CUDA 13. The CUDA runtime requires a compatible NVIDIA driver (580 or newer). CUDA and INT8 probes run before model downloads.
- Molmo2-H3 INT8 replaces the prompt encoder. Its BF16 tensors are converted in memory to FP16 for T4; INT8 weights and FP32 scales remain intact. SDPA attention is selected. This adaptation is experimental and has not been numerically validated on a T4. Non-finite conditioning is rejected.
- Text conditioning uses Molmo2. Image appearance uses native H3 reference latents through the original video VAE. This is not a SAM tracking or Director timeline implementation.
- Sol-H3 INT8 decodes the output video. Native H3 audio decoding is retained. The original VAE is kept for reference-image encoding.
- Preparation, sampling and decoding run in **three separate processes**. Molmo2 exits before the diffusion model loads; sampling exits before output decoding. This reduces simultaneous model residency but adds process startup time. Saved conditioning and latents support retries. Checkpoint identities include model hashes, input content and new extension code identity.
- Both diffusion GPUs use the existing tensor-parallel sampling path. Preparation and decoding do not promise equal utilization of both GPUs at every moment.
- Singularity/DenoMix checkpoint prefixes and scale shapes are adapted while reading. No second 20 GiB checkpoint copy is created. Weight paths, hashes and immutable revisions are embedded in the notebook and [asset manifest](../notebooks/h3_next/assets.json).
- No extra upscaling pass, sparse-attention patch or D-RoPE changes. Upscale separately in CapCut or the repository’s Movie Upscaler notebook. SCAIL motion-transfer workflows are not included in this text/reference-image experiment.

All three 18-node canvases loaded in actual ComfyUI Chromium without missing nodes or JavaScript errors. Actual exported API graphs passed ComfyUI validation for browser, preparation, sampling and decoding, with and without a reference image. Saved conditioning/reference tensors passed an exact round-trip test; native reference conditioning and duration geometry were exercised with lightweight encoder fixtures. These CPU/browser checks do **not** establish real GPU memory use, render time or output quality. No full model render was run here.

Regenerate with `python notebooks/build_h3_next.py`. Regression tests are in `tests/test_h3_next.py`; full runtime tests require the repository's pinned ComfyUI test dependencies. The generator does not rewrite older notebooks.

## Upstream sources

- [Molmo2-H3 ComfyUI integration](https://github.com/alisson-anjos/ComfyUI-Molmo2-H3)
- [Pika H3 Flex Turbo LoRA](https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA)
- [Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)
- [DenoMix](https://huggingface.co/Deno2026/MiniMax-H3-DenoMix)
- [H3 models including Sol](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Motion dials](https://github.com/matlowai/ComfyUI-Dials)
