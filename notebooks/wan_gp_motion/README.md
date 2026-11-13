# JJP WanGP Motion Test Kit

The two new notebooks are separate from Director, H3 Next, Image Studio and the Movie Upscaler. Existing notebooks are unchanged.

1. Import [`JJP_WanGP_Motion_Models.ipynb`](../../JJP_WanGP_Motion_Models.ipynb) into Kaggle. **CPU / Accelerator None, Internet ON**. Enable `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`, in Kaggle Secrets. Set `DATASET_OWNER` if needed. Run its single cell; it automatically submits **one private Dataset**.
2. Wait for Dataset processing. Import [`JJP_WanGP_Motion_Test_Kit.ipynb`](../../JJP_WanGP_Motion_Test_Kit.ipynb), choose **T4 ×2 / Internet ON**, and attach that **one Dataset** through Add Input → Your Datasets. Run its single cell and open the printed Cloudflare URL.
3. In native WanGP, upload one appearance still. Generate the baseline with seed `8675309`, **864×480 / 124 frames / 24 fps / PDD 8 steps / Euler**. The factory prompt describes the cross orbit; replace it for another scene if desired, keeping the same prompt for both tests.
4. Keep the image and all generation settings. Select **Use One Reference Video** and upload the silent Blender guide. Generate the guided comparison. Use Reference Video, rather than In-Context Guide, denoising or ControlNet. A/B output names are assigned automatically and include a unique shot ID.
5. Inspect camera movement, appearance, geometry, proxy leakage, runtime and GPU memory. Save the clips/settings and **Save Version with outputs before ending the session**. The output root is `/kaggle/working/jjp-wangp-motion`. Mute native H3 audio in Movavi. The song master stays outside H3.

There is no ComfyUI server, canvas or extension in this notebook. WanGP depends on the `comfy-kitchen` Python math library upstream; that dependency does not launch ComfyUI.

## Complete model inventory

All files come from `DeepBeepMeep/MiniMax-H3` at revision `fe905e15831c620e804ba9e9382e4d9b4f60b21e`. Exact byte sizes and SHA256 values are in [assets.json](assets.json). The Dataset is flat; the studio reconstructs WanGP's required directories while staging verified local files.

| Component | Exact file(s) | GiB |
| --- | --- | ---: |
| Ref2VA pruned rank-8 INT8 | `MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors` | 19.612 |
| Qwen3-VL layer-50 GGUF Q4_K_M | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 13.576 |
| Ref2VA PDD heads | `minimax_h3_ref2va_pdd_8step_heads_bf16.safetensors` | 0.041 |
| Ref2VA PDD LoRA | `minimax_h3_ref2va_pdd_8step_lora_bf16.safetensors` | 1.237 |
| Original FP16 video VAE | `MiniMax-H3-video_vae_fp16.safetensors` | 4.850 |
| FP32 audio VAE | `MiniMax-H3-audio_vae_fp32.safetensors` | 0.564 |
| Native latent upscaler | `minimax_h3_latent_upscaler_3d_bf16.safetensors` | 0.643 |
| Qwen configuration and tokenizer | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `vocab.json`, `chat_template.json` | 0.009 |
| **Total: 13 files, ONE Dataset** | | **40.532** |

The upscaler is an eagerly loaded WanGP support dependency, even for this one-phase PDD job. It does not enable an extra output-upscaling phase. Audio VAE remains required because H3 jointly generates video and audio. FL2VA, ControlNet, Molmo, Sol, other accelerators and Image Studio weights are not required. Appearance stills and Blender guides are ordinary GUI uploads.

The older Director/H3 Next packs have different files and are not interchangeable with this Dataset. GGUF Q4 reduces the encoder's host-memory footprint; quantization can affect prompt/reference interpretation. The distilled 4B image encoder is deliberately not used for this motion-reference experiment.

## T4 and storage behavior

WanGP is pinned at `38758d3c73366422efcf609abb63cab8385f3552`. Runtime: isolated **Python 3.11.13**, **PyTorch 2.10.0 / CUDA 12.8**, full resolved application dependencies in [requirements.lock](requirements.lock). Driver **570.26+**, T4 SM75, ≥13,000 MiB free on GPU 0, ≥24 GiB host RAM / ≥20 GiB available are checked. GPU 1 stays unused; no tensor parallelism or pooled 32 GiB memory is implied.

Generation uses Profile 5 streaming offload, FP16, SDPA, zero manual preload, a 14 GiB emulated VRAM budget, conservative VAE tiling, PyTorch INT8/FP8 fallback math and no compilation or CUDA graphs. The global `H3_ALIGNED_CONTROL_VIDEO=False` is unchanged. Reference videos remain contextual guidance; frame-exact camera control is unproven.

The uploaded guide is validated at 864×480, then WanGP's native reference-size control reduces its conditioning frames to **288×160**. This keeps the exact 9:5 aspect ratio and 32-pixel alignment, with one ninth the reference pixels, while output remains 864×480. The native dimension helper is exercised in the CPU integration harness. There is no transplanted guide encoder or full-resolution reference track hidden in the default.

Two small pinned-source patches surround native generation with validation/job receipts, force FP16 in H3's model definition (which otherwise overrides `--fp16`), report the true compute dtype in the header, and expose only this preset's 864×480 output size. No H3 conditioning/sampling algorithm is transplanted. The VAE selector still says **BF16** because that is WanGP's identifier for the original VAE file; the selected checkpoint itself is FP16. `--lock-model` is avoided because it crashes this pinned native GUI; validation rejects other model choices before loading them.

All weights are staged locally before GUI launch to avoid Dataset-mounted checkpoint reads, a documented failure in Director. Initial storage gate: **52.54 GiB free on one writable local filesystem** (40.53 GiB models + 8 GiB runtime allowance + 4 GiB reserve). Existing local model files count toward this requirement and are reverified. The code measures `/kaggle/temp`, `/tmp` and `/kaggle/working`; it does not assume a Kaggle quota. A 19.5 GiB-only writable runtime cannot hold this local mirror. It stops clearly before model load in that case; a different notebook cannot create missing disk capacity.

The CPU Dataset maker needs room for only the largest missing file, **19.61 GiB**, plus a 2 GiB CPU-runtime/reserve allowance. It uses the previously working Image Studio native streaming upload method, with no preflight `datasets.get/status` permission requirement, no second pack copy and no zip of large models. Only successfully uploaded files in its own download cache are removed. Existing inputs remain untouched. Live upload requires the user's Kaggle credentials and is not claimed from mocked tests.

## Deadlines and retained diagnostics

Each setup operation runs in an owned subprocess with a finite deadline. Local model staging: 20 min; Torch install: 20 min; app dependencies: 30 min; GUI readiness: 5 min; tunnel startup: 2 min. Setup prints a heartbeat and log location every 30 seconds. System prerequisites (FFmpeg/ffprobe, PortAudio, OpenGL library and a compiler when missing) are installed in bounded stages if the Kaggle image allows it.

A requested shot has a **20-minute absolute deadline**, including model load. The supervisor stops the owned WanGP process group on expiry, sustained available RAM under 1.5 GiB, insufficient disk or lost server/tunnel. Stopping the cell also stops owned children. Native models are released after each shot; the GUI remains. These controls bound ordinary process failures. They cannot guarantee recovery from an OS/driver fault that prevents process termination.

`shots/<id>/settings.json` includes the actual generation inputs and media hashes; `result.json` or `error.json` records native status. `resources.jsonl` records host memory and GPU use/temperature every 30 seconds. Native MP4/settings sidecars, `wangp.log`, setup logs, startup receipt and watchdog-stop reason remain in the output directory. Completed outputs are preserved; an interrupted render is not silently rerun or described as complete. Settings are diagnostic receipts; use native Export Settings / Save Queue to preserve uploaded media for replay.

## Blender

Run `blender --background --python render_blender_guide.py` on your PC, or download the [ready Blender guide MP4](examples/JJP_orbit_124_0001-0124.mp4). [The script](render_blender_guide.py) exports 124 silent H.264 frames at 864×480 / 24 fps, a centered −15° to +15° orbit at 8 m / fixed 50 mm lens. It bakes locations at every rendered frame and avoids legacy `action.fcurves`. One solid mesh avoids intersecting coplanar faces at the cross joint. Output: `director_guide/JJP_orbit_124_0001-0124.mp4` relative to the working folder. Blender 4.3.2 CPU export was executed during validation.

## Build and checks

```bash
python notebooks/wan_gp_motion/build.py
python -m unittest discover -s tests -p test_wangp_motion.py -v
```

[VALIDATION.md](VALIDATION.md) distinguishes completed CPU/browser/media checks from GPU/account-dependent work. [notebook_review.json](notebook_review.json) inventories and hashes all 45 pre-existing notebook files, including aliases, at Movies revision `17a61d90de0958675ff3e69b78a8d007d818d663`. Relevant runtime sources reviewed include phased Director, the original dual-T4 worker boundaries, H3 Next storage, Image Studio's native Dataset saver, and Movie Upscaler's persistent Cloudflare startup. Existing notebooks are byte-for-byte unchanged.
