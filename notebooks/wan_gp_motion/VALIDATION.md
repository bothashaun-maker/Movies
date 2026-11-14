# Validation record — 13 November 2026

This records what was executed. A simulated GPU identity is not evidence of GPU computation.

## Black output / safe FP16 correction — 14 November 2026

Inspected the user's `Test.mp4` with FFprobe and FFmpeg signal statistics: valid H.264/yuv420p, 864×480, 24 fps, 124 decoded frames / 5.166667 s of video, plus stereo AAC. Every video frame has YMIN=YMAX=16, U=V=128 and zero frame difference: the encoded picture is entirely black, independently of the GUI/player. The subsequent `wangp (1).log` shows a successful GUI restart after the state repair, native model/LoRA/encoder/VAE loading, 56 s prompt encoding, and 8 denoising steps in 650 s. An `invalid value encountered in cast` warning occurs at the latent preview after step 1. The run exports an MP4 despite invalid intermediate numbers. This is evidence of a real but failed T4 inference attempt, not a successful video render or a decoder-only failure. The log does not identify the first internal operation producing a NaN.

[WanGP issue 2156](https://github.com/deepbeepmeep/WanGP/issues/2156) documents the same raw-FP16/pre-Ampere failure with Ref2VA pruned rank-8 INT8: condition projection, residual streams and branch output projections exceed 65504. Its native adaptation proposal and [Amduraznak's MIT reference implementation](https://github.com/Amduraznak/minimax-h3-fp16-fix) were read directly. The previous kit changed the dtype without these protections; that adaptation was incomplete. External reported GPU results are not kit benchmarks.

SAFE-FP16-v1.5 installs the adapted native path before model construction and MMGP loading. Condition projection and residuals/norms stay FP32; normalized Attention/MLP branch inputs and large branch weights stay FP16. SwiGLU pointwise math is FP32. Native linear/LoRA calls use power-of-two scaling (64 for Attention out_proj; 256 for MLP fc2), recovering the gain during FP32 accumulation in row chunks of at most 512. This avoids full-size FP32 branch buffers and retains destructive list handoffs. The selected model has bias-free branch projections. Native PDD heads, schedule and reference packing remain in place. BF16 model construction retains the original path. The same Dataset/catalog and package locks are reused.

Stage-specific finite checks stop invalid Qwen conditioning, reference VAE encoding, denoising predictions/updates and decoded RGB before integer conversion/export. They write `numerical-error.json`; the existing guard writes the per-shot error and releases the model. They do not sanitize NaNs into zeros or silently retry. The actual GPU startup probe now includes a tiny real FP16 overflow control and corrected projection comparison. It does not load H3 weights.

Executed eight CPU numerical regressions with actual FP16/FP32 arithmetic: raw projection/SwiGLU overflow controls, corrected FP32 results, segmented gates/bias, NaN/±Inf rejection/receipt, native miniature condition projection reaching 96,000, native FP32 residuals over 65,504, native MMGP safetensors loading preserving dtype locks, BF16-path retention, and complete miniature transformer forward passes with image/video reference packing and native PDD heads. Miniature weights are synthetic, not the production checkpoint. Only native import-time GPU metadata is simulated; tensor calculations execute on CPU. These checks establish arithmetic and native API compatibility, not corrected T4 memory fit, speed, conditioning quality or successful full-size output. FP32 residuals use more activation memory; the existing resource/deadline limits remain mandatory.

The notebook also invalidates and removes its own cached runtime modules after extraction, so replacing the single cell in an existing kernel imports the updated launcher/patches. Cancel the prior launcher cell first. This avoids relying on a second repair cell or a model Dataset rebuild.

All **37 regression tests passed** in the full CPU video environment, including the native miniature arithmetic and MMGP checks. The standalone CPU builder environment passes its 28 applicable checks (nine video-only checks skipped). The native A/B validation/workspace/download dispatch harness also passes with safe-FP16 installation asserted and zero production model loads. Both notebook payloads include the current numerical code/probe/launcher. The same-kernel extraction regression verifies eviction of the old owned launcher while retaining unrelated modules.

## Generate dispatch correction — 14 November 2026

The next user-supplied `wangp.log` confirms the GUI starts. Both Generate attempts then fail at `workspace_storage.py` while reading `kwargs['state']`. The kit's guard had rebound the call using `inspect.signature.bind` and forwarded `bound.args` / `bound.kwargs`. Python moves supplied keyword arguments into positional slots in `bound.args`; that breaks WanGP's outer workspace decorator, which explicitly requires `state` in its keyword dictionary. This failure is in the kit's call adapter, before model inference.

Reproduced the exact `KeyError: 'state'` in two CPU regressions before fixing the guard. GENERATE-FIX-v1.4 preserves the original argument layout and changes only the output filename, including when the filename was supplied positionally. Validation, unique filenames, job receipts, model release and watchdog behavior remain in place. Both shipped notebooks embed the corrected guard; the GPU notebook displays the new build marker. The CPU builder remains CPU-BUILDER-v1.2 and the 13-file model catalog is unchanged.

The expanded native CPU harness executes the real pinned `workspace_generation` and `generation_downloads` decorator chain for both A and B, using the actual `generate_media` signature and full shot validation. A sentinel replaces only the final generation function, and model release is mocked. Both calls retain the same `state` object, receive the appropriate unique output filename and complete without loading models. **28 regression tests passed** in the full CPU video environment. These checks establish corrected dispatch and artifact packaging; actual T4 H3 rendering, memory fit and motion fidelity remain unverified. No new Dataset build is required.

## GUI startup correction — 14 November 2026

The user's first T4 startup advanced through local model staging, package installation/checks and the CUDA/kernel/import probe. The standalone WanGP server then exited while importing Matplotlib through `models.wan.scail.scail_pose_draw`. Kaggle had supplied `MPLBACKEND=module://matplotlib_inline.backend_inline`, but the isolated server environment does not contain that notebook-only backend. This is a process-environment integration failure, before GUI readiness and before a user-requested render.

Reproduced the **exact same Matplotlib ValueError** locally with the supplied environment value and the installed video dependencies, using CPU only. GUI-STARTUP-v1.3 now sets `MPLBACKEND=Agg` in a copied child environment and preserves the notebook kernel's environment. The runtime probe checks the effective Matplotlib backend before GPU checks. A regression test imports Matplotlib and draws with Agg in an isolated child process while its supplied parent environment contains the failing Kaggle value.

The native CPU GUI harness now uses the launcher's complete command definition, including memory budgets, allocator choices and GPU argument, rather than a reduced set of flags. With the failing Kaggle backend value supplied to the harness, the corrected child reached native `/config` readiness and opened in Chromium. Hardware metadata is still explicitly simulated; CUDA-only startup paths and video inference are not established by this CPU test. **25 regression tests passed** in the video test environment. Startup failures now also print the real log tail and process exit code in notebook output. The model catalog and saved Dataset are unchanged.

## CPU builder correction — 14 November 2026

The user's Kaggle run exposed a missing `six` dependency before any model transfer. Reproduced the exact failure in a new Python 3.11.13 environment containing only the original CPU lock. The previous SDK inspection had run inside the much larger video environment, which already included `six`; that was an insufficient check of the CPU builder's package environment.

Compared `notebooks/h3_next/dataset.py` and the generated H3 Next model-pack notebook directly. The user confirms that Dataset builder works. It uses the notebook's own Python/Kaggle packages. CPU-BUILDER-v1.2 now uses that same setup, with an explicit import preflight and a same-interpreter `six` repair only if missing. The working H3 Next notebook/source is unchanged. The GPU application still uses its separate pinned environment.

Standalone CPU requirements now explicitly include `six==1.17.0`. Installed only those 34 CPU packages in a second fresh environment; the actual Kaggle API and generated `UploadFile.to_dict()` imported/executed successfully. New SDK integration tests exercise real `dataset_create_new`, `upload_files`, filesystem symlink handling, `UploadFile`, native file-token conversion and private-request construction. Only remote status/blob/create operations are stubbed, with socket access forbidden and dummy credentials. Test both success and rejected creation, source retention, absence of a false success receipt, and reuse of existing SDK packages without reinstalling them. These tests do not establish a live authenticated upload.

The model allowlist remains exactly the 13 files in `assets.json` for WanGP Ref2VA Pruned PDD8; tests reject adding H3 Next's model families. A notebook-payload regression check verifies that the actual delivered notebook contains the corrected uploader, dependency probe, CPU lock and unchanged model catalog.

After this correction, **22 tests passed** in the standalone CPU dependency environment, including four real-SDK integration checks. Both regenerated notebooks passed nbformat/schema and Python compilation checks. The same-interpreter dependency probe also completed successfully in the environment that originally reproduced the missing-`six` failure. No live Dataset upload or GPU render is claimed from these checks.

| Check | Result and boundary |
| --- | --- |
| Existing Movies notebooks | All 45 existing `.ipynb` files indexed/scanned at `17a61d90de0958675ff3e69b78a8d007d818d663`, including aliases. Their hashes remain unchanged. Relevant runtime/storage/upload sources reviewed directly. |
| Official model inventory | 13 exact files, 43,520,765,195 bytes / 40.531871 GiB. Large-file sizes and SHA256 values from Hugging Face blob metadata at `fe905e15831c620e804ba9e9382e4d9b4f60b21e`; six small config/tokenizer files downloaded and hashed. Large model bytes were not downloaded in this development environment. |
| Application dependencies | Full adapted upstream requirements installed successfully in isolated Python 3.11.13 with CPU PyTorch 2.10.0; package compatibility check passed. GPU lock pins Triton 3.6.0 to match CUDA PyTorch 2.10.0 rather than the CPU environment's independently resolved Triton. |
| CUDA installation plan | Official `cu128` index resolved exact Torch 2.10.0, TorchVision 0.25.0 and TorchAudio 2.10.0 with their CUDA 12.8 dependencies in a dry run. GPU packages were not installed/executed here. |
| Pinned Python bootstrap | Exact uv wheel downloaded, SHA256 verified and binary extracted; isolated Python 3.11.13 environment created successfully using the shipped bootstrap. |
| Native H3 validation | Real pinned WanGP imported with CPU Torch and explicitly simulated T4 **metadata only**. Native baseline `I` and guided `IV-U` settings both validated; actual compute-dtype selection is FP16; requested resolution is locked; no model loaded. Reproducible harness: `tests/wan_gp_motion_native.py`. |
| Native GUI / Chromium | Real WanGP Gradio UI opened in Chromium with the same metadata-only CPU harness. Confirmed FP16 header, 864×480 as the available resolution, FG one-prompt mode, correct encoder/VAE selections, and “No Model is Loaded”. No render button was used. |
| CPU regression tests | 17 tests passed: missing/corrupt models, disk-space gates, local-cache reuse, nested staging, input preservation, owned-process deadlines, child failure logs, sustained low-RAM stop, A/B equivalence, invalid shot/media rejection, private streaming-upload mock, and no success receipt after upload failure. |
| Notebook artifacts | Both notebooks pass nbformat schema validation; each has one executable Python cell that compiles. No outputs, credentials or installed-environment paths are embedded. Rebuild is deterministic. F821 undefined-name check passed. |
| Blender guide | Script executed with Blender 4.3.2, Cycles CPU. Output inspected with FFprobe and the kit's own media validator: silent H.264 / yuv420p, 864×480, exactly 124 decoded frames, 24/1 fps, 5.166667 s. Camera coordinates bake the specified −15° to +15° arc. Visual inspection caught coplanar overlapping cube faces at the cross joint; the final script uses one watertight extruded cross with the same dimensions. |
| Cloudflare binary | Pinned 2026.10.7 binary downloaded, exact byte count and SHA256 verified, `--version` executed. |
| Live Cloudflare tunnel | **Blocked in this development environment:** quick-tunnel request to `https://api.trycloudflare.com/tunnel` failed with TCP connection refused at `104.16.231.132:443`. This does not establish Kaggle connectivity. The launcher requires a registered edge connection and a successful public `/config` response before printing a ready URL; it stops after the deadline otherwise. |
| Live private Dataset creation | **Not executed:** no user's Kaggle API credentials in this environment. Installed Kaggle 2.2.5 uploader/create method signatures were inspected; streaming behavior tested with a mock. A mock submission is not a saved Dataset. |
| Actual Kaggle T4 H3 render | User subsequently completed a failed real eight-step render: 650 s denoising, invalid preview after step 1, 124 black frames. See the SAFE-FP16-v1.5 correction above. This executor has no CUDA GPU; corrected full-size output, fit, speed and camera fidelity remain unverified. |

## Reproduction

```bash
python notebooks/wan_gp_motion/build.py
python -m unittest discover -s tests -p 'test_wangp_motion*.py' -v
```

For native CPU integration, use a disposable checkout of WanGP at the pinned commit, CPU PyTorch 2.10.0 and the locked application dependencies. The harness applies the kit's source patches to that checkout:

```bash
python tests/wan_gp_motion_native.py \
  --app /path/to/disposable/WanGP \
  --work /path/to/scratch/native-check \
  --guide /path/to/JJP_orbit_124_0001-0124.mp4
```

Add `--serve 7871` to exercise the real GUI with simulated hardware metadata. This harness deliberately requires a CPU-only environment and does not exist in the notebook payload. Actual T4 validation must use the notebook's unsimulated `gpu_probe.py` and requested H3 renders.

To include native miniature-model numerical regressions in the CPU video environment:

```bash
JJP_WANGP_TEST_APP=/path/to/disposable/WanGP python -m unittest discover -s tests -p 'test_wangp_motion*.py' -v
```
