# Validation record — 13 November 2026

This records what was executed. A simulated GPU identity is not evidence of GPU computation.

## Runnable default restored — RENDER-v1.8

The user explicitly rejected the blocked v1.7 notebook and further experimental GPU runs. The notebook code cell and launcher now default to `sdpa`, reaching the rendering path that produced Test 2 without manual mode changes. The existing SDPA regression verifies no optional package install/probe is invoked. Model catalog, existing package locks, reference-matching prompt, numerical safeguards, settings and resource limits are unchanged. There is no new speed claim or full GPU rerender here. One self-contained notebook is delivered; repair cells are unnecessary.

## Real T4 Sage compiler failure / quota guard — 14 November 2026

The user's startup output explicitly reports a failed optional Sage probe and `H3 attention: sdpa`. Their attention-probe.log identifies `arith.extf` applied to an INT8 dot operand, followed by `TritonGPUAccelerateMatmul` / `PassManager::run failed`, in SageAttention 1.0.6's `_attn_fwd_inner`, targeting `cuda:75`. Quantization kernels compiled; the attention kernel did not. This is a compiler incompatibility, not a Dataset problem or measured Sage render. The earlier CPU application environment had Triton 3.9.0; package/import/arithmetic checks there did not exercise the GPU lock's exact 3.6.0 target compiler. That was an insufficient compatibility boundary.

Installed only the Triton 3.6.0 compiler into an isolated local target directory, without changing the working CPU video environment or the notebook stack. Offline compilation of the actual Sage 1.0.6 kernel with `GPUTarget('cuda', 75, 32)`, head dimension 128, 128×64 tiles, eight warps/four stages reproduced the exact `arith.extf` error. Four warps fails too. The regression is `tests/wan_gp_motion_compile.py`; it compiles on CPU and never launches a kernel or loads models. Native compiler stderr and result were inspected. A local experimental cast-to-FP16 dot compiled, but its generated SM75 PTX contained no Tensor Core MMA instructions. It is not a validated speed improvement and was **not shipped**. No Torch/Triton downgrade, custom kernel workaround or claimed acceleration is added.

SPEED-CHECK-v1.7 changes the public launcher to reject the known-bad `auto` speed request **before hardware checks, output/lock creation, model staging, package installation or probes**. It cannot silently continue into another slow render. Explicit `sdpa` passes this gate and keeps the known-working backend, math, prompt, Dataset and limits. The old optional probe helper remains available for future development; the notebook's public launcher does not invoke it on the blocked stack. No further GPU test is requested to rediscover this failure. This update prevents repeated wasted setup/render time; it does not provide a faster model.

The exact-target compiler regression passed by reproducing the expected failure, with no GPU execution. All 49 full CPU regression checks passed, including assertions that a blocked speed request makes no hardware/setup/probe calls and creates no launcher output/lock. The CPU builder environment passes 39 applicable checks (ten video checks skipped). These are failure-detection/cost-boundary checks, not successful acceleration tests.

## Non-black baseline, reference mismatch and optional speed — 14 November 2026

The user's Test 2 is a valid non-black H.264/AAC output: 864×480, 124 decoded video frames, 24 fps, 5.166667 s of video. Frames at 0/30/60/90/123 were inspected. The user confirms this was image-only A and supplied the original reference. The reference shows Jesus in front of the wood with his face/chest/body visible, a large left-side crucifix, Roman soldiers/mourners in the foreground and two distant crosses on the right. Test 2 obscures Jesus behind the upright, reframes the cross and omits most of the foreground people. This establishes full-size non-black output for the prior safe-FP16 path, not reference fidelity or Blender-guide control. The user reports about 15 minutes denoising; no new timing log accompanies this clip.

MOTION-SPEED-v1.6 retains the safe-FP16 path, all model/dependency locks and resource limits. Its default prompt now explicitly requests the complete reference scene, matching body-to-cross orientation and front-side camera motion with the left-side framing. The earlier prompt described a centered wooden cross/grass and omitted the crowd/Jesus; that was an inappropriate default for the supplied image. Prompt correction is not a geometric guarantee, and the existing MP4 is not modified. The original bare-cross Blender guide is not relabeled as a match for this scene.

The optional acceleration is SageAttention 1.0.6, selected for H3 only inside a scoped forward decorator; Qwen and VAE stay on SDPA. The scope restores both attention mode and value scaling on success or exceptions. The package wheel's exact SHA256 and URL are pinned in `sage-requirements.lock`. It is installed separately with `--no-deps --require-hashes` (90 s limit), without changing Torch/Triton or the main application lock. The exact install command completed in the CPU validation environment. That is not a GPU kernel test.

The runtime performs the GPU test in its own bounded child (120 s): native WanGP head-128 attention on actual SM75, ordinary and large V inputs, finite output and relative L2 error no greater than 0.05 against SDPA. Native Sage value scaling is enabled to avoid FP16 accumulation overflow. A passing receipt enables H3-local Sage; install errors, timeouts, missing/malformed receipts or failed numerical checks retain SDPA. A prior success receipt is reset before each attempt; user cancellation propagates. The probe does not establish full-sequence speed/quality. No Sage GPU computation or speedup was executed/measured in this CPU-only development environment. `SPEED_MODE='sdpa'` skips optional setup.

CPU regressions cover attention-state restoration, nesting/absent keys, H3-local dispatch, fixed setup deadlines, explicit dependency isolation, failed installation/probe fallback, stale/invalid receipts and cancellation. Native miniature numerical/PDD tests and A/B dispatch remain required. For a speed comparison, use the same updated image/prompt/seed/settings with `auto` versus `sdpa`; do not attribute the old-versus-new prompt difference solely to the backend.

All **47 regression checks passed** in the full CPU video environment, including the real native split-QKV attention accepting the probe's FP32 RoPE layout and the miniature PDD/reference/numerical tests. The CPU-only builder environment passed 37 applicable checks (ten video checks skipped). The native A/B settings/workspace/download harness passed with the H3-local decorator installed and no production models loaded. The runtime additionally probes the actual native split-QKV H3 Attention class on the T4, so a direct attention-kernel pass alone is insufficient to enable the accelerator. The pinned optional wheel installed successfully without dependency changes; the full CPU environment's package compatibility check passed. Both notebooks pass schema/embedded-source checks; the other 45 notebooks, working FP16 math, guard, model catalog, Dataset code and existing dependency locks are unchanged.

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
