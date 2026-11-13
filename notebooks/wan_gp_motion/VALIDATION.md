# Validation record — 13 November 2026

This records what was executed. A simulated GPU identity is not evidence of GPU computation.

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
| Actual Kaggle T4 H3 render | **Not executed:** this development executor has no CUDA GPU / `nvidia-smi`. Memory fit, render time, numerical behavior under full model load, frame output and camera fidelity remain unmeasured. The shipped runtime probes actual T4 kernels/imports before opening the GUI. |

## Reproduction

```bash
python notebooks/wan_gp_motion/build.py
python -m unittest discover -s tests -p test_wangp_motion.py -v
```

For native CPU integration, use a disposable checkout of WanGP at the pinned commit, CPU PyTorch 2.10.0 and the locked application dependencies. The harness applies the kit's source patches to that checkout:

```bash
python tests/wan_gp_motion_native.py \
  --app /path/to/disposable/WanGP \
  --work /path/to/scratch/native-check \
  --guide /path/to/JJP_orbit_124_0001-0124.mp4
```

Add `--serve 7871` to exercise the real GUI with simulated hardware metadata. This harness deliberately requires a CPU-only environment and does not exist in the notebook payload. Actual T4 validation must use the notebook's unsimulated `gpu_probe.py` and requested H3 renders.
