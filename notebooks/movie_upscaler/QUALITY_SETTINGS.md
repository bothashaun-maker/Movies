# T4 quality settings — reviewed 16 November 2026

Scope: the existing Movie Upscaler model Dataset and both native connected graphs. No AIQuest, H3 camera, generation notebook or model catalog change. The user explicitly authorized this upscaler rewrite and GitHub publication.

Sources reviewed at the exact shipped SeedVR2 revision `4490bd1f482e026674543386bb2a4d176da245b9`:

- [Upstream operating guidance](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/README.md)
- [Upscaler schema](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/video_upscaler.py)
- [DiT schema](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/dit_model_loader.py)
- [VAE schema](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/vae_model_loader.py)
- [Actual temporal batching and blending](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/core/generation_phases.py)
- [Official Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and pinned ComfyUI native `ImageUpscaleWithModel` / `ImageScaleBy` implementations.

## SeedVR2: every node

| Node / control | Shipped value | Reason |
|---|---|---|
| Upload / source_path | Filled by Upload clip | Browser file picker and progress, with no path copying required. Existing Kaggle paths remain usable. |
| Source / scale | 2 | A bounded enlargement, rather than assuming more pixels always improve fidelity. |
| Source / chunk_frames | 17 | Bounds host frame buffers. This is not the model's temporal batch size. |
| Source / context_frames | 4 | Reconstruct with neighbouring frames and crop context before saving, with no duplicate output frames. Does not guarantee invisible outer chunk seams. |
| Source / max_target_edge | 1920 | Explicit target guard. It is not certification that every aspect ratio/content fits. |
| Source / resume_tick | 0, advanced by continuation | Queue counter only, not a quality setting. |
| DiT / model | seedvr2_ema_3b-Q8_0.gguf | Existing pinned high-precision GGUF pack, not the more aggressive Q4. FP16/7B may have advantages but are not in this Dataset and have no measured T4 fit here. No unsupported 'best possible' claim. |
| DiT / device | cuda:0 | One inference GPU. Memory does not pool. |
| DiT / blocks_to_swap | 32 | All 3B blocks can offload, protecting activation headroom. Swapping affects residency/speed, not the selected model's mathematical precision. |
| DiT / swap_io_components | true | Offload embeddings/norm-related I/O components as supported upstream. |
| DiT / offload_device | cpu | Does not consume the VAE GPU's memory between phases. |
| DiT / cache_model | true | Reuse CPU-loaded model across chunks. |
| DiT / attention_mode | sdpa | Supported baseline on T4. No Sage 2/3 or Ampere-only Flash backend. |
| DiT / torch_compile_args | unconnected | No speculative compilation dependency in the default path. |
| VAE / model | ema_vae_fp16.safetensors | Existing FP16 VAE, with no FP8 substitution. |
| VAE / device | cuda:1 if present; otherwise cuda:0 | Use the second T4 for VAE phases without pretending memory is pooled. No speed claim. |
| VAE / encode_tiled, decode_tiled | true, true | Bound spatial VAE buffers. This does not spatially tile DiT attention. |
| VAE / encode_tile_size, decode_tile_size | 512, 512 | Keep existing conservative tile sizes. Larger tiles consume more memory. |
| VAE / encode_tile_overlap, decode_tile_overlap | 128, 128 (was 64) | Upstream recommended overlap provides more blending room against tile seams. It adds compute, not a guarantee of better every-frame quality. |
| VAE / tile_debug | false | No diagnostic tile overlays in output. |
| VAE / offload_device, cache_model | cpu, true | Reuse model on CPU between chunks. |
| VAE / torch_compile_args | unconnected | Preserve baseline execution. |
| Upscaler / seed, control_after_generate | 42, fixed | Reproducible settings across automatically queued chunks. |
| Upscaler / resolution | connected short edge × scale | Exact geometry. Do not use a separate arbitrary target. |
| Upscaler / max_resolution | 0 | Source node enforces the cap. Avoid a second resize that conflicts with committed geometry. |
| Upscaler / batch_size | 5 | The smallest normal temporal batch in the required 4n+1 family. Larger batches can improve temporal context, but have a higher unmeasured T4 activation peak. Do not default to 1, which sacrifices temporal processing. |
| Upscaler / uniform_batch_size | true | Pad a short final batch, then remove padding. This avoids inconsistent short-tail processing. |
| Upscaler / temporal_overlap | 2 | Upstream blends overlapping predictions. It must stay below batch_size, with no repeated output frames. |
| Upscaler / prepend_frames | 0 | Outer chunk context already supplies neighbouring frames. Avoid adding synthetic reversed frames to every chunk. |
| Upscaler / color_correction | lab | Upstream recommended perceptual colour matching to the source. Content-specific comparisons may favour another option. This is not universal proof. |
| Upscaler / input_noise_scale, latent_noise_scale | 0, 0 | Preserve source fidelity rather than injecting variation or softening fine structure. |
| Upscaler / offload_device | cpu | Intermediate tensor storage off the inference GPU. |
| Upscaler / enable_debug | true | Retain numerical/memory diagnostics, with no output-quality effect. |

## Real-ESRGAN: every node

Same upload node and source scale/cap/chunk count. `context_frames=0`: frame-independent inference receives no temporal benefit from repeating context.

| Node | Shipped values / reason |
|---|---|
| UpscaleModelLoader | RealESRGAN_x4plus.pth, unchanged verified official checkpoint. |
| ImageUpscaleWithModel | Actual native tiled model. No invented denoise/sharpen/face controls, as this node exposes none. |
| ImageScaleBy | Lanczos, with scale_by connected to source scale / 4. It is 0.5 for 2× and 1 for 4× automatically. No mismatched geometry or required second edit. |
| Commit | Same quality/export settings below. |

The model can flicker because it has no temporal model. A 4× output is not proof of four times the real detail.

## Export: every setting, both workflows

- `export_format=lossless_mkv` by default: FFV1 level 3 / BGR0, preserving the exact processed RGB8 pixels without YUV420 chroma loss or a lossy final encode. This is a master, not the most broadly playable format. Large file size and disk preflight are intentional.
- `high_quality_mp4`: libx264, `crf=14`, `preset=slow`, YUV420p for sharing/editor compatibility. Both CRF and preset are visible and editable. Slow is a compression-efficiency choice, not more model detail.
- CRF/preset are ignored in lossless mode. Export format/CRF/preset do not change restoration identity. Re-export uses the same verified chunks. Export receipts/filenames distinguish encodings.
- `auto_continue=true`. Frontend Auto Queue stays OFF. This changes queue management, not quality.
- Original audio is stream-copied. Unsupported MP4 audio falls back to AAC 192k. MKV also preserves supported subtitles. Frame count, rational fps, geometry and audio-stream count are checked.
- This pipeline decodes to RGB8 before model processing. A lossless final master does not restore lost HDR/10-bit precision. Non-finite model frames cause a clear failure before integer conversion.

## Validation boundary

Real streamed upload HTTP tests and browser file-picker tests validate user interaction and byte integrity. Real tiny Real-ESRGAN CPU inference validates graph execution, scaling, automatic continuation and export. Exact RGB comparisons validate lossless encoding. CPU schema/browser tests for SeedVR2 inspect actual pinned node definitions. They are not a SeedVR2 render or a T4 memory/speed/quality benchmark. No full-size T4 result is claimed.
