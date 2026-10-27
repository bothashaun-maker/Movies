# Uploaded Director workflows and T4 adaptation

Reviewed on 2026-10-27. JSON content was treated as workflow data, not instructions. Model URLs in the supplied files were cross-checked against publisher metadata.

## Delivered experimental sampler adaptation

[Experimental notebook](../minimax_h3_taomate3_experimental.ipynb) · [Canvas JSON](../workflows/minimax_h3_taomate3_experimental.json)

The TA three-step workflow selects:

- TaoMate FL2VA 3-step EMA LoRA, strength 1.0. Both its FL2VA and Ref2VA branches select that same file.
- `res_multistep` sampler.
- `ManualSigmas`: `1.0, 0.961165, 0.853333, 0.0`.
- INT8 ConvRot video VAE; FP32 audio VAE and the existing NVFP4 AWQ text encoder.
- Comfy Kitchen attention; KJNodes memory-reduction patches.

The adaptation carries the LoRA/sampler/sigma/VAE choices into the visible T4 graph. It removes the four-step sigma-shift node and its scheduler connection. It preserves tensor-parallel loading, isolated sampling/decode processes, Cloudflare access and first-frame image conditioning. KJNodes' single-device model patches are not stacked onto the tensor-parallel block implementation: their interaction has not been validated.

This is **not the full Director interface**. The supplied `MiniMaxH3DirectorCS` node manages timeline/reference conditioning, with additional dependencies and a different conditioning path. Multiple references, timeline editing and long-clip planning are not implemented in this adaptation. Existing short-clip resolution/frame limits remain enforced. The eight-step PDD and RTX upscaler files are reviewed references, not installed runnable workflows.

The baseline four-step notebook keeps its default model selection. The experimental notebook requests only the selected diffusion family and reuses the encoder/audio weights. New files total 5,292,062,608 bytes (about 4.93 GiB); existing old weights are not deleted. The storage planner accounts for those files before downloading.

## Verified model provenance

| File | Repository revision | Bytes | SHA-256 |
|---|---|---:|---|
| `experimental/minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors` | `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI@633e6c48cd05f7800e2cb2c007b5c9023b92c576` | 2480997424 | `6b119983683b265ad65224a9a643eea1714efe5b8bcb93799999635ccc455c73` |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6` | 2811065184 | `52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e` |

[LoRA publisher guidance](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/633e6c48cd05f7800e2cb2c007b5c9023b92c576/README.md) confirms the manual sigmas and labels TaoMate experimental. Its cross-family use is evidenced by the supplied creator workflow, not a measured T4 quality result. [Video VAE repository](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6/vae).

## Other supplied workflows

`MiniMax H3 Director - PDD8.json` uses matching PDD model nodes, eight-step PDD scheduling and `res_multistep`, with FP16 video VAE. It is an alternative sampling recipe, not evidence of acceleration over the existing four-step setup. Its Director/reference controls cannot be reproduced merely by connecting a first-frame image.

`RTX SR Upscaler Video.json` uses `RTXVideoSuperResolution` from `comfyui_nvidia_rtx_nodes`, targeting portrait 2160×3840 with ULTRA quality. VideoHelperSuite imports/exports the video and audio. It is a separate post-processing stage, adding work after generation; it does not speed up H3. The supplied workflow does not establish that its RTX SDK runtime supports Kaggle Linux/T4, and that compatibility has not been verified here.

## Validation and limits

41 checks pass, including real pinned-ComfyUI graph schema checks for both model families and text/image paths, selected-weight isolation, canvas-link integrity and preservation of the existing four-step configuration. These checks do not load the new multi-gigabyte model weights or execute GPU inference. Actual T4 speed, numerical compatibility, peak memory and perceptual quality remain unmeasured. CPU validation is not a GPU-render guarantee. The experimental canvas also opened automatically in the real ComfyUI browser with 16 nodes, 17 connections and no page errors; its browser-exported sampling/decode graphs validated for text and image inputs. [Browser screenshot](h3_taomate3_workflow.png).

For a controlled comparison, use the same image, prompt, seed, dimensions and frame count. `SAMPLING_PRESET='turbo4'` with `VIDEO_VAE='int8_convrot'` isolates the VAE change; `SAMPLING_PRESET='taomate3'` with `VIDEO_VAE='fp16'` isolates the sampler change. Changing both at once cannot attribute a speed or quality difference to either component. Semantic Bridge with TaoMate is blocked until validated.
