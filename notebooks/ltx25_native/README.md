# JJP LTX-2.5 native test notebooks

Two separate self-contained notebooks, using AIQUEST's native Gradio/Wan2GP engine and the existing JJP automatic private Dataset upload method. Existing H3, image and upscale notebooks are untouched.

## Run in this order

1. Import `JJP_LTX25_Models.ipynb` into Kaggle. Accelerator **None**, Internet **ON**. Enable the existing `KAGGLE_API_TOKEN` secret, or `KAGGLE_USERNAME` and `KAGGLE_KEY`. Run the one cell. It automatically submits one private Dataset. Wait for processing.
2. Import `JJP_LTX25_T4.ipynb`. Attach that **one complete Dataset**, enable **GPU T4 x2**, Internet **ON**, and run its one startup cell. Open the printed Cloudflare/Gradio link.
3. Upload a Start frame, describe one action and camera move, keep **480p preset / 5 s / 8 steps**, and press Generate. The interface also supports text-only and optional first/last frames. Nothing renders automatically at startup.

Use the images you already approved for composition. Start-frame input is first-frame conditioning, not a loose character reference. End-frame input controls the endpoint; it does not guarantee a perfect trajectory or identity in between. Keep important faces readable and plan exact edits in the editor. A single short shot is the most useful first test of your image and motion needs.

This edition generates its own audio. It does not take your existing song as an audio-conditioning input. Singing/lip-sync from supplied audio would use a separate audio-driven mode. No extra LoRA/model is added for that here. Use your song as the final master in the editor for this test.

## Dataset and storage

13 required files, **36.867 GiB** total, each with exact bytes, SHA-256 and an immutable Hugging Face revision. The CPU builder streams one verified file through Kaggle's native uploader at a time, requiring only the largest missing file (**18.145 GiB**) plus 2 GiB reserve locally. It reuses verified attached files and releases only its own uploaded temporary downloads. Dataset creation is private. No credentials are included in receipts, notebooks or manifests.

The GPU cell needs enough measured local scratch for the missing part of the complete set plus 6 GiB. It stages/hashes from the attached Dataset, preserving it, and does not download models. Models are not inferred directly from a potentially slow Dataset mount. Software/source setup still needs Internet. The native engine loads once and stays resident between jobs.

Keep the cell running. Native Stop cancels a job. Defaults allow 1200 seconds for engine startup and 1800 seconds per generation, independent of model staging/package setup. These are editable spending limits, not speed claims. Only the owned app/process tree is stopped. Completed files remain in `/kaggle/working/jjp-ltx25/outputs/`; save them before the session ends. Dataset submission does not save generated clips.

## Pinned integration and changes

- AIQUEST notebook revision `e297aa0fb6eb1c9eaa57c855fc1cf66657b4b66b`, Git blob `1fe403e80116e4b261a004e57b17c6a683f20d58`.
- Native Wan2GP revision `f02a38275fa43057ea75814e8ef4368e5530dcf2`.
- Model repository `DeepBeepMeep/LTX-2`, revision `1798f80b3340d2f87cffae42f9fa5367d0963d0c`.
- Retains the native tensor-parallel blocks, INT8 kernels, FP16 computation, FP32 numerical protections, reference preparation and decoding.
- Eight-step schedule only. Invalid image conditioning is never silently replaced with zeros. Invalid text features, denoising callbacks or decoded video/audio fail clearly.
- All nine upstream numerical patches must apply to the pinned source before model loading.
- Exact default wide-format text-to-video dimensions are **832×512**, not 480 pixels tall. An 864×480 image at that preset is prepared on a rounded **896×512** grid. Exact job geometry is logged. Higher presets exist but full-size T4 fit is not certified.
- Native MP4 export remains H.264 CRF 17 / veryfast with generated AAC audio. This is not an upscale notebook or a lossless restoration pipeline.

AIQUEST/TeamAIQ credit remains in the native interface and engine. Original model/source licences apply. [Original notebook](https://github.com/TeamAIQ/Colab-notebooks/blob/e297aa0fb6eb1c9eaa57c855fc1cf66657b4b66b/kaggle_notebooks/ltx-2-5-22b-distilled-video-generator.ipynb).

## Rebuild and local checks

From the source bundle's root:

```sh
python -m pip install nbformat 'kaggle==2.0.2' six 'gradio==5.29.0' numpy
python notebooks/ltx25_native/build.py
python notebooks/ltx25_native/test_ltx25.py
```

The tests use the actual Kaggle SDK's file-token/upload/create flow with remote blob/status/create calls stubbed and network sockets disabled. They also check embedded notebook payloads, source identities, permitted inference changes, model catalog completeness, Dataset isolation, checksum staging and budget controls. Numerical guard tests use tiny NumPy-backed tensor doubles; they do not execute the full model. Actual Gradio components and its HTTP `/config` response are checked without generating a video.

No authenticated Kaggle upload, full-size GPU load/render, real output quality, measured generation speed or live public tunnel was performed. These are runnable test artifacts with local validation, not a claim of confirmed T4 performance. Nothing has been pushed to GitHub.
