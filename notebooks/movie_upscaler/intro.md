# Movie Upscaler v1 — Kaggle T4 / ComfyUI

Two visible connected workflows: **SeedVR2 temporal restoration** and **Real-ESRGAN fast enlargement**.
Single-cell startup; no movie runs automatically. Uses **GPU 0** (one T4), leaving the second GPU unused.

1. Run **Movie_Upscaler_Model_Packs.ipynb** with **Accelerator: None**, Internet ON and private Kaggle API authentication. It automatically uploads a private Dataset and prints its link.
2. Import this notebook; select **T4 GPU**, Internet ON. After Dataset processing, attach it via **Add Input → Your Datasets**. Saved Notebook Outputs also remain supported.
3. Attach your source video (or copy an H3 output to accessible Kaggle storage). Set `source_path` in the leftmost ComfyUI node to its full path.
4. Run this single startup cell and open the printed **MOVIE STUDIO** link. It loads SeedVR2's graph directly.
5. Press **Run once** in ComfyUI. Keep frontend **Auto Queue OFF** and queue count **1**. The save node queues one next chunk until finished.
6. To pause, **Interrupt then Clear Queue**. Run the same graph/settings again to resume verified chunks. Set `auto_continue=false` for manual one-chunk-at-a-time processing.
7. Final movies and checkpoints are under `/kaggle/working/movie-studio/movie-jobs/`. **Save Version with output files** or preserve these files before ending Kaggle.

## Settings

- **SeedVR2:** 3B Q8 model, FP16 tiled VAE, CPU offloading, 5-frame temporal batches, overlap 2. Source chunks have 4 context frames on each side; these are cropped before saving, with no duplicated output frames. Context reduces boundary risks but does not guarantee invisible seams. Defaults to 2×, with an explicit 1920-pixel longest-edge cap. Increase the cap only after testing a short clip; VAE tiling does not bound transformer attention memory.
- **Real-ESRGAN:** select it from Workflows. Defaults to 2× (4× model then downsample). For **4×**, set source `scale=4` AND `ImageScaleBy.scale_by=1`. Mismatched dimensions fail before checkpoint commit. This model processes frames independently and can flicker.
- `chunk_frames=17` bounds the images in each ComfyUI execution; increasing it uses more RAM. Changing restoration, chunk, context, or target settings starts a distinct checkpoint job. Changing auto-continuation/export container does not discard restored chunks.
- Models must be attached by default. `REQUIRE_ATTACHED_MODELS=False` deliberately allows online model downloads. `ENABLE_SEEDVR2=False` runs only Real-ESRGAN and needs only its checkpoint. The image pack's VAE/Real-ESRGAN files can be reused, but its native INT8 SeedVR2 model is **not** the Q8 GGUF used here.

## Media and storage

This version supports **SDR constant-frame-rate, square-pixel video**, producing RGB8 checkpoints and H.264 YUV420 output. HDR, variable-rate input, and display rotation must be converted explicitly first. High-bit-depth SDR is reduced to 8-bit; this is not an HDR preservation pipeline. Input timestamps are checked before inference.

Frame count, rational frame rate, dimensions and audio-stream count are checked. Original audio streams are copied when supported. MP4 falls back to AAC if stream-copy fails; MKV keeps supported original audio and subtitles. Video is re-encoded at CRF 18. No frame interpolation or automatic face redrawing. Metadata/chapters are copied; arbitrary attachments/data tracks are not retained.

Lossless FFV1 chunk files cost disk space. Startup checks measured free space against a conservative full-job checkpoint estimate and reserve. Nothing automatically deletes source files, other jobs or completed checkpoints. Chunk decoding seeks with a short preroll and selects validated source timestamps, avoiding a full prefix decode for every chunk. Full-length films may exceed Kaggle time/disk quotas; start with a short representative clip.

Same-session interruptions resume automatically. Across sessions, restore the saved `movie-jobs` directory to the configured working output path before Run. Models are reused through the automatically created Dataset or saved Notebook Outputs. Software and tunnel setup still need Internet. Keep the startup cell running; anyone with its temporary URL can access the studio.

## Validation status

See `notebooks/movie_upscaler/README.md` for completed checks and remaining limits. Local CPU/media checks cannot establish T4 speed, memory fit, or visual restoration quality. No 4K or real-time performance promise is made.
