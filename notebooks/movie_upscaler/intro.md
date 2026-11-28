# Movie Upscaler v2 — Upload clip · T4 quality settings

1. Attach your **existing Movie Upscaler model Dataset**. No new model pack is required.
2. Select **GPU T4 ×2**, Internet ON. Stop any older studio first. Run this single startup cell and open **MOVIE STUDIO**.
3. In **Movie • Upload clip**, click **Upload clip** and choose your MP4/MKV/MOV/WEBM/AVI/M4V. Wait for **Ready**. The source connection is automatic. No Kaggle clip Dataset or path copying is needed.
4. Use **Movie Studio - SeedVR2** for temporal restoration, or choose **Movie Studio - RealESRGAN** from Workflows. Upload/select a clip in the workflow you use.
5. Press **Run once**, queue count **1**, frontend **Auto Queue OFF**. The save node continues through the clip. To pause: **Interrupt, then Clear Queue**. Run again to resume.

Both workflows default to **2×**, a **1920-pixel longest-edge cap**, and **lossless RGB MKV** export. This avoids a final lossy encode. It does not invent detail missing from the source. For a smaller, widely compatible video, select **high_quality_mp4** in the save node: **CRF 14 / slow** are already set. Lower CRF increases file size. Lossless ignores MP4 CRF/preset. Both retain original audio where the container supports it. MP4 may convert incompatible audio to AAC.

**SeedVR2 defaults:** existing **3B Q8** model, **FP16 VAE**, SDPA, full 32-block CPU swapping, 17-frame saved chunks with 4 context frames on each side, **5-frame temporal batches**, **uniform final batch**, **overlap 2**, **LAB** colour matching, **zero injected noise**, fixed seed 42, and tiled encode/decode **512 / overlap 128**. DiT uses GPU 0; VAE uses GPU 1 when available, otherwise GPU 0. These are separate processing stages, not pooled VRAM or a promised speedup. No unsupported FP8/Sage/Flash/compilation path is enabled.

**Real-ESRGAN defaults:** existing **x4plus** model with native tiled upscaling, followed by **Lanczos** to the exact target. Changing source **scale** from 2 to 4 automatically changes the final resize factor. There is no second scale setting to synchronize. Context is 0 because this model processes frames independently. It can flicker.

The connected SeedVR2 **resolution** is the source short edge × scale. Leave the connection intact. `max_resolution=0` prevents a silent secondary resize. The source node enforces the explicit target cap. Raising batch size, tile size or target resolution is not a guaranteed quality upgrade or a proven T4 fit. Larger temporal batches increase memory. 5 retains temporal processing within a conservative budget. VAE tiling does not tile DiT attention.

Uploads show progress and use small HTTP chunks. Failed uploads never replace a completed clip. Uploaded clips live under `/kaggle/working/movie-studio/input/movie-clips/`. Restored chunks and final movies are under `/kaggle/working/movie-studio/movie-jobs/`. Completion shows the exact output path in the save node and log. Save Version with output files before ending Kaggle. Re-running a completed graph reuses restored chunks. Changing only export format/CRF/preset re-exports without upscaling again.

This remains an **SDR, constant-frame-rate, square-pixel, RGB8** workflow, not HDR/10-bit preservation. Lossless means exact preservation of the processed RGB8 frames. Output frame count, resolution and audio-stream count are checked. Disk checks reserve space for checkpoints and the final lossless master. A scene cut should be processed as a separate clip to avoid temporal mixing across shots.

Settings rationale and validation are in `notebooks/movie_upscaler/QUALITY_SETTINGS.md`. Upload/browser/media tests are not evidence of full-size T4 SeedVR2 inference, memory fit, speed or comparative quality. AIQuest was read for upload/interaction reference and remains unchanged. Your model Dataset and model builder are unchanged.
