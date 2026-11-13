# Director: sequential preparation, sampling and decoding

Import the current [Director notebook](../Director.ipynb) into a **fresh Kaggle session**. Attach the same four compatible H3 packs (ref2va, fl2va, encoder, support), wait for Adding data to finish, select T4 ×2, enable Internet, and run its one setup cell. Existing Kaggle copies do not update from GitHub.

The prior diffusion-only staging fix left the Qwen encoder loading directly from the Dataset mount. Uploaded diagnostics showed repeated GPU-idle snapshots after `Requested to load MiniMaxH3TEModel_`. The earlier all-model copying approach used too much disk. This revision uses the following sequence:

1. Stage the encoder and reference VAEs for preparation. No diffusion model is loaded. The actual pinned Director compiles the same timeline and encodes the same image/video/audio references.
2. Save conditioning, initial joint latent, audio and Director metadata to an atomic safetensors checkpoint, then exit the preparation process.
3. Release only exact ledger-tracked inactive local model files. Stage the active diffusion family and its adapter into the same cache, then sample in a fresh process. SigmaShift is applied to the actual model here.
4. Save sampled latents and exit. Stage the selected decoders, decode and export in a third process.

The model cache is capped at 24 GiB per phase independently of the host filesystem's free-space report. The cap does not measure Kaggle's total quota: software, unrelated files and output clips also consume disk. Reference inputs, outputs and untracked files are preserved. Native audio and timeline replacement audio remain distinct and are preserved across the process boundaries. A phase cache is temporary; it does not replace your saved Datasets or output backups.

Missing model files stop with their filenames by default; Director no longer silently downloads missing weights. Set `ALLOW_MODEL_DOWNLOADS=True` only when intentionally allowing online fallback. No Dataset rebuild is required if all pinned files are present.

The controller reports **Preparation**, **Sampling** and **Decoding** separately. Execution stops after 5, 10 and 5 minutes respectively, with process cleanup; startup and file staging are separate. These are stop limits, not expected render times. Successful preparation and sampled latents can be reused on retry. Changed input media, timeline, models or implementation invalidate recovery identity.

Validation covers both shipped Director graphs with references on/off, correct model-family selection, preservation of external audio, exact safetensors round trips, process order/retry after sampling failure, local phase-cache replacement, file preservation and deadlines. The pinned upstream Director preparation method is also exercised on CPU with lightweight encoders, including native image-reference conditioning and frontend-only inputs. Full-weight T4 execution, runtime and visual results remain unverified here.

Sources: `notebooks/h3_suite/director_*.py`; regenerate with `python notebooks/build_h3_collections.py` and run `tests/test_director_phases.py` with the pinned CPU runtime and Director checkout for the optional integration checks.
