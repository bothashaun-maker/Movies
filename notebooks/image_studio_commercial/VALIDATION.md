# Validation — commercial v7

Completed locally on CPU, 9 November 2026:

- Eight commercial-profile tests: catalog/SPDX allowlist, rejection of unexpected weights/incomplete uploads, reuse of a saved input without download, included license notices, read-only metadata triage, isolated runtime settings, three-reference duet structure, private Dataset submission/receipt reuse with a mock API, embedded pack compilation/catalog checks.
- Four notebook/graph checks: single executable cell compiles, complete connections/reachability, live ComfyUI input schemas, embedded workflows equal generated source.
- Nine JJP tests: eight worlds, custom wing/palette, exact scripture references, explicit cast approval, saved-reference hashes, branded master/delivery records, output geometry and duet records.
- Real pinned ComfyUI CPU server loads only the two new custom-node directories under the whitelist. All **24 workflows** load in Chromium and serialize to API prompts without missing nodes. Four-step scheduler, CFG 1, BFS 4B filename/strength, model connection, brief dimensions, cast prefixes and duet reference count were checked.
- Real MediaPipe Face Landmarker inference and piecewise-affine composite of NASA/US Navy public-domain sample portraits, followed by SaveImage through ComfyUI’s `/prompt` API. Target geometry is retained. Visual inspection confirms this simple compositor can have poor edges/mouth artifacts and pose/expression distortion. It is a supplemental method, **not** the recommended quality replacement for InSwapper. Feather preserves source features more strongly than optional Poisson blending. No user photos were used.
- Real official Real-ESRGAN x4plus inference through ComfyUI on a 24×24 synthetic image: 2× produced 48×48 and 4× produced 96×96. This tests execution/geometry, not photographic quality or speed.
- Downloaded the exact BFS 4B LoRA, verified its full SHA256 against the pinned author artifact, opened the safetensors file, confirmed `flux2_klein_4b` training metadata and 160 adapter tensors. Native adapter parsing was checked separately; this does not establish compatibility by successful full-model inference.
- All three Google component model cards explicitly list Apache 2.0. Original model-license evidence, BFS card/guide and Apache/BSD notices are packaged.

Not verified here: FLUX/BFS GPU image generation, full SeedVR2 restoration, T4 peak VRAM/speed, user-specific likeness, comparative quality, temporal video stability, live Kaggle runtime/Cloudflare, and actual authenticated Kaggle Dataset creation. The pack API path is implemented and testable with existing credentials when run by the user; no Dataset was uploaded from this environment. Two T4s are not pooled by this profile.

No model is described as a measured overall “best.” BFS 4B is the selected dedicated **head-swap candidate** for the commercial/T4 constraints, with generic reference editing and a simple CPU mesh fallback. See RESEARCH.md for alternatives and exclusions.

Rebuild from `notebooks/image_studio_commercial`: `python build.py`. Run `python test_commercial.py`; with the pinned CPU ComfyUI server run `python check.py`, `python test_jjp.py`, `python test_face_api.py`, `python integration_check.py` and `python browser_check.py` (Playwright/Chromium required for the latter). Large-model inference is deliberately not triggered by these tests.

## Dataset finalisation repair — 10 November 2026

The user's real create_edit run downloaded and uploaded all three model files before the final Dataset operation failed. The old exception handler suppressed the server reason. Corrected an invalid underscore in the Dataset slug (`create_edit` → `create-edit`), added pre-upload name validation, retained sanitised server diagnostics, excluded internal state from uploads using temporary hardlinks, and checked remote status/manifest before repeat uploads. Weight sources and checksums are unchanged. Nine new regression tests cover valid create-edit names, private/curated upload, preserved rejection reason and models, receipt reuse, existing remote manifest reuse, processing/mismatched datasets, access errors, and credential redaction. The eight existing commercial tests also pass. No live Kaggle finalisation was performed here.

## Direct Dataset creation repair — v3

The user's v2 log failed at our added lookup: `403 Permission datasets.get was denied`, before `dataset_create_new` ran. Compared the earlier image pack API builder and the supplied H3 saved-output notebook directly. Restored the image builder's direct private `dataset_create_new` call, removing our own read/status/remote-manifest prerequisite. The SDK retains its own supported creation behaviour. The corrected slug, saved weights, private flag and actionable errors remain. Tests cover the exact denied-lookup regression, no false success for name conflicts, the actual creation endpoint rejecting a request, and SDK response objects. Full authenticated creation cannot be executed here without the user's Kaggle session.

## Automatic finish-pack preparation

The user attached create_edit successfully, but startup still required a separately prepared finish pack. The main notebook now prepares missing finish weights automatically, reuses verified saved inputs, and calls the working v3 direct Dataset publisher. A publication failure retains usable local models and reports the need to save outputs; it does not claim persistence. Missing core models still fail before downloads. CPU regression tests cover these paths; full GPU startup and authenticated publication remain unverified here.
