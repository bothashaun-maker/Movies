# User preferences and project continuity

These instructions record the user's explicit requests from the MiniMax H3
Kaggle notebook conversation, saved on 2026-10-25.

## Notebook changes and GitHub

- The user instructed: "Whenever changes are made to the notebook, automatically
  write it to my github."
- After changing a notebook, run the relevant checks, commit the intended
  changes, and push to `bothashaun-maker/Movies` on `main` without asking for
  another confirmation. If the local branch has another name, use
  `git push origin HEAD:main`.
- Preserve unrelated user changes. Do not force-push. Reconcile remote updates
  if necessary, and report any validation or publication blocker honestly.
- Give the user the updated GitHub notebook link. Do not claim a successful push
  until it has succeeded.
- The user wants a complete, self-contained notebook, rather than separate repair
  cells they must paste into Kaggle. Starting its single executable cell must
  continue through setup, weight verification and a Cloudflare link to ComfyUI without
  another manual cell start. The user explicitly rejects notebook video previews
  and download prompts. Do not automatically render; open persistent ComfyUI. Install the adapted
  `MiniMax H3 - Full T4 Workflow` canvas in ComfyUI’s workflow menu, ready for the user
  to press Run; browser access alone is insufficient. The user requires a visible,
  connected native node graph. Do not collapse it into one wrapper node. Avoid repeated setup and wasted GPU time.

## Current project

- Target: Kaggle with two NVIDIA T4 GPUs, four-step MiniMax H3 Turbo LoRA video
  generation, and native audio. Act as the user's engineering collaborator and
  carry fixes through implementation, validation and publication.
- Current notebook: `minimax_h3_full_workflow.ipynb` (FULL-GRAPH-v1). The complete and cloudflare_fixed filenames remain identical aliases. GitHub updates do not update an already-imported Kaggle copy; give the current distinctly named notebook. Its Cloudflare link must open the full graph directly.
- Edit its sources in `notebooks/h3_complete/`, then regenerate it with
  `python notebooks/build_h3_complete.py`. Keep source and generated notebook in
  the same commit.
- Read `H3_KAGGLE_SETUP.md` for setup, model storage, optional Semantic Bridge,
  supported acceleration and validation commands. Earlier notebooks and repair
  scripts are retained for reference; use the complete notebook for new work.
- A prior run exhausted GPU memory when loading the dual-GPU helper VAE while
  sampler allocations remained resident. The rewrite saves joint latents,
  stops and waits for the sampling process, then decodes in a fresh process.
  Preserve that process boundary and automatic checkpoint reuse when fixing or
  optimizing the workflow.
- Model storage must use measured available space and reuse verified files;
  do not assume Kaggle provides a fixed temporary-storage quota.
- CPU graph/controller tests and MP4/audio export checks passed for the rewrite.
  The latest uploaded Kaggle notebook confirms a successful 124-frame, 512 × 768
  render with audio. Browser startup through Cloudflare is now the default; a live
  Kaggle tunnel is not verified from this development environment. Distinguish those checks from actual GPU
  rendering, and report measured performance only when evidence exists.

This file preserves actionable project context and preferences, not the full
chat transcript. Update it when the user changes these instructions.

## Experimental preset (2026-10-27)

- `minimax_h3_taomate3_experimental.ipynb` adds TaoMate 3-step + INT8 VAE as an optional comparison; baseline aliases remain four-step FP16.
- Do not call this the full Director workflow: it adapts sampling/VAE into the existing first-frame graph, without Director timeline/multi-reference controls or RTX upscaling.
- Model hashes and manual sigmas are verified; actual GPU render compatibility/performance remain unverified. See `docs/H3_DIRECTOR_WORKFLOW_REVIEW.md`.

## User decisions and workflow collections (2026-10-27)

- The user explicitly wants the Director included. Omitting its timeline and multi-reference interface from the experimental adaptation was an incomplete scope decision, not a demonstrated T4 limitation. The new `Director.ipynb` integrates both real Director canvases and their timeline/multi-reference interface. Browser and CPU validation passed; GPU execution remains unverified.
- The user will handle upscaling in CapCut. Do not add RTX upscaling to the required Kaggle pipeline.
- The uploaded Tensor Alchemist three-step and PDD8 workflows contain `MiniMaxH3DirectorCS`. Preserve their reference/timeline intent when adapting conditioning to isolated GPU workers.
- The subsequently uploaded `Ep32 Workflows.zip` (Pixaroma) was inspected directly. It contains five workflows: text-to-video, first-frame image, first/last-frame images, reference image + speech audio, and reference image + singing audio. It does NOT contain the Director timeline node.
- Pixaroma's speech/singing graphs use `MiniMaxH3ReferenceToVideo` and `PixaromaH3AudioSync`, which are useful candidates for true reference conditioning and supplied-audio synchronization. These features are now integrated in `Pixa_5_Workflows.ipynb`, including preservation of supplied audio across isolated decoding.
- Pixaroma selects the FL2V four-step v1.1 LoRA and FP16 video VAE. Its saved dropdown state selects eight steps for text/first-frame/first-last-frame workflows (four-step options available), and four steps for speech/singing. Follow linked dropdown controls rather than stale KSampler widget values when auditing these graphs.
- Creator benchmark claims are not measured Kaggle T4 results. Existing diagnostic evidence confirms both T4s were used in our earlier run; sampling including initialization took 193.09 seconds and decode/export took 43.38 seconds. Speed and quality remain optimization goals, not resolved outcomes.
- User asked to remember these decisions. These repository instructions preserve actionable continuity; do not claim guaranteed permanent recall of the full chat.

## Four separate notebooks (latest request, 2026-10-27)

- User superseded the combined-notebook request with four named notebooks: `Current_Working_Workflow.ipynb`, `Pixa_5_Workflows.ipynb`, `Director.ipynb`, and `Three_Step_Workflow.ipynb`.
- Regenerate all four using `python notebooks/build_h3_collections.py`. It also runs the baseline builder. Collection code lives in `notebooks/h3_suite/`; original uploaded JSON is retained in `workflows/sources/`.
- Pixa installs all five visible graphs. Director installs the actual three-step and PDD eight-step Director graphs. Three-Step is the simpler experimental first-frame graph, not a substitute for Director.
- Preserve the lazy output boundary, separate GPU sampling/decode processes, external-audio sidecar, and pre-download geometry checks. Default collection size is about 0.4 MP and five seconds; caps are 430,000 pixels and 124 decoded frames.
- Select weights from the reachable graph, including Director's active reference branch. Family switches may require downloads. Reclaim only verified, notebook-owned model caches when measured space requires it; never remove inputs, outputs, or checkpoints.
- The uploaded Pixa archive has two unreadable media entries. Intact examples are bundled; Speaking Audio requires the user's recording. Do not invent a speech sample.
- See `docs/H3_NOTEBOOK_COLLECTIONS.md` for menus, verification and remaining GPU limits. 47 tests passed; all seven new canvases loaded in real ComfyUI Chromium and validated as browser/sample/decode graphs. This is not proof of T4 speed or quality.

- `H3_Model_Packs.ipynb` prepares all four notebooks’ default weights as four CPU-only reusable outputs (`ref2va`, `fl2va`, `encoder`, `support`). Regenerate with `python notebooks/build_h3_model_packs.py` when catalogs change. Each pack is below 20 GiB; combined storage is about 73 GiB. Attach saved outputs through Add Input; temporary renderer files/symlinks are not durable datasets.

## Movie upscaler (2026-11-07)

- User now requested a separate Kaggle movie upscaler and explicitly authorized
  building and pushing it. This supersedes the earlier CapCut-only preference
  for this separate tool; do not add it as a required H3 generation stage.
- `Movie_Upscaler_Kaggle_v1.ipynb` opens native connected SeedVR2 and Real-ESRGAN
  graphs. Rebuild both it and `Movie_Upscaler_Model_Packs.ipynb` with
  `python notebooks/movie_upscaler/build.py`. Keep sources and notebooks together.
- Preserve explicit Run, saved model-pack outputs, durable chunk receipts,
  audio/frame-count checks and no notebook previews/download prompts.
- Uses GPU 0 only. CPU/media/browser checks are not evidence of T4 performance
  or SeedVR2 inference. See `notebooks/movie_upscaler/README.md` for validation.

## H3 Next experiment (2026-11-07)

- User authorized building and pushing the new research-based notebook. `H3_Next_T4.ipynb` is separate from the existing four collections; regenerate only with `python notebooks/build_h3_next.py`. Sources are `notebooks/h3_next/`.
- Installs three visible 18-node canvases: Singularity/Pika four-step, Singularity/Pika two-step draft, DenoMix eight-step. Molmo2 INT8 uses an experimental FP16/SDPA adaptation for T4; Sol INT8 decodes output video. Original H3 VAE handles reference images. Optional motion dial defaults to zero. No upscaler, sparse attention, SCAIL motion-transfer or Director timeline is implied.
- Preserve three isolated preparation/sampling/decoding processes and exact conditioning/reference checkpoints. Molmo2 must exit before loading diffusion weights to reduce Kaggle host RAM pressure. Include new code identity in retry keys. Prefix/scale adaptation must remain a reader operation, never a second full checkpoint copy.
- `H3_Next_Model_Packs.ipynb` prepares four reusable CPU-only saved outputs: singularity, denomix, molmo, support; combined about 62 GiB, each below 20 GiB. These differ from the older H3 packs. Defaults to automatic private Dataset submission (`AUTO_CREATE_DATASET=True`) using Kaggle Secrets authentication, checked before downloading. Run the cell and attach via Your Datasets after processing; no manual Save Version is required for submission. Saved Notebook Outputs remain an optional fallback. Upload tests use a mock API, not real Kaggle credentials.
- All three canvases loaded in actual ComfyUI Chromium; browser/preparation/sample/decode schemas passed for text and image; conditioning/reference safetensors round-trip and native reference geometry passed CPU checks. Full new-model GPU rendering, actual memory use, speed and quality remain unverified. Do not describe preset names as benchmark results.
- Read `docs/H3_NEXT_T4.md` for simple import/use/storage instructions and limitations. Existing notebooks must remain unchanged when updating this experiment.

- Movie model-pack correction: user explicitly requested the automatic Dataset
  creator/saver already present in `notebooks/image_studio/asset_builder.py`.
  `Movie_Upscaler_Model_Packs.ipynb` now defaults to automatic private Dataset
  submission, using private Kaggle API credentials. Saved notebook outputs are
  an optional fallback, not the default persistence instruction. Never claim
  live Kaggle upload success from mocked API checks.

- H3 Next model-pack storage fix: automatic Dataset uploads may use measured `/kaggle/temp` or `/tmp` space, with same-filesystem hardlink staging and partial-download accounting. Never force the 19.53 GiB checkpoints into 19.50 GiB working storage. Saved Notebook Outputs mode must remain working-only; temporary storage is not durable before upload. Do not remove the free-space guard or assume a 50 GiB quota.

- Director/Pixa staging correction (2026-11-13): copy ONLY the active diffusion checkpoint from `/kaggle/input` to scratch for tensor sharding. Encoder, VAEs and LoRAs stay attached. Cap local diffusion copies at 24 GiB regardless of host free-space reports. Release only exact ledger-tracked inactive checkpoint files on a family switch; preserve untracked/user files and all attached inputs. Account for pending downloads and reserve in free-space checks. This cap bounds checkpoint copies, not Kaggle's total runtime/output quota. Recommend a fresh session after the old all-model staging version; do not automatically delete its other files.

- Director three-phase correction supersedes its diffusion-only staging rule: the encoder also stalled on Dataset-mounted reads. Director now prepares timeline conditioning/reference latents in one process with only that phase's local weights, saves exact safetensors, exits, then releases tracked inactive cache and stages diffusion/LoRA for sampling. Decode is a third isolated process. Share one 24 GiB phase cache; never stage encoder and diffusion together. Preserve timeline/image/video/audio/retake semantics, output slots and actual SigmaShift in sampling. `director_prepare_patch.py` exposes the pinned upstream conditioning method without its two model-selection/shift statements. Missing weights default to a clear failure (Director downloads OFF); no silent fallback downloads. Completed preparation/latents support retries. Execution deadlines are preparation 5 min, sampling 10 min, decode 5 min, separate from setup/copy time. CPU checks do not establish T4 speed, fit or output quality.


## Image Studio storage — user decision, 10 November 2026

- User explicitly requires **all image creation, face/head editing and upscaling models in ONE saved Kaggle Dataset**, and only that one Dataset input in the studio. Do not split create_edit/finish inputs or prepare/download missing packs during GPU studio startup.
- Use the confirmed working Model Packs v3 direct private Dataset creation method. Preserve authentication and upload behavior; do not add a datasets.get/status permission prerequisite.
- Current pair: `Image_Studio_All_Models_Dataset.ipynb` (CPU, all eight reviewed weights, PACK='all') and `Image_Studio_Commercial_v9.ipynb` (GPU, verify saved models only). Workflows belong in the studio notebook, not model Dataset files. Keep this simple working structure unless the user requests a change.

- Combined Dataset storage correction: the complete model set is 20.53 GB, but the CPU creator must not require that entire set plus a second local/staging copy. Use `Image_Studio_All_Models_Dataset_v2.ipynb`: preserve the original authenticated `dataset_create_new(public=False)` call and original SDK uploader, supplying one verified file at a time inside that single request. Reuse existing sources without copying/deleting them; release only this creator's successfully uploaded temporary downloads. Peak missing-file storage is the largest missing model plus reserve. Studio v9 remains unchanged and takes one Dataset input.


## General image workflows — latest user requirement

- The user explicitly rejected episode/story/brand production features. Image Studio must provide ordinary cinematic image creation, image editing/inpainting, face/head swaps and upscaling with direct prompt/upload/run/save controls. No episode presets, scripture fields, cast approval, branded exports or required production stages in the active notebook/workflows.
- Current active notebook: `Image_Studio_Normal_v10.ipynb` (alias `Image_Studio_Normal.ipynb`), sources under `notebooks/image_studio_normal/`. It uses the SAME existing ONE ALL MODELS Dataset. Do not rebuild/change the Dataset, model set, working upload/authentication method or original upscalers for a UI/workflow change.
- Earlier production notebooks/playbooks remain historical artifacts, not the active image interface. Archive old app-managed menu files outside the workflow menu; preserve user photos, images and personal graphs. Keep the visible native ComfyUI graph.

## JJP WanGP Motion Test Kit (2026-11-13)

- User explicitly requested a new standalone WanGP notebook, native WanGP GUI through Cloudflare, complete Dataset requirements and bounded T4 behavior. This experiment uses no ComfyUI server or canvas; earlier ComfyUI graph requirements do not apply to this new notebook. Existing notebooks remain unchanged.
- Sources: `notebooks/wan_gp_motion/`; rebuild only the new pair with `python notebooks/wan_gp_motion/build.py`. Outputs: `JJP_WanGP_Motion_Test_Kit.ipynb` and CPU-only `JJP_WanGP_Motion_Models.ipynb`.
- One new private Dataset contains all 13 pinned WanGP-native files (40.531871 GiB). Preserve Image Studio's working native streaming Dataset-upload method and private authentication flow. Director/H3 Next packs are not substitutes. No model downloads during GPU inference.
- GPU 0 only, Profile 5, FP16/SDPA, native Ref2VA pruned PDD8, fixed 124 frames / 864x480 / 24 fps. Two T4 memories do not pool. Pin upstream WanGP and dependencies; retain exact-source guards and hard deadlines. Do not casually add Triton/FP8/compilation paths or change the frame-alignment global.
- H3's hard-coded BF16 model definition overrides `--fp16`; preserve the kit's narrow FP16 patch and compute-dtype header fix. Use the original FP16 VAE through WanGP's `bf16` selector identifier. Native `--lock-model` crashes this pinned GUI, so model restrictions are enforced by validation instead. Resolution choice must use WanGP's two-element `[label, WxH]` schema and be scoped to the specific PDD preset.
- Initial local mirror capacity gate is about 52.54 GiB; measure writable scratch and count reusable local files. This trades disk capacity for avoiding Dataset-mounted weight-read stalls. Fail clearly before model load when unavailable; never claim every Kaggle runtime has this space.
- CPU/bootstrap/native settings/browser/media and 17 regression checks passed. CPU native checks explicitly simulate only hardware metadata, never inference. Real T4 H3 rendering, memory fit and motion fidelity are unverified. Cloudflare live test here failed on the development network before tunnel creation; live Kaggle upload/tunnelling are also unverified. Read `notebooks/wan_gp_motion/VALIDATION.md` before making performance/readiness claims.
- User reports the H3 Next **Dataset builder** works and explicitly requests reusing the working builders. CPU-BUILDER-v1.2 (14 November) now uses the notebook's own Python and installed Kaggle SDK, as H3 Next does; preserve that choice. Do not reintroduce an isolated CPU environment or upgrade a working SDK unnecessarily. The GPU app environment remains isolated. The initial CPU lock omitted Kaggle's undeclared `six` dependency; its failure was reproduced in a fresh environment, and the explicit dependency plus real SDK/file-token tests were added. Test the CPU path with only its own dependencies, never just inside the full video environment.
- User explicitly requires **only WanGP's needed models**. Reuse H3 Next's setup/authentication approach, not its model catalog. Keep the separate exact 13-file WanGP Ref2VA Pruned PDD8 allowlist; no Singularity, DenoMix, Molmo, Sol, FL2VA or ControlNet additions for this test.
- First real T4 startup passed model staging, installs/checks and the kernel/import probe, then failed GUI startup because inherited `MPLBACKEND=module://matplotlib_inline.backend_inline` is unavailable in the isolated child. GUI-STARTUP-v1.3 forces `MPLBACKEND=Agg` in the copied child environment only. Exact exception reproduced and corrected on CPU; full-command native GUI/Chromium check passed with the inherited failing value. Preserve this environment normalization and probe. Startup errors must expose log tail and exit code. Native tests share the actual launch command, not a reduced approximation. These checks do not prove GPU video generation or Cloudflare connectivity.
