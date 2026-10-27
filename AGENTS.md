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
