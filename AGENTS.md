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
  cells they must paste into Kaggle. Avoid repeated setup and wasted GPU time.

## Current project

- Target: Kaggle with two NVIDIA T4 GPUs, four-step MiniMax H3 Turbo LoRA video
  generation, and native audio. Act as the user's engineering collaborator and
  carry fixes through implementation, validation and publication.
- Current notebook: `minimax_h3_kaggle_complete.ipynb`.
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
  A full pretrained render of that rewrite on Kaggle T4 hardware was not verified
  in the development environment. Distinguish those checks from actual GPU
  rendering, and report measured performance only when evidence exists.

This file preserves actionable project context and preferences, not the full
chat transcript. Update it when the user changes these instructions.
