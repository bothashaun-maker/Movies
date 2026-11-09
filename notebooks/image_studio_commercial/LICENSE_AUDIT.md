# Model and dependency audit — 9 November 2026

The earlier choice of InSwapper for a music-video production pipeline was unsuitable. A permissive application license does not grant commercial use of all model weights downloaded by it. This audit covers the bundled image profile, not every historical file across the user's music/movie repositories. It is a source-based license review, not a guarantee about training data, likeness rights or individual output ownership.

| Component | Finding | v7 action |
|---|---|---|
| InSwapper 128 | InsightFace downloadable pretrained models are non-commercial research only, distinct from MIT code | Remove |
| buffalo_l | Detection/recognition/landmark pack has the same pretrained-model restriction; changing only the swap generator is insufficient | Remove entire pack |
| Juggernaut XL v9 | OpenRAIL-M metadata, but model card adds paid API/commercial licensing wording; not a blanket prohibition on all commercial images | Replace default; older output needs exact-terms review |
| FLUX 3 optional | Custom FLUX Kommunity license/gated terms; full accepted terms were not available in this environment | Remove from this commercially permissive default; do not label every output non-commercial |
| FLUX.2 Klein 4B distilled | Creator explicitly licenses 4B under Apache 2.0; 9B is a different license | Use only pinned 4B |
| Qwen3-4B text encoder, Flux2 VAE | Apache 2.0; pinned Comfy-Org split artifacts | Include; no FaceID recognition encoder |
| BFS head v1 Klein 4B LoRA | Author’s original model repository declares MIT; paired only with Apache Klein 4B, no recognition/detection dependency | Include exact pinned 4B adapter; preserve original model card |
| Google Face Landmarker bundle | Official BlazeFace short-range, Face Mesh V2 and Blendshape V2 model cards explicitly state Apache 2.0 | Use for mesh coordinates, not identity embeddings |
| Real-ESRGAN x4plus | BSD 3-Clause | Retain verified weights and notices |
| SeedVR2 3B/EMA VAE | Apache 2.0, including Comfy-Org conversion metadata | Retain pinned weights |

Face-swap alternatives researched: GHOST (`ai-forever/ghost`, Apache generator) has a recognition-weight dependency question documented in its issue tracker. GhostV2 (`dimitribarbot/ghostv2`, BSD generator) grants use of its own weights, but the default CVLFace AdaFace ViT WebFace4M card directs users to the training-dataset license. Neither is an unqualified drop-in commercial recommendation without resolving the complete chain. SimSwap's official pretrained usage is CC-BY-NC 4.0, FaceDancer is CC-BY-NC-SA, and a permissive FaceFusion wrapper does not override HyperSwap/AlphaFace/UniFace weight-specific or undocumented restrictions. We did not scrape all of GitHub or benchmark every model; no such claim is made.

The expanded search added BFS’s dedicated MIT-declared Klein 4B head-swap adapter. v7 therefore offers dedicated BFS head replacement, generic reference identity editing, and a supplemental landmark pixel compositor. These are different methods, and no quality leaderboard has been measured here. For an InSwapper-style learned swap with dependable commercial rights, obtain written permission covering **every** required generator, recognition and detection weight, then add that exact licensed stack in a separately reviewed profile. No such permission is assumed here.

Primary sources:

- https://github.com/deepinsight/insightface#license — code versus pretrained weights.
- https://huggingface.co/RunDiffusion/Juggernaut-XL-v9 — model card and added terms.
- https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy — optional old model's terms.
- https://github.com/black-forest-labs/flux2 and https://huggingface.co/black-forest-labs/FLUX.2-klein-4B — creator's Apache 4B statement.
- https://huggingface.co/Comfy-Org/flux2-klein-4B/tree/5b3abca5c0dc1ecd4eaf3340868d09063fc1c965 — exact split weights; https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE.
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker — links to all three original model cards; exact source URLs and license excerpts in licenses/mediapipe-model-evidence.txt.
- https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE and https://github.com/ByteDance-Seed/SeedVR/blob/main/LICENSE.
- https://github.com/ai-forever/ghost/issues/88 and https://github.com/dimitribarbot/ghostv2 — dependency review.
- https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m — default GhostV2 recognition card.
- https://github.com/neuralchen/SimSwap and https://github.com/felixrosberg/FaceDancer — pretrained restrictions.

License snapshots are retained with the pack. Redistribute only with their original notices/conditions; your photo, music, likeness and publication permissions remain separate.

BFS original weight declaration and matching-family guide: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/2209ae537ac5e883d6ec7db5b7bd6dccf1a0eba7/README.md and docs/flux-2-klein.md. Author workflow includes optional third-party nodes (including RMBG); those are **not** imported. Our rebuilt graph uses only the reviewed 4B base, encoder/VAE and adapter, with native reference latents.
