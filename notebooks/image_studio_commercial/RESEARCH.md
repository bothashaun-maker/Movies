# Source review and recommendation

Reviewed 9 November 2026 for commercial music-video **stills** on Kaggle T4. “Best” means a candidate suited to this hardware and licensing goal; no comparative likeness or temporal-stability benchmark has been run. Repository stars and a model named “Best Face Swap” are not evidence of winning quality.

| Candidate | Commercial dependency finding | Recommendation |
|---|---|---|
| BFS head v1 + FLUX.2 Klein **4B distilled** | Original adapter repository declares MIT; 4B base/encoder/VAE are Apache 2.0; no recognition model | First dedicated head-swap candidate in this build. Compact relative to Qwen; head/hair replacement, not just inner-face transfer. Full GPU inference unverified here. |
| BFS Face V1 / Head V5 + Qwen-Image-Edit 2509/2511 | Adapter MIT declaration; official 2511 model Apache 2.0. Face V1 and Head V5 have different input orders and matching base requirements | Serious alternative for comparison, especially face-only versus full-head needs. Not bundled: substantially heavier separate architecture/download/offload requirements; no evidence here that it beats Klein on T4. Do not substitute Qwen Image **2.1**, which has different terms. |
| GhostV2 | Own pretrained generators BSD; default recognition model’s primary card refers to dataset-license obligations | Commercial permission chain unresolved; not installed |
| Original GHOST | Generator Apache; recognition weights and third-party dependencies require separate review | Not an unqualified commercial replacement |
| InSwapper + buffalo_l | InsightFace pretrained-model NC research terms | Excluded unless separately licensed |
| SimSwap / FaceDancer / CSCS / HyperSwap | NC/research restrictions on typical pretrained routes | Excluded |
| Generic Klein 4B reference edit | Apache native multi-reference editing, no dedicated identity training | Included comparison/baseline; likeness drift remains possible |
| MediaPipe mesh compositor | Explicit Apache model cards, no recognition or neural swap generator | Supplemental CPU pixel transfer for similar poses; not presented as a quality winner |

## Pixaroma checked directly

- GitHub **pixaroma/ComfyUI-Pixaroma** tree at `413e0b2775cfe15be0ab41cc0eff4ce56120d5ea`, including reference-image node source and crop/resize helpers. Its code MIT license is not a license for all models shown in tutorials.
- Official [course index](https://workflows.pixaroma.com/comfyui-course/), [Episode 4](https://workflows.pixaroma.com/ep/4/) and downloaded **EP04 Workflows.zip**. Inspected all eight Klein 4B/9B generation/edit/inpaint/two-image JSON workflows. They confirm the native reference-latent approach. Our production profile uses 4B, not 9B's non-commercial base license.
- [Episode 5](https://workflows.pixaroma.com/ep/5/) and downloaded **EP05 Workflows.zip**. Inspected its eight upscale graphs: resize-only, model upscale, Z-Image/Fluxmania generative routes, SeedVR2 and paid cloud options. Their exact models have separate rights. We retain verified Real-ESRGAN and SeedVR2, without importing the entire extension or paid services.
- [YouTube Klein episode](https://www.youtube.com/watch?v=kNap0VWP1xs), [upscale episode](https://www.youtube.com/watch?v=BRHodz0_Uc4), [H3 character swap/talk/sing episode](https://www.youtube.com/watch?v=YFDXCq4qor4), and corresponding official episode pages. Direct YouTube readers exposed limited HTML/throttling; **no claim of watching full videos or reading full transcripts**. Episode pages and actual downloadable graph files were the implementation evidence.
- Episode 36's official page labels Qwen Image 2.1 non-commercial. Episode 41 describes H3 video character/head replacement and territory conditions. They are not blanket commercial endorsements. Video character replacement is a different task from still-image face/head swapping; no H3 video model is smuggled into the image profile.

The earlier creator review also inspected [AIQUEST/TeamAIQ notebooks](https://github.com/TeamAIQ/Colab-notebooks), its HYPIR implementation and [AI Late to Class](https://www.youtube.com/@ailatetoclass)'s discoverable public material. Their recommendations identify candidates; model-owner terms decide inclusion. HYPIR/SUPIR and optional generators must not be assumed commercially clear because a tutorial is free. No paid creator workflow was obtained/copied and no exhaustive scrape of all GitHub was performed.

BFS was found during the expanded review after the user challenged the initial shortlist. The default dedicated head workflow was revised accordingly. The original author workflow uses 9B and optional nodes including background removal; we rebuilt a visible 4B graph without those unchecked dependencies. Preserved author model card, guide and exact revision/SHA are included in the dataset. Source: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap. Companion GitHub implementation reviewed as a candidate: https://github.com/Adeliox/klein-head-swap; it is not required by this ComfyUI graph.

To establish a quality winner, compare the same consented references/clean targets across profiles with fixed sizes/seeds; inspect likeness, pose, expression, lighting, hair, background drift and failures. Music-video use also needs frame-to-frame stability checks. This environment has CPU inference only, so no ranking is fabricated.
