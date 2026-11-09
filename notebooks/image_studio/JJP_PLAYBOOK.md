# JesusJuiceProductions — Image Production System

Adapted from the user's **JJP Launch System, Edition I**, supplied 9 November 2026. This is the image-production companion to that creative plan. It is not a new music-distribution plan, a promise of audience growth, or a claim that model rights have been cleared. The original PDF and private cast photographs are not included in this public repository.

## The production rule

One studio, recurring approved cast, changing cinematic worlds. Use live-action cinematic realism as the default. Human leads remain dignified and recognizable; antagonists are non-human and faceless. The Seeker remains faceless and cloaked. The tools provide prompts and records for those rules; they cannot guarantee a generator obeys them or certify identity fidelity.

Use the existing general workflows for unconstrained work. Use the new **JJP** workflows when making imagery for this anthology. No image or reference is generated, approved or published automatically at startup. Press Run only when the current graph and inputs are ready.

## Start a session

Import **Image_Studio_Kaggle_v6.ipynb**, attach the same standard pack, enable Internet and T4, then run its single startup cell. Optional FLUX3 still requires its existing separate pack and enable flag. The v5 missing-face-pack fix remains: missing face weights are prepared on CPU and the notebook attempts private Dataset submission with existing authentication. Existing model packs and sampling settings have not changed. No extra generation model is added by the JJP tools.

The studio installs five small JJP production nodes: episode brief, approve reference, load reference, combine two cast records, and artwork export. The graph remains visible and editable; the rest of the existing image pipeline is retained.

## Build your private cast library

1. Open **JJP - Approve Cast**. Upload a reference you have inspected and want to keep.
2. Choose Lead One, Lead Two, Accuser (non-human), or Seeker (faceless), give it a useful label, and write a stable prompt prefix describing the approved appearance, clothing and visual cues. The library starts empty; no appearance is invented from the PDF.
3. Inspect the image, set `approved=true`, then Run. The node refuses to save an unapproved reference. Aim for the PDF's 6–10 approved stills per character, with useful angles and lighting.
4. Save the notebook's outputs privately. The library lives under the studio output directory's `JJP/cast`. Attach that saved output in future sessions; the loader discovers its PNG/`.cast.json` pairs recursively. Refresh the browser/node options after adding references.

The approved PNG is checksummed. A later file change requires review and re-registration. Human approval in this library is your choice, not face-recognition evidence. Keep these personal reference files out of GitHub. The code never uploads the cast library as a public asset or automatically creates a Dataset containing your photos.

**JJP - Face Swap from Cast** uses an approved reference as the source identity. **JJP - Duet Face Swap from Cast** uses a Lead One reference and a Lead Two reference, applies them in two visible stages, and records both sources. Set target indices carefully: faces are ordered left to right starting at 0; the defaults assume two targets at indices 0 and 1. If a target is absent, the swap fails explicitly. The second stage re-detects the target after the first swap. A cast record and prompt prefix are not a guarantee of likeness.

InsightFace/InSwapper weights remain **non-commercial research only**. Their presence in a ministry/public-release plan does not grant commercial rights. Do not use these face-swap weights for a commercial release without suitable model authorization or a licensed replacement. This toolkit does not claim it has supplied such a replacement.

## Plan each still

Select **JJP - Create**, a specific episode graph, or the JJP version of restyling, inpainting, detail, face swap or upscaling. The **JJP Episode / Shot Brief** controls:

- Episode/world and palette, with custom world/palette/wing overrides for new songs.
- Shot purpose: episode key art, square cover, YouTube thumbnail, vertical still or cast reference.
- Scene-specific instruction, cast choice and reusable character prefix.
- Exact title and scripture reference.

A generation graph receives the composed positive/negative prompt and planned dimensions. Face-swap/upscale graphs retain their real input images and use the brief for context and export records; text prompts do not secretly alter a RealESRGAN or face-swap model.

The SDXL generation sizes remain around one megapixel: 1024 square, 1216×832 landscape or 832×1216 portrait. Those are source-art sizes, not claims of native 4K/3000px quality. FLUX3 retains its prompt enhancer and model-specific settings. Its optional model-access requirements are unchanged.

## The eight launch worlds

| Episode | World / palette | Default cast | Reference supplied by the PDF |
| --- | --- | --- | --- |
| Verdict of the Lamb | Cosmic courtroom; crimson on charcoal | Duet | John 19:30 |
| Anastasis | Courtroom to empty-tomb dawn; crimson to gold/green | Duet | Enter the intended passage |
| God-Breathed | Twilight battlefield and illuminated scrolls | Lead One | 2 Timothy 3:16 |
| Every Thought Captive | Neural war room to a river of light | Duet | Enter the intended passage |
| Present Your Body | Sunset hillside and starlit garden | Duet | Romans 12:1 |
| Jesus, Thank You For Loving Me | Intimate sunrise garden | Lead One | John 20:16 |
| Mirror of Glory | Golden desert and a transformative mirror | Duet, with faceless Seeker imagery | James 1:23–25; 2 Corinthians 3:18 |
| Love Your Enemies | Galilean hillside in a sun-shower | Lead Two; Lead One as distant listener | Matthew 5:45 |

These are editable source-document defaults, not machine-verified exegesis. Empty references stay empty instead of being invented. Use `use_episode_reference=false` to clear a preset reference, or type your own exact reference. The system does not fabricate a scripture quotation or silently select a translation.

## Choose the operation

**Create / Restyle / Inpaint / Detail:** compose a scene, retain approved character cues, and inspect the resulting human faces and non-human antagonists. Inpainting is editing, not an identity guarantee. The detail pass redraws content and can change faces.

**Upscale 2x / 4x:** use RealESRGAN for enlargement with fewer structural changes than a diffusion redraw. **Upscale Creative** uses the existing tiled SDXL process. **Restore SeedVR2** reconstructs detail with the existing native graph. These can alter fine detail or likeness. Inspect again after every enhancement; reject drift rather than calling a larger file approved.

**Face Swap:** use source and target photographs you have permission to use. The CPU workflow avoids taking generation's T4 VRAM. InSwapper's aligned internal face is 128px; a large final canvas is not proof of high-resolution facial detail. Two-character swapping is photo-only, not a video identity pipeline.

## Export one coherent delivery pack

The **JJP Artwork + Episode Export** stage works after every operation. It writes a unique run folder under `JJP/episodes/<episode>/<run>` containing:

- `master.png`: untouched pixels entering the export stage, with graph/brief metadata.
- `delivery.png`: optional deterministic brand/title/reference lettering and an export layout.
- `episode.json`: world, palette, cast/prefix, supplied scripture reference, actual dimensions, review notes, and selected cast provenance when connected.

Use square covers, YouTube 16:9 and vertical 9:16 as needed. Aspect conversion pads the image rather than silently cropping someone's face. `Original size` leaves the dimensions alone. The delivery resize uses Lanczos interpolation, clearly recorded as such; it does not pretend to be neural upscaling. Run a learned upscaler before export when you need learned detail. Check the actual upload specifications for the destination when preparing a final release.

Titles and verse references are drawn with a consistent font after generation, rather than asking the diffusion model to render lettering. The master is always retained. Face-swap exports default to lettering off; other JJP workflows can disable lettering or change layouts. No output is marked publication-approved automatically.

## A two-hour working rhythm

On planning day, select the episode and draft a handful of specific still briefs. On queue days, generate the clean art and review against the cast folder. On edit day, enhance only accepted candidates, then export the cover, thumbnail or vertical version you actually need. Keep a short list of rejected identity/palette examples so you do not repeat the same prompt failure.

The PDF's audio hook timing, Sabbath/rest choices, release calendar and platform posting are still editorial decisions outside this image notebook. Nothing here schedules a post or publishes an image to a social platform. The JSON record keeps the episode's visual intent available to your editor.

## Review and persistence

Before a release, verify likeness/dignity, non-human villain design, reference accuracy, spelling, image composition, output dimensions, model rights and the appropriate synthetic-content disclosure for the destination. Those are requested production checks from the supplied plan, not automated certification.

Save private outputs before ending Kaggle. Model packs do not back up the approved cast or finished art. The original PDF is not being pushed to the public repository. Existing v1–v5 notebooks remain available; import v6 for the added tools.

## Validation limits

Local checks cover brief presets, exact reference preservation, explicit cast approval, reference checksums, untouched master pixels, unique run folders, metadata, aspect ratios, live node schemas, browser graph loading and real ComfyUI export. Existing face/model tests cover their earlier local CPU paths. These checks are not live Kaggle/T4 inference, comparative image quality, automatic likeness certification or a successful real Dataset upload.
