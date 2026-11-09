# JesusJuiceProductions
## The Movie Creation Launch System

**Edition I · 9 November 2026 · Existing tools, one production process**

From an approved song and reference image to a finished cinematic episode, an upscaled master and platform-ready cuts. Built for one person, roughly two hours a day, and a workflow that protects Kaggle time.

**Start with one 15-second scene. Finish it, inspect it, save it. Then build the episode.**

Adapted from your supplied *JJP Launch System, Edition I*. Its faith-first mission, eight episode worlds, recurring cast and editing ideas are creative inputs. This guide does not change any notebook, workflow graph, model, sampler or runtime. Song timestamps below come from that PDF; they have not been checked against your actual audio files. Tool instructions come from the Movies repository as reviewed on this edition's date.

## 00 · Your first successful session

If you want the shortest route today, do these steps in order.

1. Pick **Verdict of the Lamb**, or the song you already have ready. Use your final audio file, not a temporary draft.
2. Choose a 15-second section. For Verdict, the PDF proposes **1:28–1:43**; listen to your file and check that this is the right passage.
3. Pick one approved reference still of the character in the scene. If needed, create it in Image Studio Commercial v7 first. Save the clean still without scripture or branding burned into it.
4. Open your **H3 Next T4** notebook. Attach the **Singularity, Molmo and Support** H3 Next Datasets. You do not need DenoMix for the Pika four-step workflow.
5. Select **GPU T4 ×2**, turn Internet ON, run the single setup cell and open its ComfyUI link. Choose **Singularity - Pika 4 Step Final**.
6. Upload the still in **Upload reference image**. Connect **IMAGE** to **reference_image** on **Prompt / size / seconds / optional reference**.
7. Enter one action and one camera move. For example: “The woman in <Picture 1> stands in a vast stone courtroom. She slowly lifts her gaze toward a shaft of warm light. A gentle camera push-in. Preserve her facial features and clothing. Cinematic live-action realism. No lettering.”
8. Set **seconds = 5**, keep the default 512 × 768 for this portrait test, leave the motion dial at zero, and press **Run**. Inspect the result before making more clips.
9. Generate two more simple shots. Cut usable parts together over the original 15-second song section in CapCut or DaVinci Resolve. Mute generated audio when using the finished song as the soundtrack.
10. Play the edit once without sound and once with sound. Reject face drift, broken hands, wrong emphasis or weak timing. Only then try upscaling the accepted footage using Section 07.
11. Save the accepted source clips, edited project, prompts, workflow JSON and final output before ending Kaggle. A saved model Dataset does not save your movies.

H3 Next is experimental. If it fails on your session, keep the diagnostics and return to your already proven Current Working workflow for the pilot. Do not spend the whole session repeatedly switching models.

## 01 · The master brief

**Identity:** JesusJuiceProductions is the studio identity. Faith and the meaning of the song lead; spectacle supports them.

**Mission:** make a viewer feel the lyric, recognize the scripture reference and want to read the passage. The movie must work as a scene, not just a sequence of impressive images.

**Visual language:** cinematic realism, physical materials, controlled camera movement and emotionally readable faces. The PDF's direction is no anime, clean language, no shame or condemnation, and non-human antagonists. Treat those as the visual constitution for these episodes.

**Recurring cast:** Lead One is you; Lead Two is your late wife, handled as a dignified memorial presence. Use only references whose use you have authorized and whose rights are understood. Reference conditioning does not guarantee an exact likeness. The faceless Seeker represents the viewer; darkness and the Accuser stay non-human.

**Capacity:** two hours a day is a planning budget, not a promise that a full music video can be produced weekly. Free Kaggle capacity, model storage, transfer time and session limits vary. Make the first release date depend on a finished pilot and a small buffer of accepted footage.

**Definition of done:** a coherent edit, approved likeness, checked audio sync, verified scripture, an archived master, platform versions, captions and a completed release checklist. An AI node reporting “completed” is only a render finishing.

## 02 · Which notebook does what?

All links below point to existing files in your GitHub repository. Import them into Kaggle; opening a notebook on GitHub does not run it. Use one ComfyUI setup at a time in a Kaggle session. Their ports and runtime environments can conflict if launched together.

| Existing notebook | Use it for | Important distinction |
| --- | --- | --- |
| [Image Studio Commercial v7](../../Image_Studio_Commercial_v7.ipynb) | Cast references, scene stills, identity edits, key art and thumbnails | Stills only. Use the current commercial profile, not the historical InsightFace face-swap profile. |
| [Current Working Workflow](../../Current_Working_Workflow.ipynb) | The baseline four-step H3 workflow | Earlier baseline renders succeeded in your supplied runs; that does not validate every later variant. |
| [H3 Next T4](../../H3_Next_T4.ipynb) | Singularity/Pika 4-step, Pika 2-step draft, DenoMix 8-step comparisons | All three accept text or one optional reference image. New GPU combinations remain unverified here. |
| [Pixa 5 Workflows](../../Pixa_5_Workflows.ipynb) | Text, first-frame image, first/last-frame image, speaking and singing | Use the audio workflows when supplied speech or singing is part of the shot. Do not assume arbitrary full-song lip sync. |
| [Director](../../Director.ipynb) | Timeline/local prompts and reference controls | Includes actual Three Step and PDD Eight Step Director canvases. This is the control-heavy option. |
| [Three-Step Workflow](../../Three_Step_Workflow.ipynb) | Simpler experimental TaoMate first-frame comparison | Not the full Director timeline workflow. Fewer steps do not guarantee better throughput or likeness. |
| [Movie Upscaler](../../Movie_Upscaler_Kaggle_v1.ipynb) | Existing video: Real-ESRGAN enlargement or SeedVR2 temporal restoration | Separate video finishing session. It currently uses GPU 0 only. |
| CapCut / DaVinci Resolve | Story edit, original song, captions, scripture, colour and final delivery | Use the editor for the film-length timeline; the H3 notebooks generate short shots. |

**Selection rule:** use the simplest workflow that supplies the control the shot needs. A quiet character shot usually needs one approved reference. A deliberate first-to-last pose needs Pixa First and Last Frame. A complex timed sequence may justify Director. Singing needs a short audio-conditioned test, not merely an image prompt containing “singing.”

Do not switch from your working renderer just because another preset says Draft, Final or Quality. Those are recipe names, not measured rankings on your T4s.

## 03 · Model packs: prepare once, attach only what you need

Models are the reusable tools. Cast photos, source songs and finished movies are project assets. Keep the two categories separate when choosing Kaggle inputs and saving outputs.

| Tool family | Preparation notebook | What to attach |
| --- | --- | --- |
| H3 Next, Pika 2/4-step | [H3 Next Model Packs](../../H3_Next_Model_Packs.ipynb) | Singularity + Molmo + Support |
| H3 Next, DenoMix 8-step | Same H3 Next pack builder | DenoMix + Molmo + Support; add Singularity only if using it too |
| Current Working / Pixa / Director / Three-Step | [H3 Model Packs](../../H3_Model_Packs.ipynb) | The compatible legacy ref2va, fl2va, encoder and support saved packs; attach all four for the collection's full selection |
| Image Studio Commercial v7 | [Commercial Image Model Packs](../../Image_Studio_Commercial_Model_Packs.ipynb) | create_edit + finish |
| Movie Upscaler | [Movie Upscaler Model Packs](../../Movie_Upscaler_Model_Packs.ipynb) | Its own verified video-upscaler pack; attach your source video separately |

For H3 Next automatic Datasets:

1. Import the model-pack notebook into Kaggle. Select **Accelerator: None**, **Internet ON**.
2. Enable your **KAGGLE_API_TOKEN** in Kaggle Secrets. Enter your Kaggle username in `DATASET_OWNER` if it cannot detect it. Never put the token into shared code.
3. Leave `AUTO_CREATE_DATASET=True`. Choose one `PACK` per preparation copy/session.
4. Run the cell. It measures working/temporary disk space, verifies the files, uploads a private Dataset and prints a link.
5. Wait for Kaggle processing to finish. In the renderer, use **Add Input → Your Datasets** to attach the required packs.

All four H3 Next packs total about 62 GiB; the older four H3 packs total about 73 GiB. These are separate collections, not four copies per movie notebook. Your account's actual storage allowance still applies. Do not rely on a fixed 50 GB temporary disk. Keep the preparation session open until upload succeeds.

The older **H3_Model_Packs.ipynb** uses **Save Version with output files → Notebook Outputs** rather than the new automatic Dataset uploader. Follow that notebook's current instructions; this guide does not silently add an uploader to it. Image Commercial and Movie Upscaler already have their own private Dataset creation paths.

An image SeedVR2 model is not automatically interchangeable with the Movie Upscaler's Q8 GGUF model. Attach the matching pack rather than renaming weights. Source paths under `/kaggle/input` are read-only.

## 04 · Cast, continuity and the episode folder

Create one private folder per episode. Keep the cast source folder separate so you can reuse the approved references across worlds.

Suggested structure:

- `EP01_Verdict/00_brief/` — episode card, shot list, checked lyric/scripture notes.
- `EP01_Verdict/01_audio/` — approved song master and exact source filename.
- `EP01_Verdict/02_references/` — approved scene stills and cast records.
- `EP01_Verdict/03_renders/` — original clips, exported workflow JSON, prompt, seed and logs.
- `EP01_Verdict/04_edit/` — editor project and linked-media list.
- `EP01_Verdict/05_upscale/` — accepted restored clips, source mapping and chunk receipts if needed.
- `EP01_Verdict/06_delivery/` — clean master, branded versions, subtitles, thumbnail and captions.
- `EP01_Verdict/07_records/` — permissions/provenance, review checklist and release results.

Name a take `EP01_SH010_T01`. Keep the same shot ID in the CSV, prompt notes, editor timeline and upscale log. A rejected take stays marked rejected; do not accidentally promote it just because it has a larger resolution.

Build a private cast board with a front, three-quarter, profile and restrained-expression reference for each lead. Six to ten approved stills is a useful target from the PDF, not a requirement before the first test. Fix hair, apparent age, wardrobe, jewellery and lighting direction for each scene. For two leads in one shot, prefer an approved two-person scene still so their placement is already established; check both identities separately.

Use Image Studio's approved-cast records where already available. The video notebook consumes the selected image, not an automatic cast database connection. Export the still and upload/attach it explicitly. Do not claim the image studio supplies temporal face tracking or video face replacement.

**Approve the clean scene first.** Add text later in the editor. Generated lettering and a scripture overlay inside a reference still can deform during motion.

## 05 · Plan a movie as short shots

Your song is the timeline. Mark the final audio's verses, impacts, silences, instrumental windows, key changes and ending before generating footage. The PDF's energy-map descriptions are a starting point; no original energy-map JSON or audio files came with this request, so this guide does not invent measurements.

For every shot write: **who, where, one action, one camera move, emotional purpose, duration, reference and exact place in the song**. Use the templates in this folder. Start with simple actions: turning toward light, a hand opening, cloth moving, one step forward. Separate a transformation and a camera orbit into different shots if the first test cannot handle both.

**Prompt pattern:** “The [person] in <Picture 1>, [location and light]. [One visible action]. [One camera move]. Preserve [key identity/wardrobe details]. [Emotional tone and physical style]. No lettering.” H3 Next uses one optional reference; the Director and Pixa interfaces have different controls. This is a writing pattern, not a replacement node or universal prompt syntax.

Make a coverage set: one establishing shot, two medium performances, two close details and one transition. Reuse approved environments and cutaways intelligently. A full six-minute song does not require every second to be newly generated motion: use intentional still holds, slow editor moves, silhouettes, detail cutaways and repeated visual motifs where they serve the song.

**Reality check on workload:** 360 seconds divided by five seconds is 72 generated shots before retries. At an illustrative four minutes per shot, that would be 4.8 hours of generation alone. That is arithmetic, not a benchmark for these models. Track *accepted footage per GPU hour*; a two-step clip that needs five retries may cost more than a four-step keeper.

For landscape H3 Next shots, 768 × 512 stays within the existing pixel cap; its portrait default is 512 × 768. These are 3:2 and 2:3 working frames, not exact 16:9 or 9:16 delivery. Plan crop-safe framing and crop/pad in the editor. Do not stretch faces or raise source resolution beyond the notebook's guard to get a platform ratio.

## 06 · Generation session: protect your Kaggle time

Before starting GPU time, have the song segment, references, prompts and model Datasets ready. Importing a new GitHub notebook is necessary only when you want its updated code; a GitHub push does not update an already imported Kaggle copy.

1. Attach only the model packs for that notebook plus the project media it needs.
2. Run setup once and wait for the ComfyUI link. Model checking/downloads can show little GPU use; network and disk work are not accelerated by the GPU.
3. Load the named visible workflow from **Workflows**, not a stale tab from an older import.
4. Render one representative reference shot at the normal safe settings. Inspect face, hands, clothing, camera motion and background.
5. Once it passes, generate the next planned shot. Keep the accepted graph and seed with each take. A fixed seed helps comparison but does not guarantee character identity.
6. Record generation time and accepted seconds. Compare recipes using the same reference, action, duration and dimensions; change one setting at a time.
7. If a job fails, capture the error/log/diagnostic ZIP before restarting. Use the existing saved preparation/latent recovery where the notebook supports it. Do not repeatedly Run All or reinstall models to solve a bad prompt.
8. Save project outputs before ending the session. The temporary Cloudflare link is access to the live session, not permanent storage.

H3 Next separates prompt/reference preparation, sampling and decoding into different processes. Its dual-T4 sampling does not imply equal GPU usage during every phase. Movie Upscaler currently uses only GPU 0. GPU utilization screenshots alone cannot tell you whether a job is incorrectly configured.

For **Pixa Singing/Speaking**, supply your own short audio segment and reference in the visible loaders. Audition lip timing and expression before committing to a longer sequence. For other cinematic shots, preserve your original song in the editor and mute generated audio. Do not replace a finished song with H3-generated audio by accident.

For **Director**, first save a five-second test with one local prompt and one intended reference change. Inspect the active reference branch and timing before expanding the sequence. The supplied timeline interface remains the source of control; this guide does not convert it into an automatic full-song director.

## 07 · Movie upscaler: finish accepted footage

Upscale after you know which takes you will use. Enlargement cannot reliably repair wrong identity, missing fingers, failed movement or poor story choices. It can amplify those defects. Begin with one five-second representative clip, including a face, fabric and motion.

**Two choices already installed:**

- **Movie Studio - RealESRGAN:** a simpler frame-by-frame enlargement path. Try it first for a clean source. It may flicker because it does not model temporal consistency.
- **Movie Studio - SeedVR2:** temporal restoration with chunk/context handling. Try it when the source needs restoration, then compare against the original for invented texture, changed eyes and boundary flicker. No measured speed/quality winner is claimed here.

**Step by step:**

1. Prepare **Movie_Upscaler_Model_Packs.ipynb** on CPU once; wait for its private Dataset to finish processing.
2. Import **Movie_Upscaler_Kaggle_v1.ipynb**. Select a T4 GPU and Internet ON. Attach the upscaler model Dataset and your source video.
3. Run its setup cell and open the printed **MOVIE STUDIO** link. Select one of the two workflows.
4. In the leftmost source node, set `source_path` to the exact full path of your video in that session. A path from a different notebook session will not work unless you saved/attached or copied the file into this session.
5. Start at **2×**. Keep `chunk_frames=17`, the existing context settings and longest-edge limit **1920**. Source input must be square-pixel, constant-frame-rate SDR video. Convert HDR, variable-rate or rotated video explicitly in the editor first.
6. Keep frontend **Auto Queue OFF** and queue count **1**. Press **Run once**. The save node queues the next chunk until the job is finished.
7. To pause, **Interrupt**, then **Clear Queue**. Run the same source/settings again to reuse verified chunks. Set `auto_continue=false` only if you want to run each chunk manually.
8. Open the finished movie and inspect the start, middle, end and chunk transitions. Compare at normal playback and at full size. Check facial likeness and audio sync; confirm the export report rather than assuming a completed node means the whole film is complete.
9. Save the final video and any checkpoint folder you need before ending the session. The documented job location is `/kaggle/working/movie-studio/movie-jobs/`; use the path printed by the running notebook if its configuration differs.

For **RealESRGAN 4×**, both controls must match: source `scale=4` and `ImageScaleBy.scale_by=1`. The default 2× route runs the 4× model then downsamples by 0.5. The target-edge cap still applies; mismatched dimensions fail before committing a chunk. Do not raise the cap merely to label the result “4K.” A T4 fit or 4K throughput has not been established.

SeedVR2 uses its existing Q8 model, FP16 tiled VAE and offloading settings. Tiling the VAE does not make transformer memory unlimited. Large edits can exceed disk or session limits; upscale the chosen clips or a short finished section before considering the full song.

**Sound and files:** the pipeline checks frame count, rate, dimensions and audio-stream count. It attempts original audio stream copy; MP4 may fall back to AAC. MKV can retain supported subtitles. Video is re-encoded; this is not a lossless final-delivery promise. Keep the approved audio master separately and check it again in the final editor export.

**Resume between sessions:** preserve the whole job folder with its source mapping, lossless chunks and receipts, then restore it to the configured job location. A model Dataset contains weights, not these jobs. Changing restoration settings creates a different job. Keep source files and existing jobs; do not delete them to free space during an unfinished run.

## 08 · Edit, sound, scripture and delivery

Build the story at the source resolution first. Use the original song master as the timing authority. Cut where meaning and performance support the beat; do not force every low-energy passage into fast edits. Respect a hard stop, false ending, decay or whisper fade. The PDF suggests silence-to-impact hooks, but not every passage needs the same trick.

After selecting footage, upscale the chosen clips and relink carefully, or upscale a short clean picture edit. Add typography, subtitles, logos and verse references **after** neural upscaling so the model cannot redraw them. Preserve an unbranded clean version. Compare the conformed edit against the original timeline for shifted in/out points and audio drift.

Check every reference against the final lyrics and chosen Bible translation. Put a readable verse reference at the intended moment; place longer text in captions or description when it would cover the performance. Do not label a song **WORD 4 WORD** until every sung line has been compared with the stated translation. The PDF's descriptive lyrics alone do not establish that claim.

The source PDF mentions -14 LUFS and -1.5 dBTP. Treat them as a proposed music-delivery approach, not evidence that your current file has been measured or a universal video-platform requirement. Avoid repeated loudness processing and clipping. Listen to the final export on headphones and a phone speaker, especially around silences, loud peaks and fades.

Make these deliverables from one approved timeline:

- **Clean master:** highest useful accepted picture quality, original approved soundtrack, no platform stickers or watermark.
- **YouTube episode:** planned landscape framing, readable titles, description and checked chapter times. A 1920 × 1080 timeline can include crop/padding; export dimensions alone do not prove native detail.
- **Vertical cut:** 1080 × 1920 delivery if useful for your source, framed for the faces and action. Review platform UI safe areas at upload time. Avoid blindly cropping a two-person wide shot.
- **Thumbnail/key art:** use the approved Image Studio output, not a distorted paused AI frame. Match the episode's palette and title treatment.
- **Caption package:** song title, one honest context line, verified scripture reference and the real full-episode/song link. Do not invent a release URL or audience claim.
- **Optional Spotify Canvas:** a short loop from an accepted moment; check Spotify's current duration, aspect and content requirements before submitting.

Uploading, account creation, distribution and scheduling are separate actions you perform. This guide does not publish to any platform or claim that a handle, model license or distributor policy has been verified today.

## 09 · Eight episode production cards

These are adaptations of the supplied PDF, not audio analysis. Hook times, durations and ending descriptions must be checked against the exact approved master. References are starting points for review; empty fields in the planning CSV mean “choose and verify,” not permission to invent a quotation. Use the full cards in `templates/episode_cards.csv` alongside this section.

### EP01 · Verdict of the Lamb

**World:** cosmic courtroom; crimson/charcoal, monumental arches and restrained ivory light. Duet. The Accuser is faceless smoke, never a human villain. **Ending:** hard stop. PDF duration: **6:27**.

**Proposed windows:** 1:28–1:43; 4:22–4:37; 5:25–5:40. **Reference to verify:** John 19:30.

**Pilot coverage:** wide courtroom before the impact; close portrait looking toward light; chains released as a detail shot. Make the 15-second passage from three modest actions rather than a single complicated transformation. Keep the final cut decisive; do not attach an extra fade to the hard stop.

### EP02 · Anastasis

**World:** the same visual universe opening into an empty-tomb garden at dawn; crimson becomes gold/green. Duet, no Accuser. **Ending:** joy-fade. PDF duration: **7:32**.

**Proposed windows:** 3:35–3:50; 4:22–4:37; 5:38–5:53. Select and verify the passage before adding scripture text.

**Pilot coverage:** the second lead turns from shadow to dawn; light crosses the tomb entrance; both leads in a calm approved wide composition. Let joy read through expression and light before adding movement. The source's tagline can be an editorial end card; do not present it as a verbatim Bible quotation.

### EP03 · God-Breathed

**World:** twilight battlefield, broken swords and illuminated scrolls. Lead One alone. Charcoal, crimson and warm light. **Ending:** false end then hard stop. PDF duration: **7:52**.

**Proposed windows:** 2:26–2:41; 3:31–3:46; 6:05–6:20. **Reference to verify:** 2 Timothy 3:16.

**Pilot coverage:** warrior-scribe holds a closed scroll; a low detail shot of broken metal; a grounded medium performance. Add readable words in post rather than asking H3 to spell on moving paper. Preserve the false-ending pause in the edit.

### EP04 · Every Thought Captive

**World:** symbolic mind war room, indigo/cyan neural grids and faceless glitch entities, resolving into white calm. Duet. **Ending:** slam into peace. PDF duration: **5:48**.

**Proposed windows:** 3:14–3:29; 4:11–4:26; 5:29–5:44. Select and verify the scripture passage.

**Pilot coverage:** still close face against a controlled moving background; a separate architectural collapse or light transition; quiet water. Keep glitch effects away from the face unless deliberately approved. Let the calm ending breathe instead of continuing the stutter-cut rhythm.

### EP05 · Present Your Body

**World:** sunset hillside to starlit garden; gold, green and warm night blue. Duet. **Ending:** false end with organ fade. PDF duration: **6:08**.

**Proposed windows:** 1:56–2:11; 3:51–4:06; 5:21–5:36. **Reference to verify:** Romans 12:1.

**Pilot coverage:** a hand opens beside loosened shackles; one gentle step through the garden; bloom or cloth detail. A baptism sequence needs separate carefully planned shots; do not expect one five-second render to handle a complex underwater transition and exact facial consistency.

### EP06 · Jesus, Thank You For Loving Me

**World:** intimate empty-tomb garden at sunrise; soft gold/ivory. Lead One. No armies or towers. **Ending:** whisper fade. PDF duration: **4:33**.

**Proposed windows:** 2:01–2:16; 2:48–3:03; 3:56–4:11. **Reference to verify:** John 20:16, in its actual scriptural context.

**Pilot coverage:** held expression, a restrained push-in and light crossing the garden. Use slower dissolves and stable framing. The optional distant female silhouette in the PDF is a creative choice, not a mandatory representation of your wife or an asserted biblical identity.

### EP07 · Mirror of Glory

**World:** golden desert and a solitary mirror; the Seeker is faceless and cloaked. Duet. **Ending:** decay. PDF duration: **6:09**.

**Proposed windows:** 2:31–2:46; 3:45–4:00; 5:16–5:31. **References to verify:** James 1:23–25; 2 Corinthians 3:18.

**Pilot coverage:** wide mirror and cloak movement; reflection-light insert; reaction close-up. Treat mirror/identity continuity as a difficult shot: test it, or use an editorial cut between approved images rather than insisting on a perfect continuous transformation.

### EP08 · Love Your Enemies

**World:** Galilean hillside in a sun-shower, rain and light on the entire scene. Lead Two carries the performance; Lead One is a distant silent listener if used. **Ending:** echo fade. PDF duration: **3:37**.

**Proposed windows:** 1:24–1:39; 2:27–2:42; 3:07–3:22. **Reference to verify:** Matthew 5:45 and the actual passage used in the lyrics.

**Pilot coverage:** rain on foliage; a steady second-lead portrait; a wide shot that includes the quiet listener without losing the scene's intimacy. Verify any WORD 4 WORD claim line by line; do not inherit it automatically from the PDF. Keep the echo tail.

## 10 · Weekly rhythm and the 90-day launch

Use the PDF's recurring rhythm as a capacity guide. Rendering sessions need monitoring, available quota and saved outputs; do not assume they can run unattended indefinitely.

- **Monday, 2 hours:** approve the song passage, make the episode card, prepare references and six shot briefs.
- **Tuesday, 2 hours active work:** render the representative shot, check it, then work through the accepted shot list. Record actual GPU time separately.
- **Wednesday, 2 hours:** review takes, repair only the shots the edit really needs, archive accepted clips.
- **Thursday, 2 hours:** assemble the short sequence; approve timing and identity; upscale a representative accepted segment.
- **Friday, 2 hours:** finish sound, scripture and delivery; publish only if the release checklist passes. Otherwise use this as a finishing day.
- **Saturday, 2 hours:** extend the full episode or prepare the next short. Save a clean master and project backup.
- **Sunday:** rest, or your chosen rest day. A brief next-week plan is optional; the schedule should survive a difficult week.

**Days 1–30: establish the process.** Complete one 15-second pilot from reference to upscaled delivery. Build the cast folder and archive routine. Finish the first episode to an achievable standard; prepare Anastasis as the next planned chapter. Record time, failure rate and accepted seconds before promising weekly films.

**Days 31–60: improve what the first release taught you.** Develop the next episode and a small library of reusable environments/cutaways. Compare one rendering recipe and one upscale route on the same source. Build a one-release buffer if capacity allows. Reuse an accepted scene in different editorial cuts without pretending it is new footage.

**Days 61–90: expand only what is stable.** Grow the anthology across the eight worlds. Assemble a coherent compilation when the accepted material supports it, not simply because day 90 arrived. Plan lyric/verse-led shorts and behind-the-scenes posts that explain the real production process.

Start with one or two strong shorts per week and a full episode every two to three weeks *if actual capacity supports it*. Increase only after maintaining the archive and quality checks. These are proposed operating targets, not promises of followers, streams, revenue or platform distribution. The PDF's numeric audience targets are not forecasts.

## 11 · Release review, provenance and respectful use

Use `templates/release_checklist.md` before each release. Verify scripture in context, spelling, translation permissions where applicable, actual song rights, source-media permissions and the terms of the specific model/LoRA versions you used. Open weights and a notebook titled Commercial do not automatically clear the whole film, a likeness or the soundtrack.

The Image Commercial v7 repository contains a component license audit; consult it for that still pipeline. This adaptation did not audit every H3, Pika, Molmo, Singularity, DenoMix or Director model license for commercial release. Record model provenance and check applicable terms before distributing. Do not carry forward the PDF's “clean by design” claim as blanket clearance.

Handle the memorial portrayal with the dignity described in the brief. Avoid misleading viewers into believing synthetic performance is archival footage. Use platform-required synthetic/altered-content disclosures and check current upload forms and policies. Confirm relevant permissions and estate/family considerations rather than assuming ownership of every likeness right.

No forced shame, fabricated quotations, bought engagement or unsupported results claims. A caption can say “AI-assisted cinematic interpretation” without diminishing the song's meaning. If a scene is symbolic rather than a literal reconstruction, frame it honestly.

## 12 · Measure, troubleshoot and preserve

Track production metrics first: planned shots, accepted shots, retries, accepted seconds, generation minutes, upscale minutes and total active editing time. Later add real viewing data: retention at the opening, completion, meaningful comments and clicks. Compare like-for-like formats; no metric overrides the mission.

| Symptom | First action | Avoid |
| --- | --- | --- |
| Old 19.53/19.50 GiB pack assertion | Check you imported the corrected H3 Next pack notebook; it prints Storage checks | Repeatedly running the same old Kaggle copy |
| No nodes or wrong canvas | Open Workflows and choose the named full workflow | Assuming the Cloudflare link alone selects the right old tab |
| Face changes between takes | Return to the approved reference, simplify action and compare both faces | Upscaling a rejected identity |
| Pixa has no speech input | Upload your short recording and select it in the audio loader | Relying on a missing example file |
| Upscaler stops after a chunk | Check errors, queue state, disk and auto_continue | Enabling frontend Auto Queue and duplicating jobs |
| Source path missing | Attach/copy the real source into this session and use its full path | Reusing a path from an ended session |
| Model Dataset exists but clips are gone | Restore your saved episode outputs/project backup | Expecting model weights to contain generated footage |
| One GPU is idle | Identify the phase and notebook; Movie Upscaler is single-GPU | Assuming all phases pool both T4s |

Before ending any GPU session, confirm your accepted footage and project records exist outside temporary storage. Keep at least one durable copy you have checked by opening it. Preserve needed upscale checkpoints if work is unfinished. The final handoff is **clean master + delivery versions + project + source media + references + graph/settings + provenance + review record**.

## Appendix · Source, links and validation boundary

Creative source: user-provided `jjp_launch_system.pdf`, Edition I, received for this adaptation. Its eight episode titles, proposed hook windows, worlds and ending dialects are attributed planning inputs. The supplied document's original audio-analysis assets, claimed rights confirmations, handle availability and platform-policy claims were not independently established here.

Technical sources within the repository:

- [H3 Next operation and limits](../H3_NEXT_T4.md)
- [All four H3 collections](../H3_NOTEBOOK_COLLECTIONS.md)
- [Movie Upscaler operating guide](../../notebooks/movie_upscaler/intro.md)
- [Movie Upscaler implementation and validation](../../notebooks/movie_upscaler/README.md)
- [Commercial Image Studio guide](../../notebooks/image_studio_commercial/README.md)
- [Commercial image component license audit](../../notebooks/image_studio_commercial/LICENSE_AUDIT.md)
- [Printable PDF](JJP_MOVIE_LAUNCH_SYSTEM.pdf)

This delivery adds planning documents and reusable templates only. It performs no model download, render, upload to a media platform or change to any notebook. Local validation covers the document links, generated PDF and template structure. Existing GPU, memory, rendering-quality and live Kaggle limitations remain as documented by each tool.
