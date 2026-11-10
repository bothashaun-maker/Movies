# Image Studio — Normal v10

**Create images. Edit images. Face/head swap. Upscale.**

Attach your existing **ONE ALL MODELS Dataset**, select a Kaggle T4 GPU and enable Internet for software/tunnel setup. Run the single startup cell, open the printed ComfyUI URL, choose **Image Studio - …** from the workflow menu and press Run. No image starts automatically. Your model dataset, authentication and dataset creator are unchanged. No model download or Dataset creation runs in the studio.

| Workflow | What you do |
|---|---|
| Create - Cinematic | Write your prompt, set width/height and seed, then Run. Defaults to 1216×832 landscape; use multiples of 16. |
| Image Edit | Upload a picture and describe the change. |
| Inpaint - Masked Edit | Upload a picture, right-click → MaskEditor, paint white over the area to replace, then describe the change. |
| Face Swap - Reference Edit | Target is image 1; source identity is image 2. Describe which person changes. |
| Face Swap - Masked | Same inputs, with a painted replacement mask to keep changes local. |
| Head Swap - BFS | Target is Picture 1; identity is Picture 2. This dedicated adapter can replace head and hair. |
| Face Composite - Simple | Transfer source pixels onto a similar-pose target. Choose face indices (0 = leftmost). Simple fallback with possible visible edges. |
| Upscale 2x | Upload, enlarge to twice the dimensions, save. |
| Upscale 4x | The same Real-ESRGAN x4plus workflow as before. |
| Restore SeedVR2 | Upload a still for reconstruction/upscaling; inspect invented detail. |

Prompt, dimensions and seed are ordinary visible nodes. Results save as PNGs with graph/settings metadata, under `/kaggle/working/image-studio/images`. There are no episode presets, cast-approval stages, scripture fields, lettering or branded exports. Older app-managed workflow files are backed up outside the workflow menu. Your source photos and existing output images are not deleted.

The reviewed models are unchanged: Klein 4B distilled generation/reference editing, BFS 4B adapter, Real-ESRGAN and SeedVR2, with the optional simple landmark compositor. Klein starts at four steps/CFG 1; increasing steps is not a proven quality upgrade for this distilled model. Cinematic quality depends on the prompt, scene, references and chosen output; no measured “best model” claim is made. Reference editing and head replacement can change likeness and scene details. Masked edits are generated candidates pasted within your mask/feather, not a separate trained inpainting model. The upscaling graphs are reused unchanged.

Keep the cell running while using ComfyUI, keep the session URL private, and Save Version with outputs before ending the session. The existing commercial model license notices/provenance are preserved. Full GPU inference and T4 quality/speed remain unverified here; validation results are in VALIDATION.md.
