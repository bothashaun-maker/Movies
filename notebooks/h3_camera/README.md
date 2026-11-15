# AIQuest H3 — independent saved-Dataset edition

This pair retains [AIQuest's actual Kaggle engine and Gradio interface](https://github.com/TeamAIQ/Colab-notebooks/blob/d3e016968428318f163912d582cecf172c37413d/kaggle_notebooks/minimax-h3-turbo-int8-video-audio.ipynb). It does not adapt that engine into ComfyUI. AIQuest's credit and original interface are retained.

1. Run [JJP_H3_Camera_Models.ipynb](../../JJP_H3_Camera_Models.ipynb) with Accelerator None and Internet ON. Enable the same Kaggle Secrets used by the working builders. It automatically submits one complete private Dataset; no manual Save Version or PC upload is required.
2. Wait for Dataset processing. Attach that one Dataset to [JJP_H3_Native_Camera.ipynb](../../JJP_H3_Native_Camera.ipynb), select T4 x2 and Internet ON, and run its single cell.
3. Use **Start frame** for an image-anchored shot. The default recipe is AIQuest's native Qwen GGUF + FL2VA pruned INT8 + PDMD four-step adapter. The model stays resident between clips. Startup still needs one model load.

The 13 model/config files total 37.956 GiB. The CPU saver uses the existing native one-file-at-a-time upload method; peak missing-file storage is 19.611 GiB plus reserve. It measures scratch capacity and never deletes attached sources. Original file licenses remain in force. All files, revisions, exact sizes and SHA256 values are in `assets.json`.

The renderer verifies one matching Dataset manifest, stages/hashes its files once onto measured local scratch, and performs AIQuest's original setup. It does not download model weights. Its optional adapters are limited to the saved Camera Motion and 360 Orbit files. The latter's publisher recipe uses 28 steps and 73 frames; four-step compatibility is not established here. This is a PDMD Dataset, not a pack for the source notebook's other three variants.

`integrate.py` verifies the upstream notebook's Git blob, extracts its engine and checks that the inference and UI AST remain unchanged. Only the camera catalog and local-file resolver differ. The launcher changes scope cleanup to this app's own recorded process. Generated notebook payloads are self-contained.

The current shot's short prompt uses [Pixaroma's first-frame format](https://workflows.pixaroma.com/gems/). It is embedded in the GPU notebook. Structured prompts pass through AIQuest's builder unchanged. No claim is made that wording guarantees an exact camera path.

Local validation covers notebook/payload compilation, model manifest integrity, source preservation, structured prompt passthrough, complete-Dataset selection, corrupt-file rejection, startup handoff and native Kaggle SDK file-token/private-create behavior. Remote upload/storage calls are stubbed; Windows tests substitute hardlinks for Linux symlinks and do not exercise Linux file locking. A live authenticated upload, full model load, T4 memory fit, render timing and motion quality have not been tested for this edition.

Rebuild with `python notebooks/h3_camera/build.py`. Run `python -m unittest discover -s tests -p test_h3_camera.py -v` with nbformat and the native Kaggle SDK installed. Existing notebooks are not regenerated.
