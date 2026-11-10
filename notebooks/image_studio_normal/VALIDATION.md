# Normal Image Studio validation

Scope: remove the production UI and restore ordinary image workflows, retaining the reviewed models and saved-model setup.

Reviewed GitHub's historical `Image_Studio_Kaggle_v1.ipynb` and original `notebooks/image_studio/build.py`: ordinary Create/Restyle/Inpaint/Detail and later face/upscale graphs were the original interaction model. The new normal UI uses prompt/upload/run/save without project or episode stages. The commercially reviewed Klein/BFS/ESRGAN/SeedVR2 model set remains unchanged; no historical restricted weights are reintroduced.

Verified locally:

- Six tests confirm only the 10 normal workflows, standard SaveImage outputs, no production fields/nodes, exact unchanged catalog/face support, byte-identical original upscale graphs, direct prompt/dimension controls, and preservation/archival of old managed menus without changing personal graphs.
- Four notebook/graph checks pass: one executable cell compiles, graph connectivity/reachability, live registered node schemas, and embedded workflows equal source.
- A real pinned ComfyUI CPU server loads only `commercial_image_tools`. None of the former production custom nodes are enabled. Chromium loads all 10 workflows and serialises them into API prompts. Prompt strings, width/height connections, steps=4, CFG=1, source/target reference order and the BFS 4B adapter/strength are checked. The creation canvas was visually inspected.
- Real Real-ESRGAN CPU inference through ComfyUI produces 2× and 4× PNGs from a small synthetic source. The face compositor produces a saved PNG through the API using public-domain test portraits. These tests check execution/geometry, not photographic or likeness quality.
- The dataset catalog, saved-model resolver, face/masked editing implementation and original upscaling graphs are retained. The working Dataset creator, upload method, credentials and models are not changed by this UI correction.

Not established here: successful full FLUX/BFS/SeedVR2 GPU inference, T4 speed/peak memory, user-specific likeness, comparative quality or live Kaggle/Cloudflare access. “Cinematic” is an editable starting prompt and landscape composition, not a measured universal quality ranking. The simple pixel compositor remains a limited fallback.

Rebuild: `python notebooks/image_studio_normal/build.py`. Against the pinned CPU ComfyUI server run `test_normal.py`, `check.py`, `browser_check.py`, `integration_check.py` and `test_face_api.py`. Playwright/Chromium are required only for the browser check. The build never regenerates a model Dataset.
