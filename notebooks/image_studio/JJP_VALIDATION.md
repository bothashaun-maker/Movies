# JJP v6 validation and source trace

The user supplied **jjp_launch_system.pdf**, Edition I. Its creative rules were translated into editable image-production presets and nodes, rather than treating its publishing instructions as permission to post to social platforms. The PDF, private biography and personal cast images are not copied into this public repository.

Implementation mapping:

| PDF requirement | Implemented artifact |
| --- | --- |
| Cinematic per-song worlds, no anime | Shared brief, negative prompt and eight episode/world/palette presets |
| Recurring approved cast; 6–10 stills per character | Explicit approved-reference save/load with roles, reusable prompt prefixes, PNG checksums and private saved-output discovery |
| Two human leads, non-human antagonists, faceless Seeker | Brief rules and cast-role metadata; not claimed as an automatic image classifier or guaranteed generator behavior |
| Duet, solo Lead One, solo Lead Two | Episode defaults plus editable cast selection; one-identity and two-identity face-swap graphs |
| Open wing system for future songs | Custom episode with world, palette and wing override fields |
| Consistent covers and scripture references | Deterministic post-generation lettering, exact supplied reference string, square/16:9/9:16 padded delivery |
| Reusable episode folder | Untouched master PNG, delivery PNG, embedded graph/brief, episode JSON and cast provenance where connected |
| Two-hour solo workflow | Companion playbook and no automatic render, approval or publishing |
| $0-oriented Kaggle tooling | Existing model packs and optional profiles retained; no new model weights needed by JJP nodes |

The system does not implement audio analysis, energy-map extraction, Spotify upload, release scheduling, royalty rights verification, automatic Bible exegesis, guaranteed character identity, or a commercially licensed face-swap replacement. Those were not silently inferred from the document.

Local verification:

- Nine JJP unit checks: episode defaults, future custom wing, exact/empty reference handling, explicit cast approval, reference integrity/round-trip, untouched masters with metadata, unique outputs/raw delivery, two-role cast metadata and three delivery aspect ratios.
- Four existing graph/notebook checks: graph links/reachability, live input schemas, generated code validity and embedded graph consistency.
- Nine startup/model-preparation checks and seven existing pack checks, using small synthetic files/local HTTP and mocked Dataset creation.
- All **31** original/JJP/episode graphs load in actual ComfyUI Chromium with registered node classes. Browser checks validate widget mapping and graph-to-prompt links for the JJP preset controls.
- Real local ComfyUI queue execution verifies brief → lettering/export, unchanged master pixels, exact title/reference, dimensions and episode JSON.
- Real CPU inference with public-domain NASA and US Navy portraits verifies approved cast save/load → two sequential InsightFace swaps → clean delivery at target dimensions, with both cast records saved. It is an integration test, not a likeness benchmark. Test portraits are not bundled as approved user references.

Not verified: full Kaggle/T4 startup or generation, optional FLUX3 inference, live private Dataset upload/processing, user-photo fidelity, final image quality or platform publication. Face weights remain non-commercial research only. Output exports are marked for human review, not automatically approved.

Rebuild from repository root:

```sh
python notebooks/image_studio/build.py --output Image_Studio_Kaggle_v6.ipynb
COMFY_URL=http://127.0.0.1:8189 STUDIO_NOTEBOOK=Image_Studio_Kaggle_v6.ipynb python notebooks/image_studio/check.py
python notebooks/image_studio/test_assets.py
python notebooks/image_studio/test_startup_models.py
```

JJP tests require the local Torch/Pillow environment and matching ComfyUI source; API/browser tests require a running server and the fixture/model setup described by the scripts. Standalone regression tests expect the built notebook beside their sources unless STUDIO_NOTEBOOK overrides the graph test. No new dependency installation is required for the JJP nodes beyond the existing studio stack.
