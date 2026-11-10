# ALL MODELS — one CPU-built private Dataset

Use **Image_Studio_All_Models_Dataset.ipynb**. It keeps the confirmed working Model Packs v3 direct private Dataset creation method and selects `PACK='all'` by default. No new authentication method, read-permission prerequisite or Dataset version/update operation is introduced.

1. Use CPU + Internet. **No Dataset Inputs are required for this builder.** It obtains all eight reviewed models. Any matching files already present or attached are reused automatically.
2. Run the single code cell. All eight pinned, commercially reviewed models go into one flat `all` pack. Verified attached files are copied into durable output; files already in that output are reused. Only missing weights are downloaded.
3. The builder submits **one private ALL MODELS Dataset** using your existing Kaggle credentials. Wait for Kaggle processing, then attach that single Dataset to **Image_Studio_Commercial_v9.ipynb**. Save CPU notebook outputs as a backup.

The complete pack contains FLUX.2 Klein 4B, Qwen3-4B text encoder, Flux2 VAE, Real-ESRGAN x4plus, SeedVR2 model and VAE, Google face landmarks, and the BFS 4B head-swap adapter. Total model bytes are about **20.53 GB (19.12 GiB)**, plus small manifests/license files. The builder checks actual free space for the full output plus 512 MiB reserve before transfers. Your account's Dataset/output quotas still apply; the notebook cannot change them. Uploading uses hardlinks to avoid another full-sized staging copy.

The Dataset contains model files. The studio notebook contains all 24 workflows and installs them into ComfyUI. No InsightFace swap/recognition weights or old Juggernaut/FLUX3 generators are added to this commercial pack.

The studio now **only finds and verifies saved models**. It does not prepare/download model packs or publish Datasets. Software installation and the tunnel still require Internet. Missing inputs stop startup before model downloads. The intended and documented setup is exactly **one ALL MODELS Dataset input**.

Existing credentials, private creation, safe error reporting, SHA256 verification, resumable downloads and license notices are retained. A successful receipt skips resubmission of that same saved pack. Existing Datasets are never deleted or overwritten. The v3 method worked in your Kaggle session; live publication of the new combined pack remains unverified here.
