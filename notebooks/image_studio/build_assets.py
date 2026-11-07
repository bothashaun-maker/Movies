from pathlib import Path
import json
R=Path(__file__).parent
source=(R/'asset_builder.py').read_text().replace('ASSET_CATALOG_SOURCE = None',(R/'assets.py').read_text())
intro='''# Image Studio Model Packs — same approach as your H3 model packs

1. Import this notebook into Kaggle. Enable **Internet** and select **Accelerator: None**. No GPU is needed.
2. Leave `PACK = 'standard'` and run its single code cell. It automatically downloads and verifies SDXL, RealESRGAN and SeedVR2 weights (about **11.1 GB**).
3. Wait for **PACK COMPLETE**, then **Save Version with output files**. Keep that version private. A draft alone is not durable storage.
4. In **Image Studio v3**, use **Add Input → Your Work / Notebook Output Files** and attach that saved version. Enable Internet + T4 GPU and run the image notebook.

This follows your uploaded **H3_Model_Packs Data Sets.ipynb**. No named Dataset, Kaggle API token, PC download, or re-upload is required. Saved outputs act as reusable model inputs. No model generation runs here.

For optional FLUX3, create a separate copy with `PACK = 'flux3'` (about **17.75 GB**). Accept the Hugging Face model terms and enable Kaggle Secret **HF_TOKEN** in that builder only. Do not combine both packs into one preparation session. Attach both saved outputs to v3 when using FLUX3 alongside the standard studio. The complete attached pack lets the renderer use FLUX3 without a token.

The renderer has `REQUIRE_ATTACHED_MODELS = True` by default, so missing packs stop startup rather than downloading weights again. Software setup and the studio tunnel still require Internet. Model transfers happen directly between the hosts and Kaggle; your picture uploads/downloads still use your connection.

Checksums and source URLs are recorded in a completed manifest only after the selected files pass verification. Valid existing files are reused, interrupted public downloads can resume, and saved pack files are real files rather than temporary symlinks. Save each pack's completed output before discarding its preparation session.

Optional named Dataset creation is available through `AUTO_CREATE_DATASET = True`. It requires private Kaggle API authentication and requests a new private Dataset; it never changes an existing Dataset. Leave this off to use the exact H3 saved-output approach.

Seven local checks cover synthetic packs, verification, reuse, corruption handling, private-creation arguments and failed-upload reporting. Full Kaggle pack building/saving/attachment and live Dataset upload remain unverified here.
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},'language_info':{'name':'python'},'accelerator':'none'},'cells':[{'cell_type':'markdown','id':'pack-guide','metadata':{},'source':intro.splitlines(True)},{'cell_type':'code','id':'pack-builder','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
(R/'Image_Studio_Model_Packs.ipynb').write_text(json.dumps(nb,indent=1))
print('Built CPU model pack notebook')
