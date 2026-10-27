"""Build a CPU-only, single-cell weight preparation notebook for all four renderers."""
from pathlib import Path
import hashlib,json,tempfile
import nbformat
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as scratch:
    scope={'Path':Path,'hashlib':hashlib,'json':json,'ROOT':Path(scratch)}
    exec((ROOT/'notebooks/h3_complete/assets.py').read_text(),scope)
    exec((ROOT/'notebooks/h3_suite/assets.py').read_text(),scope)
assets=scope['SUITE_ASSETS']
packs={k:[] for k in ('ref2va','fl2va','encoder','support')}
for asset in assets:
    name=asset['file']
    key=asset['family'] if name.startswith('diffusion_models/') else 'encoder' if name.startswith('text_encoders/') else 'support'
    packs[key].append(asset)
assert all(sum(a['size'] for a in pack)<20*1024**3 for pack in packs.values())
existing=nbformat.read(ROOT/'notebooks/minimax_h3_prepare_support.ipynb',4)
download=next(c.source for c in existing.cells if c.cell_type=='code' and c.source.startswith('def download_verified'))
intro='''# H3 model packs — all four notebooks

Use **Accelerator: None** and **Internet ON**. These are reusable model weights for Current Working Workflow, Pixa 5 Workflows, Director, and Three Step. No CUDA or ComfyUI installation is needed here.

Create four copies of this notebook. Give each a distinct title and set `PACK` in its code cell:

| Title | PACK | Size |
| --- | --- | --- |
'''
for k,v in packs.items():intro+=f"| H3 Models - {k} | `{k}` | {sum(a['size'] for a in v)/1024**3:.2f} GiB |\n"
intro+='''
Run and **Save Version with output files** for each copy. A completed saved run stores its selected pack under `/kaggle/working/models`; confirm those files appear in the saved version's Output/Files view. Keep these outputs private. If using Save & Run All, choose CPU/no accelerator for that saved run too. A draft alone is not a durable model store.

In any renderer choose **Add Input → Your Work / Notebook Outputs** and attach these four saved outputs. Leave `MODEL_DATASET=''`; the renderer discovers model files recursively in `/kaggle/input`. You do not need to make four duplicate collections for four renderers. The same outputs can be reused by all of them. If your Kaggle interface offers creating a Dataset from saved outputs, that is optional; notebook outputs work directly as inputs.

All packs together require about 73 GiB of saved storage. Check your account's actual storage/output limits. Each pack is below 20 GiB, but the notebook checks local free space and cannot guarantee Kaggle will accept every saved output. Do not download all packs into one preparation session: use one pack per copy/session.

This prepares every model currently pinned by the four notebooks, including both diffusion families, the common encoder, both video VAEs, audio VAE, four-step LoRAs, TaoMate and both PDD LoRAs. Optional Semantic Bridge and VSA are excluded. Future weight revisions may require updating a pack. Runtime installation and local hash checks still take time; attaching weights avoids fetching these large model files from Hugging Face.
'''
code="from pathlib import Path\nimport json,shutil\nPACK='ref2va'  # ref2va, fl2va, encoder, support\nOUTPUT=Path('/kaggle/working/models')\nPACKS="+repr(packs)+'''
assert PACK in PACKS, 'PACK must be ref2va, fl2va, encoder, or support'
selected=PACKS[PACK]
OUTPUT.mkdir(parents=True,exist_ok=True)
manifest=OUTPUT.parent/'h3-pack-selection.json'
if manifest.exists():
    assert json.loads(manifest.read_text())['pack']==PACK, 'Use a fresh CPU session for another pack; do not combine packs in one output.'
needed=sum(a['size'] for a in selected if not (OUTPUT/a['file']).exists())
free=shutil.disk_usage(OUTPUT).free
assert free>=needed+256*1024**2, f'Need {needed/1024**3:.2f} GiB plus headroom, have {free/1024**3:.2f} GiB. Use a fresh session with sufficient space.'
manifest.write_text(json.dumps({'pack':PACK,'assets':selected},indent=2))
print(f'Preparing {PACK}: {sum(a["size"] for a in selected)/1024**3:.2f} GiB. No GPU required.')
'''+download+'''
for asset in selected:
    download_verified(asset,OUTPUT)
print('Verified pack:',PACK)
print('Save this CPU notebook version with its output files. Attach the saved output to your renderer using Add Input.')
'''
compile(code,'model-packs','exec')
n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(code)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
for i,c in enumerate(n.cells):c.id=f'h3-model-packs-{i}'
nbformat.validate(n)
nbformat.write(n,ROOT/'H3_Model_Packs.ipynb')
print({k:round(sum(a['size'] for a in v)/1024**3,2) for k,v in packs.items()})
