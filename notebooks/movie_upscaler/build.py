"""Rebuild both self-contained Kaggle notebooks and visible ComfyUI graphs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

def workflow(seed=True):
    nodes, links = [], []
    def add(kind, xy, widgets, inputs, outputs, title):
        i = len(nodes)+1
        nodes.append({'id': i, 'type': kind, 'pos': xy, 'size': [340, 390], 'flags': {}, 'order': i-1, 'mode': 0,
                      'inputs': [{'name': k, 'type': t, 'link': None} for k, t in inputs],
                      'outputs': [{'name': k, 'type': t, 'links': []} for k, t in outputs],
                      'properties': {'Node name for S&R': kind}, 'widgets_values': widgets, 'title': title})
        return i
    def link(a, slot, b, inp):
        i = len(links)+1
        typ = nodes[a-1]['outputs'][slot]['type']
        links.append([i, a, slot, b, inp, typ])
        nodes[a-1]['outputs'][slot]['links'].append(i)
        nodes[b-1]['inputs'][inp]['link'] = i
    upload = add('MovieStudioUpload', [-470, 0], [''], [], [('source_path', 'STRING')], '1 • Upload clip from your computer')
    source = add('MovieStudioChunk', [0, 0], ['', 2, 17, 4 if seed else 0, 1920, 0], [('source_path', 'STRING')],
                 [('frames', 'IMAGE'), ('job', 'MOVIE_JOB'), ('target_short_edge', 'INT'), ('four_x_downscale', 'FLOAT')], '2 • 2× quality target / resumable chunks')
    nodes[source-1]['inputs'][0]['widget'] = {'name': 'source_path'}
    link(upload, 0, source, 0)
    if seed:
        dit = add('SeedVR2LoadDiTModel', [390, 0], ['seedvr2_ema_3b-Q8_0.gguf', 'cuda:0', 32, True, 'cpu', True, 'sdpa'], [],
                  [('SEEDVR2_DIT', 'SEEDVR2_DIT')], '3 • 3B Q8 / T4 / memory offload')
        vae = add('SeedVR2LoadVAEModel', [390, 450], ['ema_vae_fp16.safetensors', 'cuda:1', True, 512, 128, True, 512, 128, 'false', 'cpu', True], [],
                  [('SEEDVR2_VAE', 'SEEDVR2_VAE')], '4 • FP16 VAE / 512 tiles / 128 overlap')
        up = add('SeedVR2VideoUpscaler', [800, 0], [42, 'fixed', 1024, 0, 5, True, 'lab', 2, 0, 0.0, 0.0, 'cpu', True],
                 [('image', 'IMAGE'), ('dit', 'SEEDVR2_DIT'), ('vae', 'SEEDVR2_VAE'), ('resolution', 'INT')], [('IMAGE', 'IMAGE')],
                 '5 • Temporal quality / uniform 5 frames / overlap 2')
        # Connected resolution retains its serialized widget position; the link supplies its actual value.
        nodes[up-1]['inputs'][3]['widget'] = {'name': 'resolution'}
        link(source, 0, up, 0); link(dit, 0, up, 1); link(vae, 0, up, 2); link(source, 2, up, 3)
    else:
        loader = add('UpscaleModelLoader', [390, 420], ['RealESRGAN_x4plus.pth'], [], [('UPSCALE_MODEL', 'UPSCALE_MODEL')], '3 • Real-ESRGAN x4plus')
        up4 = add('ImageUpscaleWithModel', [390, 0], [], [('upscale_model', 'UPSCALE_MODEL'), ('image', 'IMAGE')], [('IMAGE', 'IMAGE')], '4 • Native tiled 4× model')
        link(loader, 0, up4, 0); link(source, 0, up4, 1)
        up = add('ImageScaleBy', [800, 0], ['lanczos', 0.5], [('image', 'IMAGE'), ('scale_by', 'FLOAT')], [('IMAGE', 'IMAGE')], '5 • Lanczos / automatic exact target size')
        nodes[up-1]['inputs'][1]['widget'] = {'name': 'scale_by'}
        link(up4, 0, up, 0); link(source, 3, up, 1)
    save = add('MovieStudioCommit', [1210, 0], ['lossless_mkv', 14, 'slow', True], [('frames', 'IMAGE'), ('job', 'MOVIE_JOB')], [('status', 'STRING')],
               '6 • Lossless RGB master / optional high-quality MP4 / original audio')
    link(up, 0, save, 0); link(source, 1, save, 1)
    for node in nodes:
        node['pos'][0] += 500  # Keep the upload button visible on first open.
    return {'last_node_id': len(nodes), 'last_link_id': len(links), 'nodes': nodes, 'links': links, 'groups': [], 'config': {},
            'extra': {'ds': {'scale': 0.6, 'offset': [120, 220]}}, 'version': 0.4}

def notebook(title, guide, code):
    compile(code, title, 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'}},
            'cells': [{'cell_type': 'markdown', 'id': 'guide', 'metadata': {}, 'source': guide.splitlines(True)},
                      {'cell_type': 'code', 'id': 'start', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}

def build():
    workflows = {'Movie Studio - SeedVR2': workflow(True), 'Movie Studio - RealESRGAN': workflow(False)}
    files = {'uploads.py': (ROOT/'uploads.py').read_text(), 'web/upload.js': (ROOT/'upload.js').read_text(), 'media.py': (ROOT/'media.py').read_text(), 'nodes.py': (ROOT/'nodes.py').read_text(),
             '__init__.py': 'from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\nfrom .uploads import register_routes\nimport folder_paths\nfrom server import PromptServer\nfrom pathlib import Path\nWEB_DIRECTORY = "web"\nif getattr(PromptServer, "instance", None) is not None:\n    register_routes(PromptServer.instance.routes, Path(folder_paths.get_input_directory())/"movie-clips")\n',
             'web/open.js': (ROOT/'open.js').read_text()}
    payload = 'FILES = '+repr(files)+'\nWORKFLOWS = '+repr(workflows)+'\nCF_SIZE = 40274445\n'
    assets = (ROOT/'assets.py').read_text()
    code = assets+'\n'+(ROOT/'launcher.py').read_text().replace('EMBEDDED_PAYLOAD = None', payload)
    nb = notebook('Movie Upscaler', (ROOT/'intro.md').read_text(), code)
    for name in ('Movie_Upscaler_Kaggle_v2.ipynb', 'Movie_Upscaler_Kaggle_v1.ipynb'):
        (REPO/name).write_text(json.dumps(nb, indent=1)+'\n')
    guide = '''# Movie Upscaler — automatic private Dataset creator

Select **Accelerator: None**, **Internet ON**. Enable a private Kaggle Secret named
**KAGGLE_API_TOKEN** with your Kaggle API token, and enter your Kaggle username in
`DATASET_OWNER` if authentication does not supply it. Never paste the token into the notebook.

Run the single cell. It downloads/reuses and verifies approximately **3.94 GiB** of
weights, then **automatically creates and uploads a private Kaggle Dataset** and prints
its link. Wait for Kaggle processing, then attach it to Movie Upscaler through
**Add Input → Your Datasets**. No manual Save Version is needed to submit the Dataset.

This follows `save_private_dataset()` in the repository's Image Studio builder.
Repeated runs reuse verified files and a recorded successful submission; they never
replace existing Datasets or change their visibility. Failed uploads retain verified
files and raise an error. Live Kaggle submission/processing remains unverified here.

Set `AUTO_CREATE_DATASET=False` only if you prefer the original Save Version with
output files → attach Notebook Outputs method. Save Version is also useful for
preserving the notebook and logs, separately from the automatically uploaded Dataset.
'''
    options = "AUTO_CREATE_DATASET = True\nDATASET_OWNER = ''  # Your Kaggle username; auto-detected when available.\n"
    pack = notebook('Movie model packs', guide, options+assets+"\nbuild_and_save('/kaggle/working/movie-upscaler-model-pack', DATASET_OWNER, AUTO_CREATE_DATASET)\n")
    (REPO/'Movie_Upscaler_Model_Packs.ipynb').write_text(json.dumps(pack, indent=1)+'\n')
    (ROOT/'workflows.json').write_text(json.dumps(workflows, indent=2)+'\n')
    print('Built Movie_Upscaler_Kaggle_v1.ipynb and Movie_Upscaler_Model_Packs.ipynb')

if __name__ == '__main__':
    build()
