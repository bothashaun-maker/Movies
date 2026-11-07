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
    source = add('MovieStudioChunk', [0, 0], ['/kaggle/input/your-video/clip.mp4', 2, 17, 4 if seed else 0, 1920, 0], [],
                 [('frames', 'IMAGE'), ('job', 'MOVIE_JOB'), ('target_short_edge', 'INT')], '1 • Set movie path; 2× default; resumes saved chunks')
    if seed:
        dit = add('SeedVR2LoadDiTModel', [390, 0], ['seedvr2_ema_3b-Q8_0.gguf', 'cuda:0', 32, True, 'cpu', True, 'sdpa'], [],
                  [('SEEDVR2_DIT', 'SEEDVR2_DIT')], '2 • 3B Q8 / CPU block swapping / T4')
        vae = add('SeedVR2LoadVAEModel', [390, 450], ['ema_vae_fp16.safetensors', 'cuda:0', True, 512, 64, True, 512, 64, 'false', 'cpu', True], [],
                  [('SEEDVR2_VAE', 'SEEDVR2_VAE')], '3 • FP16 VAE; tiled encode and decode')
        up = add('SeedVR2VideoUpscaler', [800, 0], [42, 'fixed', 1024, 0, 5, True, 'lab', 2, 0, 0.0, 0.0, 'cpu', True],
                 [('image', 'IMAGE'), ('dit', 'SEEDVR2_DIT'), ('vae', 'SEEDVR2_VAE'), ('resolution', 'INT')], [('IMAGE', 'IMAGE')],
                 '4 • Temporal video restoration; target comes from source × scale')
        # Connected resolution retains its serialized widget position; the link supplies its actual value.
        nodes[up-1]['inputs'][3]['widget'] = {'name': 'resolution'}
        link(source, 0, up, 0); link(dit, 0, up, 1); link(vae, 0, up, 2); link(source, 2, up, 3)
    else:
        loader = add('UpscaleModelLoader', [390, 420], ['RealESRGAN_x4plus.pth'], [], [('UPSCALE_MODEL', 'UPSCALE_MODEL')], '2 • Verified Real-ESRGAN checkpoint')
        up4 = add('ImageUpscaleWithModel', [390, 0], [], [('upscale_model', 'UPSCALE_MODEL'), ('image', 'IMAGE')], [('IMAGE', 'IMAGE')], '3 • Native tiled 4× model')
        link(loader, 0, up4, 0); link(source, 0, up4, 1)
        up = add('ImageScaleBy', [800, 0], ['lanczos', 0.5], [('image', 'IMAGE')], [('IMAGE', 'IMAGE')], '4 • 4× → 2×; set scale_by=1 for a 4× job')
        link(up4, 0, up, 0)
    save = add('MovieStudioCommit', [1210, 0], ['mp4', True], [('frames', 'IMAGE'), ('job', 'MOVIE_JOB')], [('status', 'STRING')],
               '5 • Save chunk → continue → final movie with original audio')
    link(up, 0, save, 0); link(source, 1, save, 1)
    return {'last_node_id': len(nodes), 'last_link_id': len(links), 'nodes': nodes, 'links': links, 'groups': [], 'config': {},
            'extra': {'ds': {'scale': 0.65, 'offset': [80, 80]}}, 'version': 0.4}

def notebook(title, guide, code):
    compile(code, title, 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'}},
            'cells': [{'cell_type': 'markdown', 'id': 'guide', 'metadata': {}, 'source': guide.splitlines(True)},
                      {'cell_type': 'code', 'id': 'start', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}

def build():
    workflows = {'Movie Studio - SeedVR2': workflow(True), 'Movie Studio - RealESRGAN': workflow(False)}
    files = {'media.py': (ROOT/'media.py').read_text(), 'nodes.py': (ROOT/'nodes.py').read_text(),
             '__init__.py': 'from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\nWEB_DIRECTORY = "web"\n',
             'web/open.js': (ROOT/'open.js').read_text()}
    payload = 'FILES = '+repr(files)+'\nWORKFLOWS = '+repr(workflows)+'\nCF_SIZE = 40274445\n'
    assets = (ROOT/'assets.py').read_text()
    code = assets+'\n'+(ROOT/'launcher.py').read_text().replace('EMBEDDED_PAYLOAD = None', payload)
    nb = notebook('Movie Upscaler', (ROOT/'intro.md').read_text(), code)
    (REPO/'Movie_Upscaler_Kaggle_v1.ipynb').write_text(json.dumps(nb, indent=1)+'\n')
    guide = '''# Movie Upscaler — reusable model packs

Select **Accelerator: None**, **Internet ON**, then run the single cell.
It prepares about **3.94 GiB** of models, verifies SHA-256, and reuses matching attached weights.
No Kaggle token or Dataset API is required.

After completion, **Save Version with output files**. In Movie Upscaler select
**Add Input → Notebook Outputs**, then attach this saved version.
The pack contains actual files and a completion manifest. It does not include software or generated movies.
A saved output is reusable storage; it is not an automatically published named Kaggle Dataset.
'''
    pack = notebook('Movie model packs', guide, assets+"\nbuild_pack('/kaggle/working/movie-upscaler-model-pack')\n")
    (REPO/'Movie_Upscaler_Model_Packs.ipynb').write_text(json.dumps(pack, indent=1)+'\n')
    (ROOT/'workflows.json').write_text(json.dumps(workflows, indent=2)+'\n')
    print('Built Movie_Upscaler_Kaggle_v1.ipynb and Movie_Upscaler_Model_Packs.ipynb')

if __name__ == '__main__':
    build()
