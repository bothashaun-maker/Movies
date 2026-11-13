"""Native WanGP CPU integration harness. Hardware metadata is simulated, inference is not.

Requires a disposable WanGP checkout and the kit's Python dependencies installed
against CPU PyTorch. Never use this harness as a GPU benchmark.
"""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys
import types

parser = argparse.ArgumentParser()
parser.add_argument('--app', required=True)
parser.add_argument('--work', required=True)
parser.add_argument('--guide', required=True)
parser.add_argument('--serve', type=int, default=0)
arguments = parser.parse_args()
app, work = Path(arguments.app).resolve(), Path(arguments.work).resolve()
source = Path(__file__).resolve().parents[1]/'notebooks/wan_gp_motion'
sys.path.insert(0, str(source))
from common import atomic_json
from settings import MODEL, server_settings, shot_settings
from patch import patch_wgp, patch_h3_dtype
work.mkdir(parents=True, exist_ok=True)
patch_wgp(app/'wgp.py')
patch_h3_dtype(app/'models/minimax_h3/minimax_h3_handler.py')
config, saved = work/'config', work/'settings'
config.mkdir(exist_ok=True)
saved.mkdir(exist_ok=True)
atomic_json(config/'wgp_config.json', server_settings(work/'models', work/'output'))
atomic_json(saved/(MODEL+'_settings.json'), shot_settings())
os.environ.update(GRADIO_ANALYTICS_ENABLED='False', JJP_OUTPUT=str(work/'output'))
os.chdir(app)
sys.path.insert(0, str(app))
import torch
assert not torch.cuda.is_available(), 'This harness is for CPU-only integration checks.'
torch.cuda.get_device_capability = lambda *a, **kw: (7, 5)
torch.cuda.get_device_name = lambda *a, **kw: 'SIMULATED T4 metadata: CPU UI test only'
torch.cuda.get_device_properties = lambda *a, **kw: types.SimpleNamespace(
    name='SIMULATED T4', total_memory=15*1024**3, major=7, minor=5, multi_processor_count=40)
torch.cuda.mem_get_info = lambda *a, **kw: (14*1024**3, 15*1024**3)
torch.cuda.current_device = lambda: 0
sys.argv = ['wgp.py', '--offline', '--profile', '5', '--fp16', '--attention', 'sdpa',
            '--ram-allocator', 'default', '--vram-allocator', 'default', '--config', str(config),
            '--settings', str(saved), '--no-mcp-webui', '--lock-config', '--advanced']
if arguments.serve:
    sys.argv += ['--server-name', '127.0.0.1', '--server-port', str(arguments.serve)]
    runpy.run_path(str(app/'wgp.py'), run_name='__main__')
else:
    import wgp
    from PIL import Image
    image = work/'appearance.png'
    Image.new('RGB', (864, 480), 'tan').save(image)
    inputs = wgp.get_factory_settings(MODEL)
    inputs.update(wgp.get_default_settings(MODEL))
    inputs['image_refs'] = [str(image)]
    for mode in ('I', 'IV-U'):
        inputs['video_prompt_type'] = mode
        inputs['video_guide'] = str(Path(arguments.guide).resolve()) if mode == 'IV-U' else None
        result = wgp.validate_settings({'model_type': MODEL}, MODEL, True, inputs, silent=True)
        assert not result[-1], result[-1]
    assert wgp.get_transformer_dtype(MODEL, 'fp16') == torch.float16
    assert wgp.get_model_def(MODEL)['lock_resolutions'] is True
    dimensions = wgp.get_reference_video_dimensions(str(Path(arguments.guide).resolve()), 480, 864,
                                                   wgp.get_model_def(MODEL)['reference_video_max_size'], 32)
    assert dimensions == (160, 288), dimensions
    assert wgp.wan_model is None
    atomic_json(work/'native-validation.json', {'baseline': True, 'guided': True, 'compute_dtype': 'fp16',
                                            'resolution_locked': True, 'reference_video_hw': dimensions, 'model_loaded': False,
                                            'hardware_metadata_simulated': True, 'gpu_inference_tested': False})
    print('PASS: real WanGP A/B validation, FP16 selection, locked resolution, zero model loads. CPU metadata simulation only.')
