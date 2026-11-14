"""Native WanGP CPU integration harness. Hardware metadata is simulated, inference is not.

Requires a disposable WanGP checkout and the kit's Python dependencies installed
against CPU PyTorch. Never use this harness as a GPU benchmark.
"""
from pathlib import Path
import argparse
import inspect
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
from patch import patch_wgp, patch_h3_dtype, patch_h3_numerics
from launcher import native_command, headless_environment
work.mkdir(parents=True, exist_ok=True)
patch_wgp(app/'wgp.py')
patch_h3_dtype(app/'models/minimax_h3/minimax_h3_handler.py')
patch_h3_numerics(app)
config, saved = work/'config', work/'settings'
config.mkdir(exist_ok=True)
saved.mkdir(exist_ok=True)
atomic_json(config/'wgp_config.json', server_settings(work/'models', work/'output'))
atomic_json(saved/(MODEL+'_settings.json'), shot_settings())
os.environ.update(headless_environment())
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
command = native_command(sys.executable, app, work, config, saved, arguments.serve or 7871)
sys.argv = list(map(str, command[2:]))
if arguments.serve:
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
    # Exercise the real workspace + download decorators with the real generation
    # signature, stopping at a CPU sentinel instead of entering model inference.
    from shared.utils.workspace_storage import workspace_generation
    from shared.utils.download import generation_downloads
    from guard import guarded
    from unittest.mock import patch
    native_signature = inspect.signature(wgp.generate_media)
    reached = []

    def sentinel(*args, **kwargs):
        bound = native_signature.bind(*args, **kwargs)
        reached.append(bound.arguments)
        return True

    sentinel.__signature__ = native_signature
    dispatch = guarded(workspace_generation(generation_downloads(lambda state: state['gen'])(sentinel)))
    dispatch_state = {'model_type': MODEL, 'gen': {}}
    for guided in (False, True):
        shot = dict(inputs)
        shot.update(shot_settings(guided))
        shot['image_refs'] = [str(image)]
        shot['video_guide'] = str(Path(arguments.guide).resolve()) if guided else None
        params = {name: shot.get(name) for name in native_signature.parameters
                  if name not in ('task', 'send_cmd')}
        params.update(state=dispatch_state, model_type=MODEL, mode='generate', plugin_data=None)
        with patch.object(wgp, 'release_model') as release:
            assert dispatch({'params': {'model_type': MODEL}}, lambda *a: None, **params) is True
            release.assert_called_once()
        assert reached[-1]['state'] is dispatch_state
        assert reached[-1]['output_filename'].startswith('JJP_B-guided_' if guided else 'JJP_A-baseline_')
    assert wgp.wan_model is None
    from models.minimax_h3.transformer import MiniMaxH3Model
    assert MiniMaxH3Model._jjp_safe_fp16_installed
    atomic_json(work/'native-validation.json', {'baseline': True, 'guided': True, 'compute_dtype': 'fp16',
                                            'safe_fp16_installed': True,
                                            'resolution_locked': True, 'reference_video_hw': dimensions, 'model_loaded': False,
                                            'native_decorator_dispatches': len(reached),
                                            'hardware_metadata_simulated': True, 'gpu_inference_tested': False})
    print('PASS: real WanGP A/B validation and native workspace/download dispatch, FP16 selection, locked resolution, zero model loads. CPU metadata simulation only.')
