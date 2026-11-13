"""Shot validation and observable job boundaries around native WanGP generation."""
from pathlib import Path
import functools
import inspect
import json
import os
import subprocess
import threading
import time
import uuid
from common import atomic_json, sha256

MODEL = 'minimax_h3_ref2va_pruned_pdd'
GUARD_LOCK = threading.Lock()


def media_path(value):
    if isinstance(value, (list, tuple)):
        return media_path(value[0]) if len(value) == 1 else None
    if isinstance(value, dict):
        return media_path(value.get('path') or value.get('name'))
    return str(value) if isinstance(value, (str, Path)) and str(value) else None


def probe_guide(path):
    try:
        process = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', path],
                                 capture_output=True, text=True, timeout=25, check=True, stdin=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError) as error:
        raise ValueError('Guide decoding failed or exceeded the 25-second inspection limit. Export a fresh silent H.264 MP4.') from error
    streams = json.loads(process.stdout)['streams']
    videos = [s for s in streams if s['codec_type'] == 'video']
    if len(videos) != 1:
        raise ValueError('Guide must contain exactly one video stream.')
    stream = videos[0]
    from fractions import Fraction
    if Fraction(stream.get('avg_frame_rate', '0')) != 24 or Fraction(stream.get('r_frame_rate', '0')) != 24:
        raise ValueError('Re-render the Blender guide at constant 24 fps.')
    if int(stream.get('nb_read_frames', 0)) != 124:
        raise ValueError('Re-render the Blender guide with FRAMES=124. Exactly 124 frames are required.')
    if (stream['width'], stream['height']) != (864, 480):
        raise ValueError('Guide dimensions must be 864x480, matching the output.')
    if any(s['codec_type'] == 'audio' for s in streams):
        raise ValueError('Guide contains audio. Export a silent Blender MP4; keep the song outside H3.')
    return stream


def check_shot(model_type, inputs, check_media=True):
    """Fail before queue/model load if a user action exceeds the tested kit scope."""
    if model_type != MODEL:
        raise ValueError('This kit supports only MiniMax H3 Ref2VA Pruned PDD 8-Step.')
    required = {'resolution': '864x480', 'video_length': 124, 'num_inference_steps': 8,
                'guidance_phases': 1, 'sample_solver': 'euler', 'batch_size': 1,
                'repeat_generation': 1, 'sliding_window_size': 124, 'image_mode': 0,
                'multi_prompts_gen_type': 'FG'}
    for key, expected in required.items():
        if inputs.get(key) != expected:
            raise ValueError(f'{key} must be {expected!r} in this motion test kit.')
    if str(inputs.get('config', '')) != 'gguf_q4_k_m,bf16':
        raise ValueError('Use Text Encoder: GGUF Q4_K_M and Video VAE: BF16 (the shipped FP16 VAE).')
    if str(inputs.get('force_fps', '')).strip() not in ('', '24', '24.0'):
        raise ValueError('Output fps must remain 24.')
    if int(inputs.get('seed', -1)) < 0:
        raise ValueError('Use a fixed non-negative seed; default 8675309.')
    prompt = str(inputs.get('prompt', ''))
    if not prompt.strip() or '[/' in prompt:
        raise ValueError('Use one non-empty prompt without timeline slash commands.')
    for name in ('prompt_enhancer', 'audio_prompt_type', 'activated_loras', 'overrides', 'video_source',
                 'audio_guide', 'audio_guide2', 'audio_guide3', 'image_start', 'image_end',
                 'temporal_upsampling', 'spatial_upsampling', 'video_guide2', 'video_guide3',
                 'skip_steps_cache_type', 'frames_positions', 'custom_sigmas'):
        if inputs.get(name):
            raise ValueError(f'{name} must be empty for the controlled comparison.')
    mode = str(inputs.get('video_prompt_type', ''))
    guided = set(mode) == set('IV-U') and len(mode) == 4
    if mode != 'I' and not guided:
        raise ValueError('Use one reference image with no video (A), or with ONE Reference Video (B).')
    refs = inputs.get('image_refs') or []
    if len(refs) != 1:
        raise ValueError('Upload exactly one appearance reference image.')
    if guided:
        path = media_path(inputs.get('video_guide'))
        if not path:
            raise ValueError('Upload the Blender MP4 under Reference / Control Video 1.')
        if check_media:
            probe_guide(path)
    elif inputs.get('video_guide'):
        raise ValueError('For baseline A remove the guide video, or select ONE Reference Video for B.')
    return 'B-guided' if guided else 'A-baseline'


def guarded(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        task = inputs.get('task') or {}
        params = task.get('params', {})
        model_type = inputs.get('model_type') or params.get('model_type') or inputs.get('state', {}).get('model_type') or MODEL
        # Recheck immediately before GPU work, also for restored queues and API submissions.
        label = check_shot(model_type, inputs)
        if not GUARD_LOCK.acquire(blocking=False):
            raise RuntimeError('One generation at a time. Wait for the current shot to finish.')
        output = Path(os.environ['JJP_OUTPUT'])
        run_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        bound.arguments['output_filename'] = 'JJP_'+label+'_'+run_id
        report = output/'shots'/run_id
        report.mkdir(parents=True)
        started = time.time()
        atomic_json(output/'active-job.json', {'active': True, 'started': started, 'run_id': run_id, 'label': label})
        try:
            simple = {key: value for key, value in inputs.items() if key not in ('state', 'send_cmd', 'task')}
            media = {}
            for key in ('image_refs', 'video_guide'):
                path = media_path(inputs.get(key))
                if path:
                    media[key] = {'path': path, 'sha256': sha256(path)}
            atomic_json(report/'settings.json', {'label': label, 'model_type': model_type, 'settings': simple,
                                                'media': media, 'started': started})
            result = function(*bound.args, **bound.kwargs)
            atomic_json(report/'result.json', {'native_return': result, 'elapsed_seconds': time.time()-started})
            return result
        except BaseException as error:
            atomic_json(report/'error.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.time()-started})
            raise
        finally:
            try:
                import sys
                native = sys.modules.get('wgp')
                if native is not None and hasattr(native, 'release_model'):
                    native.release_model()
            finally:
                atomic_json(output/'active-job.json', {'active': False, 'run_id': run_id, 'ended': time.time()})
                GUARD_LOCK.release()
    return run
