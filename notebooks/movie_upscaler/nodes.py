"""Only video IO/checkpoint nodes; the canvas uses real upstream upscale nodes."""
import copy
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
import torch
import folder_paths
from . import media

class MovieUpload:
    CATEGORY = 'Movie Studio'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('source_path',)
    FUNCTION = 'select'
    DESCRIPTION = 'Click Upload clip to select a video from your computer. Upload finishes before Run; the connected path is filled automatically. Attached Kaggle paths also work.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source_path': ('STRING', {'default': '', 'tooltip': 'Filled by Upload clip. You may also paste an existing Kaggle video path.'})}}

    @classmethod
    def VALIDATE_INPUTS(cls, source_path):
        return Path(source_path).is_file() or 'Click Upload clip and wait for Ready before pressing Run.'

    def select(self, source_path):
        path = Path(source_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError('Choose a video clip before pressing Run.')
        return (str(path),)

class MovieChunk:
    CATEGORY = 'Movie Studio'
    RETURN_TYPES = ('IMAGE', 'MOVIE_JOB', 'INT', 'FLOAT')
    RETURN_NAMES = ('frames', 'job', 'target_short_edge', 'four_x_downscale')
    FUNCTION = 'load'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source_path': ('STRING', {'default': '/kaggle/input/your-video/clip.mp4'}),
            'scale': ([2, 4], {'default': 2}),
            'chunk_frames': ('INT', {'default': 17, 'min': 1, 'max': 65}),
            'context_frames': ('INT', {'default': 4, 'min': 0, 'max': 8}),
            'max_target_edge': ('INT', {'default': 1920, 'min': 64, 'max': 7680}),
            'resume_tick': ('INT', {'default': 0, 'min': 0, 'max': 2147483647}),
        }, 'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def load(self, source_path, scale, chunk_frames, context_frames, max_target_edge, resume_tick, prompt, unique_id):
        # Exclude queue counters and path identity; file SHA is part of the job key.
        settings = copy.deepcopy(prompt)
        for node in settings.values():
            node.pop('_meta', None)
            if node['class_type'] == 'MovieStudioChunk':
                node['inputs'].pop('resume_tick', None)
                node['inputs'].pop('source_path', None)
            if node['class_type'] == 'MovieStudioUpload':
                node['inputs'].pop('source_path', None)
            if node['class_type'] == 'MovieStudioCommit':
                for key in ('auto_continue', 'container', 'export_format', 'crf', 'preset'):
                    node['inputs'].pop(key, None)
        root = Path(folder_paths.get_output_directory()) / 'movie-jobs'
        meta = media.prepare(source_path, root, settings, chunk_frames, context_frames, int(scale), max_target_edge)
        index = media.next_chunk(meta)
        if index is None:
            # The lazy save node can re-export without invoking either upscaler.
            return torch.empty((0, 1, 1, 3)), dict(meta, complete=True), min(meta['info']['width'], meta['info']['height'])*int(scale), int(scale)/4.0
        raw, count = media.read_chunk(meta, index)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, meta['info']['height'], meta['info']['width'], 3).copy()
        meta = dict(meta, index=index, source_node=str(unique_id), resume_tick=resume_tick)
        print(f"Movie Studio: chunk {index+1}/{meta['chunks']}, job {meta['job']}", flush=True)
        return torch.from_numpy(frames).float().div_(255), meta, min(meta['info']['width'], meta['info']['height'])*int(scale), int(scale)/4.0

def export_movie(job, inputs):
    lossless = inputs.get('export_format', 'lossless_mkv') == 'lossless_mkv'
    return media.export(job, 'mkv' if lossless else 'mp4', crf=inputs.get('crf', 14), preset=inputs.get('preset', 'slow'), lossless=lossless)

class MovieCommit:
    CATEGORY = 'Movie Studio'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('status',)
    FUNCTION = 'save'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'frames': ('IMAGE', {'lazy': True}), 'job': ('MOVIE_JOB',),
                             'export_format': (['lossless_mkv', 'high_quality_mp4'], {'default': 'lossless_mkv', 'tooltip': 'Lossless MKV preserves processed RGB8 pixels exactly. MP4 uses the CRF/preset below and is smaller.'}),
                             'crf': ('INT', {'default': 14, 'min': 0, 'max': 23, 'tooltip': 'MP4 only. Lower means less compression; 14 is high quality. Lossless MKV ignores this.'}),
                             'preset': (['slow', 'medium', 'slower'], {'default': 'slow', 'tooltip': 'MP4 only; slower presets spend more CPU time compressing.'}),
                             'auto_continue': ('BOOLEAN', {'default': True})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}

    def check_lazy_status(self, job, frames=None, **kwargs):
        return ['frames'] if frames is None and not job.get('complete') else []

    def save(self, job, export_format, crf, preset, auto_continue, prompt, frames=None, extra_pnginfo=None):
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        if job.get('complete'):
            target = export_movie(job, dict(export_format=export_format, crf=crf, preset=preset))
            message = f'COMPLETE: {target}'
            print(message, flush=True)
            return {'ui': {'text': [message]}, 'result': (message,)}
        # Keep float conversion bounded to one frame instead of doubling the full chunk.
        count, h, w, channels = frames.shape
        if channels != 3:
            raise ValueError('Expected RGB frames.')
        raw = bytearray(count*h*w*3)
        stride = h*w*3
        for i in range(count):
            frame = frames[i].detach().cpu().numpy()
            if not np.isfinite(frame).all():
                raise ValueError('Upscaler produced non-finite pixels; refusing to save invalid video.')
            raw[i*stride:(i+1)*stride] = (np.clip(frame, 0, 1)*255).round().astype(np.uint8).tobytes()
        media.save_chunk(job, job['index'], raw, w, h, count)
        pending = media.next_chunk(job)
        if pending is None:
            mm.throw_exception_if_processing_interrupted()
            target = export_movie(job, dict(export_format=export_format, crf=crf, preset=preset))
            message = f'COMPLETE: {target}'
        elif auto_continue:
            mm.throw_exception_if_processing_interrupted()
            # One ordinary ComfyUI job per chunk. Clearing the queue stops continuation;
            # interrupting the current chunk stops before committing/queuing another.
            again = copy.deepcopy(prompt)
            again[job['source_node']]['inputs']['resume_tick'] = (job['resume_tick']+1) % 2147483647
            payload = {'prompt': again, 'extra_data': {'extra_pnginfo': extra_pnginfo or {}}}
            req = urllib.request.Request(os.environ.get('MOVIE_STUDIO_URL', 'http://127.0.0.1:8188')+'/prompt',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.load(response)
            if 'prompt_id' not in result or result.get('node_errors'):
                raise RuntimeError(f'Chunk saved, but continuation failed: {result}')
            message = f"Saved chunk {job['index']+1}/{job['chunks']}; queued next. {job['job']}"
        else:
            message = f"PAUSED after chunk {job['index']+1}/{job['chunks']}. Press Run to resume. {job['job']}"
        print(message, flush=True)
        return {'ui': {'text': [message]}, 'result': (message,)}

NODE_CLASS_MAPPINGS = {'MovieStudioUpload': MovieUpload, 'MovieStudioChunk': MovieChunk, 'MovieStudioCommit': MovieCommit}
NODE_DISPLAY_NAME_MAPPINGS = {'MovieStudioUpload': 'Movie • Upload clip', 'MovieStudioChunk': 'Movie • Load next resumable chunk',
                              'MovieStudioCommit': 'Movie • Commit chunk / export with audio'}
