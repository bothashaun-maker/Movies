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

class MovieChunk:
    CATEGORY = 'Movie Studio'
    RETURN_TYPES = ('IMAGE', 'MOVIE_JOB', 'INT')
    RETURN_NAMES = ('frames', 'job', 'target_short_edge')
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
            if node['class_type'] == 'MovieStudioCommit':
                node['inputs'].pop('auto_continue', None)
                node['inputs'].pop('container', None)
        root = Path(folder_paths.get_output_directory()) / 'movie-jobs'
        meta = media.prepare(source_path, root, settings, chunk_frames, context_frames, int(scale), max_target_edge)
        index = media.next_chunk(meta)
        if index is None:
            # Re-export if the cell stopped after committing the last chunk.
            output = next(n for n in prompt.values() if n['class_type'] == 'MovieStudioCommit')
            target = media.export(meta, output['inputs']['container'])
            raise RuntimeError(f'Job already complete: {target}. Change settings or source to start a different job.')
        raw, count = media.read_chunk(meta, index)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, meta['info']['height'], meta['info']['width'], 3).copy()
        meta = dict(meta, index=index, source_node=str(unique_id), resume_tick=resume_tick)
        print(f"Movie Studio: chunk {index+1}/{meta['chunks']}, job {meta['job']}", flush=True)
        return torch.from_numpy(frames).float().div_(255), meta, min(meta['info']['width'], meta['info']['height'])*int(scale)

class MovieCommit:
    CATEGORY = 'Movie Studio'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('status',)
    FUNCTION = 'save'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'frames': ('IMAGE',), 'job': ('MOVIE_JOB',),
                             'container': (['mp4', 'mkv'], {'default': 'mp4'}),
                             'auto_continue': ('BOOLEAN', {'default': True})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}

    def save(self, frames, job, container, auto_continue, prompt, extra_pnginfo=None):
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        # Keep float conversion bounded to one frame instead of doubling the full chunk.
        count, h, w, channels = frames.shape
        if channels != 3:
            raise ValueError('Expected RGB frames.')
        raw = bytearray(count*h*w*3)
        stride = h*w*3
        for i in range(count):
            raw[i*stride:(i+1)*stride] = (frames[i].detach().cpu().clamp(0, 1).numpy()*255).round().astype(np.uint8).tobytes()
        media.save_chunk(job, job['index'], raw, w, h, count)
        pending = media.next_chunk(job)
        if pending is None:
            mm.throw_exception_if_processing_interrupted()
            target = media.export(job, container)
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

NODE_CLASS_MAPPINGS = {'MovieStudioChunk': MovieChunk, 'MovieStudioCommit': MovieCommit}
NODE_DISPLAY_NAME_MAPPINGS = {'MovieStudioChunk': 'Movie • Load next resumable chunk',
                              'MovieStudioCommit': 'Movie • Commit chunk / export with audio'}
