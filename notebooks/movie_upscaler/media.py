"""Bounded SDR/CFR video IO and durable chunk checkpoints. No ML imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction

VERSION = 'movie-studio-v1'
_HASH_CACHE = {}

def run(args):
    return subprocess.run(list(map(str, args)), check=True, capture_output=True).stdout

def digest(path):
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    value = h.hexdigest()
    _HASH_CACHE[key] = value
    return value

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    with open(temp, 'w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)

def probe(path, count=False):
    args = ['ffprobe', '-v', 'error']
    if count:
        args += ['-count_frames']
    return json.loads(run(args + ['-show_streams', '-show_format', '-of', 'json', path]))

def video_info(path):
    data = probe(path)
    videos = [s for s in data['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if len(videos) != 1:
        raise ValueError('Use a file with one video stream (excluding cover art).')
    v = videos[0]
    if v.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR needs a separate colour-managed workflow. This version accepts SDR.')
    if any(int(x.get('rotation', 0)) % 360 for x in v.get('side_data_list', [])):
        raise ValueError('Rotate the source pixels before upscaling; display-rotation metadata is unsupported.')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', 'N/A', '0:1'):
        raise ValueError('Convert anamorphic video to square pixels before upscaling.')
    try:
        fps = Fraction(v.get('avg_frame_rate', '0/1'))
    except (ValueError, ZeroDivisionError):
        try:
            fps = Fraction(v.get('r_frame_rate', '0/1'))
        except (ValueError, ZeroDivisionError):
            raise ValueError('Unsupported or unknown source frame rate.') from None
    if not 0 < fps <= 120:
        raise ValueError('Unsupported or unknown source frame rate.')
    # Scan timestamps incrementally, avoiding a whole-film JSON allocation.
    cmd = ['ffprobe', '-v', 'error', '-select_streams', str(v['index']), '-show_entries',
           'frame=best_effort_timestamp_time', '-of', 'csv=p=0', str(path)]
    n = 0
    first = None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as p:
        for line in p.stdout:
            value = line.strip().split(',')[0]
            if not value:
                continue
            try:
                timestamp = float(value)
            except ValueError:
                p.kill()
                raise ValueError('Cannot determine frame timestamps.')
            if first is None:
                first = timestamp
            if abs(timestamp - first - n / float(fps)) > max(0.002, 0.1 / float(fps)):
                p.kill()
                raise ValueError('Variable-frame-rate video detected. Convert to constant frame rate first; no frames were silently retimed.')
            n += 1
        if p.wait():
            raise RuntimeError('ffprobe timestamp scan failed.')
    if n == 0:
        raise ValueError('Video contains no readable frames.')
    return {'width': v['width'], 'height': v['height'], 'fps': str(fps), 'frames': n,
            'start': first, 'format_start': float(data.get('format', {}).get('start_time', first)), 'stream_index': v['index'], 'audio_streams': sum(s['codec_type'] == 'audio' for s in data['streams']),
            'color': {k: v.get(k) for k in ('color_space', 'color_transfer', 'color_primaries', 'color_range')},
            'pixel_format': v.get('pix_fmt')}

def prepare(source, root, settings, core_frames, context, scale, max_edge):
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('Source must be a video file.')
    if not 1 <= core_frames <= 65 or not 0 <= context <= 8 or scale not in (2, 4):
        raise ValueError('Invalid chunk, context or scale setting.')
    fingerprint = digest(source)
    settings = dict(settings)
    key = hashlib.sha256(json.dumps([VERSION, fingerprint, settings], sort_keys=True).encode()).hexdigest()[:24]
    job = Path(root) / key
    job.mkdir(parents=True, exist_ok=True)
    manifest = job / 'job.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta['source_sha256'] != fingerprint or meta['settings'] != settings:
            raise RuntimeError('Checkpoint provenance mismatch.')
        meta['source'] = str(source)  # same content can be attached at a different Kaggle path
    else:
        info = video_info(source)
        w, h = info['width'] * scale, info['height'] * scale
        if w % 2 or h % 2 or max(w, h) > max_edge:
            raise ValueError(f'Target {w}x{h} exceeds max_edge={max_edge}. Raise the explicit cap only after testing memory.')
        meta = {'version': VERSION, 'source': str(source), 'source_sha256': fingerprint,
                'settings': settings, 'info': info, 'scale': scale, 'core_frames': core_frames,
                'context': context, 'chunks': math.ceil(info['frames'] / core_frames), 'job': str(job)}
        atomic_json(manifest, meta)
    # Conservative uncompressed estimate. Never delete other jobs or user files.
    frames = meta['info']['frames']
    raw = frames * meta['info']['width'] * meta['info']['height'] * scale**2 * 3
    existing = sum(p.stat().st_size for p in job.glob('chunk-*.mkv'))
    reserve = max(512 * 1024**2, raw // 10)
    if shutil.disk_usage(job).free < max(0, raw - existing) + reserve:
        raise RuntimeError(f'Not enough disk for a conservative {raw / 1024**3:.2f} GiB checkpoint estimate plus reserve. Use a shorter clip or smaller target.')
    return meta

def bounds(meta, index):
    start = index * meta['core_frames']
    end = min(start + meta['core_frames'], meta['info']['frames'])
    return start, end, max(0, start-meta['context']), min(meta['info']['frames'], end+meta['context'])

def committed(meta, index):
    base = Path(meta['job']) / f'chunk-{index:06d}'
    receipt = base.with_suffix('.json')
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    movie = base.with_suffix('.mkv')
    if not movie.exists() or digest(movie) != record['sha256']:
        raise RuntimeError(f'Corrupt committed checkpoint: {movie}. Preserve it and its receipt before repair.')
    return True

def next_chunk(meta):
    for index in range(meta['chunks']):
        if not committed(meta, index):
            return index
    return None

def read_chunk(meta, index):
    start, end, lo, hi = bounds(meta, index)
    # Seek to a nearby keyframe, decode accurately, then select by original PTS.
    # Half-frame boundaries tolerate container timebase rounding after CFR validation.
    fps = float(Fraction(meta['info']['fps']))
    first = meta['info']['start']
    seek = max(0, first-meta['info'].get('format_start', first)+(lo-2)/fps) if lo > 2 else 0
    lower, upper = first+(lo-0.5)/fps, first+(hi-0.5)/fps
    args = ['ffmpeg', '-v', 'error', '-copyts', '-noautorotate', '-ss', f'{seek:.9f}',
            '-i', meta['source'], '-map', f"0:{meta['info']['stream_index']}",
            '-vf', f'select=gte(t\\,{lower:.9f})*lt(t\\,{upper:.9f})',
            '-fps_mode', 'passthrough', '-frames:v', str(hi-lo),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1']
    raw = run(args)
    expected = (hi-lo)*meta['info']['width']*meta['info']['height']*3
    if len(raw) != expected:
        raise RuntimeError(f'Expected {expected} decoded bytes, received {len(raw)}. No checkpoint saved.')
    return raw, hi-lo

def save_chunk(meta, index, raw, width, height, count):
    start, end, lo, hi = bounds(meta, index)
    if count != hi-lo or (width, height) != (meta['info']['width']*meta['scale'], meta['info']['height']*meta['scale']):
        raise ValueError('Upscaler changed frame count or produced the wrong target dimensions.')
    stride = width*height*3
    if len(raw) != count*stride:
        raise ValueError('Incomplete output frame buffer.')
    # Context is reconstructed with neighbours, then discarded; no duplicate frames.
    core = memoryview(raw)[(start-lo)*stride:(end-lo)*stride]
    base = Path(meta['job']) / f'chunk-{index:06d}'
    temp = base.with_suffix('.partial.mkv')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
           '-r', meta['info']['fps'], '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '1', temp]
    subprocess.run(list(map(str, cmd)), input=core, check=True, capture_output=True)
    check = probe(temp, count=True)['streams'][0]
    if int(check['nb_read_frames']) != end-start:
        raise RuntimeError('Checkpoint verification failed.')
    temp.replace(base.with_suffix('.mkv'))
    atomic_json(base.with_suffix('.json'), {'sha256': digest(base.with_suffix('.mkv')), 'start': start, 'end': end,
                                           'width': width, 'height': height})

def export(meta, container='mp4'):
    if container not in ('mp4', 'mkv'):
        raise ValueError('Choose mp4 or mkv.')
    if next_chunk(meta) is not None:
        raise RuntimeError('Cannot export incomplete chunks.')
    job = Path(meta['job'])
    target = job / ('upscaled.'+container)
    receipt = job / ('export-'+container+'.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if target.exists() and digest(target) == old['sha256']:
            return target
        raise RuntimeError('Completed export is missing or corrupt; preserve its receipt before repair.')
    listing = job / 'concat.txt'
    listing.write_text(''.join(f"file 'chunk-{i:06d}.mkv'\n" for i in range(meta['chunks'])))
    temp = job / ('upscaled.partial.'+container)
    args = ['ffmpeg', '-v', 'error', '-y', '-copyts', '-f', 'concat', '-safe', '1', '-i', listing,
            '-itsoffset', str(-meta['info']['start']), '-i', meta['source'], '-map', '0:v:0', '-map', '1:a?',
            '-map_metadata', '1', '-map_chapters', '1', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-r', meta['info']['fps'], '-fps_mode', 'cfr', '-c:a', 'copy']
    color = meta['info'].get('color', {})
    matrix = color.get('color_space')
    if matrix in ('bt709', 'bt470bg', 'smpte170m', 'smpte240m', 'bt2020nc'):
        args += ['-vf', 'scale=out_color_matrix='+matrix+':out_range=tv', '-colorspace', matrix, '-color_range', 'tv']
    for key, flag in [('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')]:
        value = color.get(key)
        if value and value not in ('unknown', 'reserved', 'unspecified'):
            args += [flag, value]
    if container == 'mkv':
        args += ['-map', '1:s?', '-c:s', 'copy']
    else:
        args += ['-movflags', '+faststart']
    audio_mode = 'copy'
    try:
        run(args + [temp])
    except subprocess.CalledProcessError:
        if container != 'mp4':
            raise
        audio_mode = 'aac 192k (MP4 stream-copy failed)'
        retry = list(args)
        at = retry.index('-c:a')
        retry[at+1] = 'aac'
        run(retry + ['-b:a', '192k', temp])
    info = probe(temp, count=True)
    v = next(s for s in info['streams'] if s['codec_type'] == 'video')
    actual_audio = sum(s['codec_type'] == 'audio' for s in info['streams'])
    if int(v['nb_read_frames']) != meta['info']['frames'] or actual_audio != meta['info']['audio_streams']:
        raise RuntimeError('Final frame/audio-stream count failed validation; partial file retained.')
    if (v['width'], v['height']) != (meta['info']['width']*meta['scale'], meta['info']['height']*meta['scale']):
        raise RuntimeError('Final video dimensions failed validation.')
    temp.replace(target)
    atomic_json(receipt, {'file': str(target), 'sha256': digest(target), 'audio_mode': audio_mode,
                          'frames': int(v['nb_read_frames']), 'fps': meta['info']['fps']})
    return target
