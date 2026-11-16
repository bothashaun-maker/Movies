"""Chunked browser video uploads; each request stays below tunnel body limits."""
import asyncio
import json
from pathlib import Path
import re
import shutil
import uuid
from aiohttp import web
from . import media

CHUNK_BYTES = 8 * 1024**2
RESERVE = 512 * 1024**2
EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}

class UploadStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = self.root/'.partial'
        self.pending.mkdir(exist_ok=True)
        self.lock = asyncio.Lock()

    def paths(self, key):
        if not re.fullmatch(r'[0-9a-f]{32}', key):
            raise ValueError('Invalid upload identifier.')
        return self.pending/(key+'.json'), self.pending/(key+'.part')

    def state(self, key):
        record, data = self.paths(key)
        if not record.is_file() or data.is_symlink():
            raise ValueError('Upload not found. Choose your clip again.')
        return json.loads(record.read_text()), data

    def start(self, name, size):
        name = Path(str(name).replace('\\', '/')).name
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS or type(size) is not int or size <= 0:
            raise ValueError('Choose a nonempty MP4, MKV, MOV, WEBM, AVI or M4V clip.')
        outstanding = 0
        for record in self.pending.glob('*.json'):
            value = json.loads(record.read_text())
            data = record.with_suffix('.part')
            outstanding += max(0, value['size']-(data.stat().st_size if data.exists() else 0))
        if shutil.disk_usage(self.root).free < size+outstanding+RESERVE:
            raise ValueError('Not enough free disk for this upload plus working reserve.')
        key = uuid.uuid4().hex
        record, data = self.paths(key)
        clean = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(name).stem)[:100] or 'clip'
        media.atomic_json(record, {'size': size, 'name': clean+suffix})
        data.touch(exist_ok=False)
        return {'id': key, 'offset': 0, 'chunk_bytes': CHUNK_BYTES}

    async def append(self, key, offset, content):
        state, data = self.state(key)
        initial = data.stat().st_size
        if offset != initial:
            raise ValueError('Upload offset changed. Retry this clip.')
        received = 0
        try:
            with data.open('ab') as output:
                async for block in content.iter_chunked(256*1024):
                    received += len(block)
                    if received > CHUNK_BYTES or initial+received > state['size']:
                        raise ValueError('Upload chunk exceeds its declared size.')
                    if shutil.disk_usage(self.root).free < len(block)+RESERVE:
                        raise ValueError('Upload paused: not enough free disk.')
                    output.write(block)
            if not received:
                raise ValueError('Empty upload chunk.')
        except BaseException:
            with data.open('r+b') as output:
                output.truncate(initial)
            raise
        return {'offset': initial+received}

    def finish(self, key):
        state, data = self.state(key)
        if data.stat().st_size != state['size']:
            raise ValueError('Upload is incomplete.')
        try:
            info = media.probe(data)
            videos = [s for s in info['streams'] if s['codec_type']=='video' and not s.get('disposition', {}).get('attached_pic')]
            if len(videos)!=1 or not videos[0].get('width') or not videos[0].get('height'):
                raise ValueError('The uploaded file must contain one readable video stream.')
        except (KeyError, OSError, media.subprocess.CalledProcessError) as error:
            raise ValueError('The uploaded file could not be read as video.') from error
        target = self.root/(key+'-'+state['name'])
        data.replace(target)
        self.paths(key)[0].unlink()
        return {'path': str(target.resolve()), 'name': state['name'], 'bytes': state['size']}

    def cancel(self, key):
        record, data = self.paths(key)
        record.unlink(missing_ok=True)
        data.unlink(missing_ok=True)

def register_routes(routes, root):
    store = UploadStore(root)
    def endpoint(function):
        async def wrapped(request):
            try:
                async with store.lock:
                    return web.json_response(await function(request))
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                return web.json_response({'error': str(error)}, status=400)
        return wrapped

    @routes.post('/movie-studio/upload/start')
    @endpoint
    async def start(request):
        value = await request.json()
        return store.start(value.get('name', ''), value.get('size'))

    @routes.post('/movie-studio/upload/{key}/chunk')
    @endpoint
    async def chunk(request):
        return await store.append(request.match_info['key'], int(request.headers.get('X-Upload-Offset', '-1')), request.content)

    @routes.post('/movie-studio/upload/{key}/finish')
    @endpoint
    async def finish(request):
        return await asyncio.to_thread(store.finish, request.match_info['key'])

    @routes.delete('/movie-studio/upload/{key}')
    @endpoint
    async def cancel(request):
        store.cancel(request.match_info['key'])
        return {'cancelled': True}
    return store
