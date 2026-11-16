"""Exercise real streamed upload HTTP routes with real video bytes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).parent
package = types.ModuleType('upload_test_package'); package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__+'.uploads', ROOT/'uploads.py')
uploads = importlib.util.module_from_spec(spec); spec.loader.exec_module(uploads)

class UploadChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.clip = self.root/'source.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=24x16:rate=24',
                        '-frames:v','3','-an','-c:v','libx264',str(self.clip)], check=True)
        self.routes = web.RouteTableDef()
        self.store = uploads.register_routes(self.routes, self.root/'input')
        app = web.Application(client_max_size=1024)  # stream route must not read the whole body through request.read()
        app.add_routes(self.routes)
        self.client = TestClient(TestServer(app)); await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close(); self.temp.cleanup()

    async def start(self, data, name='my clip.mp4'):
        response = await self.client.post('/movie-studio/upload/start', json={'name': name, 'size': len(data)})
        self.assertEqual(response.status, 200)
        return await response.json()

    async def test_real_video_crosses_chunk_boundary_and_retains_exact_bytes(self):
        data = self.clip.read_bytes()+b'\0'*uploads.CHUNK_BYTES
        state = await self.start(data, '../../my clip.mp4')
        for offset in range(0, len(data), uploads.CHUNK_BYTES):
            part = data[offset:offset+uploads.CHUNK_BYTES]
            response = await self.client.post(f"/movie-studio/upload/{state['id']}/chunk", data=part,
                                               headers={'X-Upload-Offset': str(offset)})
            self.assertEqual(response.status, 200, await response.text())
            self.assertEqual((await response.json())['offset'], offset+len(part))
        response = await self.client.post(f"/movie-studio/upload/{state['id']}/finish")
        self.assertEqual(response.status, 200, await response.text())
        result = await response.json(); path = Path(result['path'])
        self.assertTrue(path.is_relative_to(self.root/'input'))
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(result['name'], 'my_clip.mp4')
        self.assertEqual(len(uploads.media.probe(path)['streams']), 1)

    async def test_bad_offset_incomplete_and_cancel_preserve_other_files(self):
        data = self.clip.read_bytes(); state = await self.start(data)
        base = f"/movie-studio/upload/{state['id']}"
        response = await self.client.post(base+'/chunk', data=data, headers={'X-Upload-Offset':'1'})
        self.assertEqual(response.status, 400)
        response = await self.client.post(base+'/finish'); self.assertEqual(response.status, 400)
        other = self.root/'input/user-video.mp4'; other.write_bytes(data)
        response = await self.client.delete(base); self.assertEqual(response.status, 200)
        self.assertEqual(other.read_bytes(), data)
        self.assertFalse(any(self.store.pending.iterdir()))

    async def test_reject_invalid_video_and_oversized_chunk_without_partial_append(self):
        state = await self.start(b'not a video')
        base = f"/movie-studio/upload/{state['id']}"
        response = await self.client.post(base+'/chunk', data=b'not a video', headers={'X-Upload-Offset':'0'})
        self.assertEqual(response.status, 200)
        response = await self.client.post(base+'/finish'); self.assertEqual(response.status, 400)
        state = await self.start(b'x'*(uploads.CHUNK_BYTES+1))
        response = await self.client.post(f"/movie-studio/upload/{state['id']}/chunk", data=b'x'*(uploads.CHUNK_BYTES+1), headers={'X-Upload-Offset':'0'})
        self.assertEqual(response.status, 400)
        self.assertEqual(self.store.state(state['id'])[1].stat().st_size, 0)

    async def test_invalid_extension_and_capacity_fail_before_writing(self):
        response = await self.client.post('/movie-studio/upload/start', json={'name':'clip.exe','size':10})
        self.assertEqual(response.status, 400)
        with patch.object(uploads.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)):
            response = await self.client.post('/movie-studio/upload/start', json={'name':'clip.mp4','size':10})
            self.assertEqual(response.status, 400)
        self.assertFalse(any(self.store.pending.iterdir()))

if __name__ == '__main__': unittest.main(verbosity=2)
