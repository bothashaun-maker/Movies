"""Exercise real streaming and Range recovery against a local HTTP server."""
import ast
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

REPO=Path(__file__).resolve().parents[1]


class DownloadTests(unittest.TestCase):
    def test_resumes_interrupted_response_and_handles_ignored_range(self):
        data=b'verified-model-bytes'*1000
        notebook=json.loads((REPO/'notebooks/minimax_h3_prepare_diffusion.ipynb').read_text())
        scope={};exec(''.join(notebook['cells'][2]['source']),scope)
        for ignore_range in (False,True):
            with self.subTest(ignore_range=ignore_range),tempfile.TemporaryDirectory() as tmp:
                requests=[]
                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        header=self.headers.get('Range');requests.append(header)
                        if len(requests)==1:
                            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
                            self.wfile.write(data[:len(data)//2]);self.close_connection=True;return
                        offset=int(header.removeprefix('bytes=').removesuffix('-')) if header and not ignore_range else 0
                        self.send_response(206 if offset else 200)
                        if offset:self.send_header('Content-Range',f'bytes {offset}-{len(data)-1}/{len(data)}')
                        self.send_header('Content-Length',str(len(data)-offset));self.end_headers();self.wfile.write(data[offset:])
                    def log_message(self,*args):pass
                server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                real_open=urllib.request.urlopen
                def local_open(request,**kwargs):
                    return real_open(urllib.request.Request(f'http://127.0.0.1:{server.server_port}/model',headers=dict(request.header_items())),**kwargs)
                asset=dict(repo='test/model',revision='pin',file='diffusion_models/test.safetensors',size=len(data),sha256=hashlib.sha256(data).hexdigest())
                try:
                    with patch.object(urllib.request,'urlopen',side_effect=local_open),patch('time.sleep'):
                        output=scope['download_verified'](asset,Path(tmp))
                    self.assertEqual(output.read_bytes(),data)
                    self.assertEqual(requests,[None,f'bytes={len(data)//2}-'])
                    self.assertFalse(output.with_suffix(output.suffix+'.part').exists())
                    with patch.object(urllib.request,'urlopen',side_effect=AssertionError('unexpected download')):
                        self.assertEqual(scope['download_verified'](asset,Path(tmp)),output)
                finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
