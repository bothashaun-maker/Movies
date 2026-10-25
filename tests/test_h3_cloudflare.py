import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from h3_notebook_helpers import notebook_stages

ROOT=Path(__file__).resolve().parents[1]

class BrowserAccess(unittest.TestCase):
    def test_default_notebook_has_no_render_or_video_display_stage(self):
        nb=json.loads((ROOT/'minimax_h3_kaggle_complete.ipynb').read_text())
        stages=notebook_stages(nb)
        self.assertEqual(list(stages),['settings','catalog','install','weights','workflow','access'])
        entry=''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
        self.assertNotIn('display(Video(',entry)
        self.assertNotIn('FileLink',entry)
        self.assertNotIn('RESULT_PATH = render_clip()',entry)

    def test_tunnel_link_persists_until_explicit_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cloudflared').write_bytes(b'fixture')
            children=[]
            class Child:
                dead=False
                def poll(self):return 0 if self.dead else None
                def terminate(self):self.dead=True
                def wait(self,timeout=None):return 0
            def spawn(args,**kwargs):
                self.assertIn('--protocol',args);self.assertIn('http2',args)
                kwargs['stdout'].write('https://fixture-test.trycloudflare.com\n');kwargs['stdout'].flush()
                child=Child();children.append(child);return child
            scope=dict(ROOT=root,COMFY=root,PY=Path(sys.executable),STACK_READY=True,PORT=8188,
                       api_request=lambda *a,**k:{'H3IsolatedSaveVideo':{}},tail_log=lambda p:p.read_text(),
                       sha256_file=lambda p:'053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0')
            with patch.object(subprocess,'Popen',side_effect=spawn),patch('IPython.display.display') as display:
                exec((ROOT/'notebooks/h3_complete/cloudflare.py').read_text(),scope)
            self.assertEqual(scope['cloudflare_url'],'https://fixture-test.trycloudflare.com')
            self.assertIsNone(children[0].poll())
            self.assertIn('https://fixture-test.trycloudflare.com',(root/'comfyui-cloudflare-url.txt').read_text())
            display.assert_called_once()
            scope['stop_comfyui_access']()
            self.assertIsNotNone(children[0].poll())

if __name__=='__main__':unittest.main()
