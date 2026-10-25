"""Failure-path tests for the standalone notebook controller (no GPU simulated)."""
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/h3_complete'


def scope_at(tmp):
    scope={}
    settings=(SOURCE/'settings.py').read_text().replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(tmp)!r})')
    exec(settings,scope)
    for name in ('processes','assets','graphs','runtime'):
        exec(compile((SOURCE/(name+'.py')).read_text(),name,'exec'),scope)
    return scope


class CompleteNotebook(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.temp.name)
        self.s=scope_at(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_fingerprint_and_stage_partition(self):
        s=self.s
        key,directory,sample,decode=s['prepare_job']()
        self.assertEqual(s['canvas_size'](.4,(2,3)),(512,768))
        self.assertEqual(s['frame_count'](5),124)
        self.assertEqual(key,s['prepare_job']()[0])
        self.assertNotIn('VAEDecode', {node['class_type'] for node in sample.values()})
        forbidden={'UNETLoaderH3TensorParallel','CLIPLoaderMultiGPU','SamplerCustomAdvanced','LoraLoaderModelOnly'}
        self.assertFalse(forbidden & {node['class_type'] for node in decode.values()})
        self.assertEqual(sample['19']['inputs']['checkpoint_id'],decode['19']['inputs']['checkpoint_id'])
        s['SEED']+=1
        self.assertNotEqual(key,s['prepare_job']()[0])
        s['SEED']-=1;s['PROMPT']+=' changed'
        self.assertNotEqual(key,s['prepare_job']()[0])

    def test_decode_failure_then_resume_without_sampling(self):
        s=self.s;s['STACK_READY']=True;s['require_models_ready']=lambda:None
        saved=False;failed=False;events=[]
        @contextmanager
        def process(phase,directory,graphs):
            events.append(('start',phase))
            try:yield None,None
            finally:events.append(('stop',phase))
        def execute(graph,process,log,directory,phase):
            nonlocal saved,failed
            events.append(('execute',phase))
            if phase=='sampling':saved=True;return {}
            if not failed:failed=True;raise RuntimeError('decoder OOM fixture')
            return {'ok':True}
        s.update(comfy_process=process,execute_graph=execute,validate_checkpoint=lambda _:saved,
                 video_paths=lambda _: [self.root/'clip.mp4'], verify_video=lambda *args:None)
        with self.assertRaisesRegex(RuntimeError,'decoder OOM'):
            s['render_clip']()
        self.assertLess(events.index(('stop','sampling')),events.index(('start','decoding')))
        result=s['render_clip']()
        self.assertEqual(result,self.root/'clip.mp4')
        self.assertEqual(events.count(('execute','sampling')),1)
        self.assertEqual(events.count(('stop','decoding')),2)

    def test_error_history_preserves_exception_before_tensor_dump(self):
        s=self.s;log=self.root/'stage.log';log.write_text('log')
        entry={'status':{'status_str':'error','messages':[['execution_error',{
            'node_type':'VAEDecode','node_id':'13','exception_type':'OOM',
            'exception_message':'allocation failed','traceback':['traceback retained'],
            'current_inputs':{'samples':'x'*25000}}]]}}
        s['api_request']=Mock(side_effect=[{'prompt_id':'fixture','node_errors':{}},{'fixture':entry}])
        process=types.SimpleNamespace(poll=lambda:None)
        with self.assertRaisesRegex(RuntimeError,'allocation failed'):
            s['execute_graph']({},process,log,self.root,'decoding')
        message=(self.root/'decoding-error.txt').read_text()
        self.assertIn('traceback retained',message)
        self.assertLess(len(message),1000)
        self.assertEqual(json.loads((self.root/'decoding-history.json').read_text()),entry)

    def test_real_child_is_reaped_after_interruption(self):
        s=self.s
        s['COMFY'].mkdir()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        s.update(PORT=port,API=f'http://127.0.0.1:{port}',stop_owned_servers=lambda:None)
        fake=self.root/'fake_server.py'
        fake.write_text('''import http.server,json,sys
print('fp16 mixed precision enabled',flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  body=json.dumps({}).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
''')
        popen=subprocess.Popen
        children=[]
        def launch(*args,**kwargs):
            child=popen([sys.executable,str(fake),str(port)],**kwargs)
            children.append(child);return child
        with patch.object(subprocess,'Popen',side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                with s['comfy_process']('sampling',self.root,[]):
                    raise KeyboardInterrupt()
        self.assertEqual(len(children),1)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse((pathlib.Path('/proc')/str(children[0].pid)).exists())

    def test_missing_nodes_block_before_queueing(self):
        with self.assertRaisesRegex(RuntimeError,'Required node did not load'):
            self.s['check_node_catalog']({},[self.s['sampling_graph']('fixture')])


if __name__=='__main__':unittest.main()
