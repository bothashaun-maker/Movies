"""Local tests; no live Kaggle writes, model downloads or GPU execution."""
from contextlib import nullcontext
from pathlib import Path
import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import integrate
import ltx25_common as common
import runtime
import dataset


def asset(name, content, file=None):
    return dict(name=name, file=file or name, bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(), source='https://invalid.example/'+name)


class SourceAndNotebookTests(unittest.TestCase):
    def test_pinned_source_and_inference_changes(self):
        raw = (HERE/'aiquest_upstream.ipynb').read_bytes()
        original = integrate.extract_engine(raw)
        adapted = integrate.prepare_engine(original)
        self.assertEqual(adapted, (HERE/'engine.py').read_text())
        before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(adapted).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual({n for n in before if before[n] != after[n]}, integrate.ALLOWED_FUNCTION_CHANGES)
        for name in ('enable_tensor_parallel', '_replicate_block', 'ShardLinear', 'decode_video'):
            self.assertIn(name, list(before))
            self.assertEqual(before[name], after[name])
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            integrate.extract_engine(raw+b' ')

    def test_model_catalog_exact_and_complete(self):
        cat = json.loads((HERE/'assets.json').read_text())
        self.assertEqual(len(cat['assets']), 13)
        self.assertEqual(len({a['name'] for a in cat['assets']}), 13)
        for a in cat['assets']:
            self.assertEqual(len(a['sha256']), 64)
            self.assertGreater(a['bytes'], 0)
            self.assertIn('/resolve/'+a['revision']+'/', a['source'])
        self.assertLess(max(a['bytes'] for a in cat['assets']), 20*common.GIB)

    def test_notebooks_selfcontained_and_payloads_match(self):
        for name, gpu in [('JJP_LTX25_T4.ipynb', True), ('JJP_LTX25_Models.ipynb', False)]:
            nb = nbformat.read(ROOT/name, as_version=4)
            nbformat.validate(nb)
            self.assertEqual(nb.metadata.kaggle.isGpuEnabled, gpu)
            cells = [c for c in nb.cells if c.cell_type == 'code']
            self.assertEqual(len(cells), 1)
            tree = ast.parse(cells[0].source)
            encoded = next(node.args[0].value for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'b64decode')
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as z:
                for path in z.namelist():
                    self.assertEqual(Path(path).name, path)
                    mapped = {'ltx25_runtime.py': 'runtime.py', 'ltx25_dataset.py': 'dataset.py'}.get(path, path)
                    self.assertEqual(z.read(path), (HERE/mapped).read_bytes())
            self.assertIsNone(cells[0].execution_count)
            self.assertEqual(cells[0].outputs, [])

    def test_staging_hashes_and_restores_nested_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'input'; source.mkdir()
            content = b'valid-fixture'
            a = asset('config.json', content, 'encoder/config.json')
            (source/a['name']).write_bytes(content)
            common.stage_models({'assets': [a]}, source, root/'cache')
            self.assertEqual((root/'cache'/a['file']).read_bytes(), content)
            self.assertEqual((source/a['name']).read_bytes(), content)
            (root/'cache'/a['file']).write_bytes(b'invalid')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                common.stage_models({'assets': [a]}, source, root/'cache')
            self.assertEqual((root/'cache'/a['file']).read_bytes(), b'invalid')

    def test_complete_dataset_not_mixed_packs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cat = {'assets': [asset('a', b'a'), asset('b', b'b')]}
            for n in ('first', 'second'):
                (root/n).mkdir(); (root/n/'manifest.json').write_text(json.dumps(cat))
            (root/'first/a').write_bytes(b'a'); (root/'second/b').write_bytes(b'b')
            with self.assertRaisesRegex(RuntimeError, 'ONE complete'):
                runtime.find_dataset(cat, root)
            (root/'first/b').write_bytes(b'b')
            self.assertEqual(runtime.find_dataset(cat, root), root/'first')

    def test_budget_stop_semantics(self):
        self.assertIsNone(runtime.budget_reason(100, None, 1200, 1800))
        self.assertIsNotNone(runtime.budget_reason(1201, None, 1200, 1800))
        self.assertIsNotNone(runtime.budget_reason(1, {'active': True, 'elapsed': 1801}, 1200, 1800))
        self.assertIsNone(runtime.budget_reason(9999, {'active': False, 'elapsed': 9999}, 1200, 1800))

    def test_auth_failure_happens_before_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (patch.object(dataset, 'authenticated_api', side_effect=RuntimeError('auth unavailable')),
                  patch.object(dataset, 'transfer') as transfer):
                with self.assertRaisesRegex(RuntimeError, 'auth unavailable'):
                    dataset.create_dataset({'assets': [asset('a', b'a')]}, root/'out', root/'input', root/'cache')
                transfer.assert_not_called()
            self.assertFalse((root/'out/dataset-upload.json').exists())


class NumericalBoundaryTests(unittest.TestCase):
    def functions(self):
        tree = ast.parse((HERE/'engine.py').read_text())
        wanted = {'_cond_attr', '_wrap_conditioner', 'set_denoising_steps'}
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
        import numpy as np
        class Tensor:
            def __init__(self, value): self.value=np.asarray(value); self.dtype='fp32'
            def to(self, **kwargs): return self
        torch = types.SimpleNamespace(is_tensor=lambda t:isinstance(t,Tensor),
            isfinite=lambda t:np.isfinite(t.value), float32='fp32', device=lambda x:x)
        bar = types.SimpleNamespace(start=lambda *a:None,finish=lambda:None)
        ns = dict(torch=torch, BAR=bar, FULL_SIGMAS=[1,.9,.8,.7,.6,.5,.4,.3,0],
                  ltx_distilled=types.SimpleNamespace(), Tensor=Tensor)
        exec(compile(module, 'numeric-guard', 'exec'), ns)
        return ns

    def test_invalid_reference_cannot_be_sanitized_into_success(self):
        ns = self.functions(); encoder=types.SimpleNamespace(parameters=lambda:iter([types.SimpleNamespace(dtype='fp32',device='cpu')]),to=lambda **k:None)
        fn=lambda **kwargs:[types.SimpleNamespace(latent=ns['Tensor']([float('nan')]))]
        wrapped=ns['_wrap_conditioner'](fn,'test')
        with self.assertRaisesRegex(RuntimeError,'NaN/Inf'):
            wrapped(video_encoder=encoder,dtype='fp16',device='cpu',images=['reference'])

    def test_finite_conditioning_is_retained(self):
        ns=self.functions(); tensor=ns['Tensor']([1.,2.]); item=types.SimpleNamespace(latent=tensor)
        wrapped=ns['_wrap_conditioner'](lambda **k:[item],'test')
        self.assertIs(wrapped(dtype='fp32')[0].latent,tensor)

    def test_reduced_step_request_is_rejected(self):
        ns=self.functions()
        self.assertEqual(ns['set_denoising_steps'](8),8)
        with self.assertRaisesRegex(ValueError,'eight-step'):
            ns['set_denoising_steps'](4)


class NativeInterfaceTests(unittest.TestCase):
    def test_actual_gradio_image_controls_and_http_config(self):
        with patch.dict(os.environ, {'GRADIO_ANALYTICS_ENABLED': 'False'}):
            import gradio as gr
            from fastapi.testclient import TestClient
            tree=ast.parse((HERE/'engine.py').read_text())
            ns=dict(gr=gr,BRAND_HTML='AIQUEST / JJP LTX-2.5',STATUS_HTML='',FOOTER_HTML='',EXAMPLES=[])
            for node in tree.body:
                constants=isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in
                    ('RESOLUTIONS','ASPECTS','DURATIONS') for x in node.targets)
                interface=isinstance(node,ast.FunctionDef) and node.name=='_t2v_ui'
                if constants or interface:
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'native-ui','exec'),ns)
            def sentinel(*args):
                raise AssertionError('UI/config inspection must not generate a video')
            for name in ('start_generation','poll_job','stop_generation','load_latest_video'):
                ns[name]=sentinel
            with gr.Blocks() as demo:
                ns['_t2v_ui']()
            response=TestClient(gr.routes.App.create_app(demo)).get('/config')
            self.assertEqual(response.status_code,200)
            config=response.json()
            images=[c['props'].get('label') for c in config['components'] if c['type']=='image']
            self.assertEqual(images,['Start frame','End frame'])
            sliders=[c for c in config['components'] if c['type']=='slider']
            self.assertEqual(len(sliders),1)
            self.assertEqual(sliders[0]['props']['value'],8)
            self.assertFalse(sliders[0]['props']['interactive'])
            self.assertGreaterEqual(len(config['dependencies']),6)


class ActualSdkTests(unittest.TestCase):
    def exercise(self, reject=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); inp=root/'input';inp.mkdir();(inp/'a').write_bytes(b'a')
            env={'KAGGLE_API_TOKEN':'','KAGGLE_USERNAME':'jjp-sdk-test','KAGGLE_KEY':'not-a-real-key','KAGGLE_CONFIG_DIR':str(root/'config')}
            with patch.dict(os.environ,env),patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')):
                from ltx25_check_dataset import check_runtime
                api=check_runtime()();api.authenticate()
                from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetResponse
                from requests import HTTPError, Response
                contents={'a':b'a','b':b'b'}; catalog={'assets':[asset(n,v) for n,v in contents.items()]}
                uploaded={}; submitted=[]
                def blob(path,*a,**k):
                    p=Path(path);uploaded[p.name]=p.read_bytes();return 'token-'+p.name
                def submit(request):
                    submitted.append(request); self.assertTrue(request.is_private)
                    self.assertEqual(len(request.files),3)
                    result=ApiCreateDatasetResponse();result.status='error' if reject else 'success'
                    result.error='fixture rejection' if reject else '';return result
                def download(a,target):Path(target).write_bytes(contents[a['name']]);return Path(target)
                def status(_):
                    response=Response();response.status_code=403
                    raise HTTPError('fixture status unavailable',response=response)
                client=types.SimpleNamespace(datasets=types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(create_dataset=submit)))
                original=api.upload_files
                with patch.object(api,'dataset_status',side_effect=status),patch.object(api,'_upload_blob',side_effect=blob),patch.object(api,'build_kaggle_client',side_effect=lambda:nullcontext(client)),patch.object(dataset,'transfer',side_effect=download):
                    if reject:
                        with self.assertRaisesRegex(RuntimeError,'not confirmed'):
                            dataset.create_dataset(catalog,root/'out',inp,root/'cache',api=api)
                    else:
                        url=dataset.create_dataset(catalog,root/'out',inp,root/'cache',api=api)
                        self.assertIn('/jjp-ltx25-',url)
                        dataset.create_dataset(catalog,root/'out',inp,root/'cache',api=api)
                self.assertEqual(len(submitted),1)
                self.assertEqual(uploaded['a'],b'a');self.assertEqual(uploaded['b'],b'b')
                self.assertEqual((inp/'a').read_bytes(),b'a')
                self.assertFalse((root/'cache/b').exists())
                self.assertEqual(api.upload_files,original)
                self.assertEqual((root/'out/dataset-upload.json').exists(),not reject)

    def test_native_sdk_private_upload_and_repeat_receipt(self): self.exercise()
    def test_rejected_submission_never_records_success(self): self.exercise(True)


if __name__=='__main__':
    unittest.main(verbosity=2)
