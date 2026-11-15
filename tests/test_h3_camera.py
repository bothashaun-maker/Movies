"""Local controls, real SDK/file tokens and source-preservation checks; no GPU inference."""
from contextlib import nullcontext
from pathlib import Path
import ast,base64,hashlib,importlib.util,io,json,os,re,socket,sys,tempfile,types,unittest,zipfile
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];S=R/'notebooks/h3_camera'
sys.path.insert(0,str(S))
if importlib.util.find_spec('fcntl') is None:
    # Linux-only file locking is not exercised by these Windows CPU checks.
    sys.modules['fcntl']=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8)
import camera_common as common
import camera_dataset as dataset
import camera_bootstrap as bootstrap
from integrate import sources

def asset(name,data):
    return dict(name=name,file='h3_models/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source='https://invalid.example/'+name)

def notebook_payload(name):
    nb=json.loads((R/name).read_text(encoding='utf-8'));code=''.join(nb['cells'][1]['source'])
    call=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    return nb,code,zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0]))))

class CameraBuild(unittest.TestCase):
    def test_notebooks_validate_and_payloads_compile(self):
        import nbformat
        for name in ['JJP_H3_Camera_Models.ipynb','JJP_H3_Native_Camera.ipynb']:
            nb,code,z=notebook_payload(name);nbformat.validate(nbformat.from_dict(nb))
            self.assertEqual(sum(c['cell_type']=='code' for c in nb['cells']),1)
            compile(code,name,'exec')
            for p in z.namelist():
                self.assertEqual(Path(p).name,p)
                if p.endswith('.py'):compile(z.read(p),p,'exec')
        _,cpu,_=notebook_payload('JJP_H3_Camera_Models.ipynb')
        self.assertLess(cpu.index('api=authenticated_api()'),cpu.index('scratch=choose_scratch'))
        self.assertIn('create_dataset(',cpu)

    def test_verified_aiquest_engine_and_generated_code_match(self):
        setup,engine,launcher,digest=sources(S/'aiquest_upstream.ipynb')
        self.assertEqual(digest,'97a413e6d64473609dc3d548bee8e31fd877eaf3cb344a2b75d6dbed5939b8fe')
        for name,expected in [('aiquest_setup.py',setup),('aiquest_engine.py',engine),('aiquest_launcher.py',launcher)]:
            self.assertEqual((S/name).read_text(encoding='utf-8'),expected)
        self.assertNotIn('hf_hub_download',setup)
        self.assertNotIn('p.kill()',setup)
        self.assertIn('remember_camera_engine(_CAMERA_WORK,proc)',launcher)
        self.assertTrue('AIQUEST Academy' in engine,'AIQuest attribution missing')

    def test_catalog_is_complete_unique_and_pinned(self):
        c=json.loads((S/'assets.json').read_text());a=c['assets']
        self.assertEqual(len(a),13);self.assertEqual(len({r['name'] for r in a}),13)
        self.assertEqual(sum(r['name'].endswith('.gguf') for r in a),1)
        for r in a:
            self.assertRegex(r['revision'],r'^[a-f0-9]{40}$');self.assertRegex(r['sha256'],r'^[a-f0-9]{64}$')
            self.assertIn('/resolve/'+r['revision']+'/',r['source'])
        self.assertTrue(any('FL2VA' in r['name'] for r in a))
        self.assertTrue(any('lora_model_0' in r['name'] for r in a))
        self.assertFalse(any('molmo' in r['name'].lower() for r in a))

    def test_pixaroma_structured_first_frame_prompt_is_not_rewritten(self):
        engine=(S/'aiquest_engine.py').read_text(encoding='utf-8')
        f=next(n for n in ast.parse(engine).body if isinstance(n,ast.FunctionDef) and n.name=='build_prompt')
        scope={'_STRUCTURED':re.compile('integrated_multimodal_description|subject_definitions')}
        exec(compile(ast.Module(body=[f],type_ignores=[]),'AIQuest prompt builder','exec'),scope)
        prompt='For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] The camera arcs around the subject with small amplitude at slow speed.\n\noverall_soundscape: Quiet wind.\n\nnon_diegetic_music: N/A'
        result=scope['build_prompt'](prompt,'','English','','',124/24,True,False,0,'',False,[])
        self.assertEqual(result,prompt)

    def test_cannot_mix_incomplete_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c={'assets':[asset('a',b'a'),asset('b',b'b')]}
            for name,model in [('one','a'),('two','b')]:
                p=root/name;p.mkdir();(p/'manifest.json').write_text(json.dumps(c));(p/model).write_bytes(model.encode())
            with self.assertRaisesRegex(RuntimeError,'ONE saved'):bootstrap.find_dataset(c,root)
            (root/'one/b').write_bytes(b'b');self.assertEqual(bootstrap.find_dataset(c,root),root/'one')

    def test_staging_detects_corrupt_same_size_input_before_engine(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input';inp.mkdir();(inp/'a').write_bytes(b'X')
            c={'assets':[asset('a',b'a')]}
            with patch.object(common.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10*common.GIB)):
                with self.assertRaisesRegex(RuntimeError,'failed verification'):common.stage_models(c,inp,root/'cache')
            self.assertEqual((inp/'a').read_bytes(),b'X')

    def test_wrong_gpu_count_fails_before_loading(self):
        result=types.SimpleNamespace(returncode=0,stdout='Tesla T4\n')
        with patch.object(bootstrap.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'T4 x2'):bootstrap.gpu_names()

    def test_scoped_cleanup_leaves_other_projects_alone(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'engine-process.json').write_text(json.dumps({'pid':123,'created':1.}))
            process=types.SimpleNamespace(create_time=lambda:1.,cwd=lambda:str(root/'other-project'))
            fake=types.SimpleNamespace(Process=lambda _:process,NoSuchProcess=ProcessLookupError)
            with patch.dict(sys.modules,{'psutil':fake}):bootstrap.stop_camera_engine(root)

    def test_startup_handoff_uses_verified_saved_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input';inp.mkdir();run=root/'runtime';run.mkdir();cache=root/'cache';cache.mkdir()
            c={'assets':[asset('a',b'a')]};(inp/'manifest.json').write_text(json.dumps(c));(inp/'a').write_bytes(b'a')
            (run/'assets.json').write_text(json.dumps(c))
            (run/'aiquest_engine.py').write_text('print("fixture only; no model or GPU")')
            (run/'aiquest_setup.py').write_text('import os\nassert os.environ["H3_VARIANT"]=="pdmd"\n')
            (run/'aiquest_launcher.py').write_text('assert "fixture only" in RUN_SCRIPT_CODE\n')
            before=Path.cwd()
            with patch.dict(os.environ,{},clear=False),patch.dict(sys.modules,{'psutil':types.SimpleNamespace()}),patch.object(bootstrap,'find_dataset',return_value=inp),patch.object(bootstrap,'gpu_names',return_value=['Tesla T4']*2),patch.object(bootstrap,'stop_camera_engine'),patch.object(bootstrap,'choose_scratch',return_value=cache),patch.object(common.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10*common.GIB)):
                try:bootstrap.main(run,root/'work')
                finally:os.chdir(before)
            self.assertEqual((cache/'h3_models/a').read_bytes(),b'a')
            settings=json.loads((root/'work/h3_settings.json').read_text())
            self.assertEqual(settings['H3_MODELS_DIR'],str(cache/'h3_models'))

@unittest.skipUnless(importlib.util.find_spec('kaggle'),'real Kaggle SDK required')
class NativeSdk(unittest.TestCase):
    def exercise(self,reject=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input';inp.mkdir();(inp/'a').write_bytes(b'a')
            env={'KAGGLE_API_TOKEN':'','KAGGLE_USERNAME':'jjp-sdk-test','KAGGLE_KEY':'not-a-real-key','KAGGLE_CONFIG_DIR':str(root/'config')}
            with patch.dict(os.environ,env),patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden in SDK test')):
                from camera_check_dataset import check_runtime
                api=check_runtime()();api.authenticate()
                from requests import HTTPError,Response
                from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetResponse
                values={'a':b'a','b':b'b'};catalog={'assets':[asset(n,v) for n,v in values.items()]};uploaded={};submitted=[]
                def blob(path,*args,**kw):
                    p=Path(path);uploaded[p.name]=p.read_bytes();return 'token-'+p.name
                def status(_):
                    response=Response();response.status_code=403;raise HTTPError('stub status',response=response)
                def submit(request):
                    submitted.append(request);self.assertTrue(request.is_private);self.assertEqual(len(request.files),3)
                    response=ApiCreateDatasetResponse();response.status='error' if reject else 'success';response.error='stub rejection' if reject else '';return response
                def download(record,target):Path(target).write_bytes(values[record['name']]);return Path(target)
                client=types.SimpleNamespace(datasets=types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(create_dataset=submit)))
                original=api.upload_files
                # Same-filesystem hardlinks stand in for Linux symlinks only in this
                # Windows test; the real native SDK reads and uploads actual bytes.
                with patch.object(api,'dataset_status',side_effect=status),patch.object(api,'_upload_blob',side_effect=blob),patch.object(api,'build_kaggle_client',side_effect=lambda:nullcontext(client)),patch.object(dataset,'transfer',side_effect=download),patch.object(Path,'symlink_to',lambda target,source:os.link(source,target)):
                    if reject:
                        with self.assertRaisesRegex(RuntimeError,'not confirmed'):dataset.create_dataset(catalog,root/'out',inp,root/'cache',api=api)
                    else:dataset.create_dataset(catalog,root/'out',inp,root/'cache',api=api)
                self.assertEqual(len(submitted),1);self.assertEqual(uploaded['a'],b'a');self.assertEqual(uploaded['b'],b'b')
                self.assertEqual((inp/'a').read_bytes(),b'a');self.assertFalse((root/'cache/b').exists())
                self.assertEqual(api.upload_files,original);self.assertEqual((root/'out/dataset-upload.json').exists(),not reject)

    def test_real_private_sdk_upload_and_file_tokens(self):self.exercise()
    def test_native_server_rejection_does_not_record_success(self):self.exercise(True)

if __name__=='__main__':unittest.main()
