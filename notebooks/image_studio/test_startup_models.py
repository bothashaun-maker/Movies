import unittest,tempfile,pathlib,hashlib,json,http.server,threading,functools,importlib.util
R=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('startup',R/'startup_models.py');startup=importlib.util.module_from_spec(spec);spec.loader.exec_module(startup)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
  self.inputs=self.root/'input';self.inputs.mkdir();self.pack=self.root/'saved';self.runtime=self.root/'runtime'
  self.downloads=[];record=self.downloads
  class Handler(http.server.SimpleHTTPRequestHandler):
   def do_GET(self):record.append(self.path);super().do_GET()
   def log_message(self,*args):pass
  handler=functools.partial(Handler,directory=str(self.root))
  self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
  self.faces=[]
  for name,content in [('inswapper_128.onnx',b'swap fixture'*20),('buffalo_l.zip',b'archive fixture'*30)]:
   (self.root/name).write_bytes(content);self.faces.append(('insightface',name,f'http://127.0.0.1:{self.server.server_port}/'+name,hashlib.sha256(content).hexdigest(),len(content)))
  core=b'core checkpoint';(self.inputs/'generation.safetensors').write_bytes(core)
  self.core=('checkpoints','generation.safetensors','https://invalid.test/never-fetch',hashlib.sha256(core).hexdigest(),len(core))
  self.builder={'__name__':'test_support'}
  source=(R/'asset_builder.py').read_text().replace('ASSET_CATALOG_SOURCE = None',(R/'assets.py').read_text());exec(source,self.builder)
  self.builder['selected_assets']=lambda pack:self.faces
  self.saved=[];self.builder['save_private_dataset']=lambda root:self.saved.append(root)
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
 def run_preflight(self,assets=None,**kwargs):return startup.prepare_startup_models(assets or [self.core]+self.faces,'',input_root=self.inputs,runtime_root=self.runtime,pack_root=self.pack,builder=self.builder,**kwargs)
 def test_reported_missing_faces_repairs_only_faces(self):
  result=self.run_preflight();self.assertEqual(len(self.downloads),2);self.assertEqual(len(self.saved),1)
  self.assertEqual(result[self.core[1]],[self.inputs/self.core[1]])
  for a in self.faces:self.assertTrue(result[a[1]][0].is_file())
  self.assertTrue(json.loads((self.pack/'faceswap/manifest.json').read_text())['complete'])
 def test_existing_attached_faces_no_download_no_upload(self):
  for a in self.faces:(self.inputs/a[1]).write_bytes((self.root/a[1]).read_bytes())
  self.run_preflight();self.assertEqual(self.downloads,[]);self.assertEqual(self.saved,[])
 def test_missing_one_reuses_other(self):
  a=self.faces[0];(self.inputs/a[1]).write_bytes((self.root/a[1]).read_bytes())
  self.run_preflight();self.assertEqual(self.downloads,['/buffalo_l.zip'])
 def test_missing_generation_stays_strict(self):
  (self.inputs/self.core[1]).unlink()
  with self.assertRaisesRegex(RuntimeError,'standard/FLUX3'):self.run_preflight()
  self.assertEqual(self.downloads,[])
 def test_upload_failure_keeps_ready_models(self):
  def failure(root):raise RuntimeError('simulated API error')
  self.builder['save_private_dataset']=failure
  result=self.run_preflight()
  self.assertTrue(result['inswapper_128.onnx'][0].is_file())
 def test_auto_prepare_can_be_disabled(self):
  with self.assertRaisesRegex(RuntimeError,'AUTO_PREPARE_FACE_PACK'):self.run_preflight(prepare_faces=False)
  self.assertEqual(self.downloads,[])
 def test_repeat_cached_pack_no_redownload(self):
  self.run_preflight();self.downloads.clear();self.run_preflight();self.assertEqual(self.downloads,[])
 def test_faces_disabled_need_no_face_files(self):
  self.run_preflight(assets=[self.core]);self.assertEqual(self.downloads,[])
 def test_generated_notebook_embeds_helper_and_builder(self):
  n=json.loads((R/'Image_Studio_Kaggle_v5.ipynb').read_text());code=''.join(n['cells'][1]['source']);compile(code,'v5','exec')
  self.assertIn('def prepare_startup_models(',code);self.assertIn('AUTO_PREPARE_FACE_PACK = True',code)
  self.assertNotIn('PACK_BUILDER_SOURCE = None',code);self.assertNotIn('STARTUP_MODELS_SOURCE = None',code)
if __name__=='__main__':unittest.main()
