import unittest,tempfile,pathlib,json,hashlib,threading,http.server,functools,ast
R=pathlib.Path(__file__).parent
source=(R/'asset_builder.py').read_text().replace('ASSET_CATALOG_SOURCE = None',(R/'assets.py').read_text())
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
  self.raw=b'model-test-contents'*64;(self.root/'tiny.bin').write_bytes(self.raw)
  handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(self.root))
  self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
  self.url=f'http://127.0.0.1:{self.server.server_port}/tiny.bin'
  self.ns={'__name__':'test_builder'};exec(compile(source,'builder','exec'),self.ns)
  self.asset=('checkpoints','tiny.bin',self.url,hashlib.sha256(self.raw).hexdigest(),len(self.raw))
  self.ns['selected_assets']=lambda pack:[self.asset]
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
 def test_complete_and_reuse(self):
  out=self.root/'output';self.ns['build_pack']('standard',out,self.root/'inputs')
  self.assertTrue(json.loads((out/'standard/manifest.json').read_text())['complete'])
  # No HTTP access on a second run with verified existing weights.
  self.ns['urllib'].request.urlopen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected download'))
  try:self.ns['build_pack']('standard',out,self.root/'inputs')
  finally:import importlib;importlib.reload(self.ns['urllib'].request)
 def test_partial_resume_or_restart(self):
  target=self.root/'output/tiny.bin';target.parent.mkdir()
  target.with_name('tiny.bin.partial').write_bytes(self.raw[:100])
  self.ns['transfer'](self.url,target,self.asset[3],len(self.raw))
  self.assertEqual(target.read_bytes(),self.raw)
 def test_corrupt_existing_preserved(self):
  out=self.root/'output';dest=out/'standard/tiny.bin';dest.parent.mkdir(parents=True);dest.write_bytes(b'bad')
  with self.assertRaises(RuntimeError):self.ns['build_pack']('standard',out,self.root/'inputs')
  self.assertEqual(dest.read_bytes(),b'bad');self.assertFalse((out/'standard/manifest.json').exists())
 def test_input_copied_not_linked(self):
  inp=self.root/'input';inp.mkdir();(inp/'tiny.bin').write_bytes(self.raw)
  self.ns['build_pack']('standard',self.root/'output',inp)
  dest=self.root/'output/standard/tiny.bin'
  self.assertFalse(dest.is_symlink());self.assertEqual(dest.read_bytes(),self.raw)
 def test_private_dataset_and_reuse(self):
  out=self.root/'output';self.ns['build_pack']('standard',out,self.root/'inputs')
  class API:
   config_values={'username':'testuser'}
   calls=[]
   def dataset_create_new(self,*args,**kw):self.calls.append(kw);return {'error':None}
  api=API();root=out/'standard'
  self.ns['save_private_dataset'](root,api=api)
  self.assertEqual(len(api.calls),1);self.assertIs(api.calls[0]['public'],False);self.assertEqual(api.calls[0]['dir_mode'],'skip')
  self.ns['save_private_dataset'](root,api=api);self.assertEqual(len(api.calls),1)
 def test_failed_dataset_no_success_receipt(self):
  out=self.root/'output';self.ns['build_pack']('standard',out,self.root/'inputs')
  class API:
   config_values={'username':'testuser'}
   def dataset_create_new(self,*args,**kw):return {'error':'already exists'}
  with self.assertRaises(RuntimeError):self.ns['save_private_dataset'](out/'standard',api=API())
  self.assertFalse((out/'standard/dataset-upload.json').exists())
 def test_catalog_and_notebooks(self):
  cat={};exec((R/'assets.py').read_text(),cat)
  self.assertEqual(len(cat['model_assets']()),4)
  self.assertEqual(len([a for a in cat['model_assets'](False,False,False,True) if a[0]=='insightface']),2)
  self.assertEqual(len(cat['model_assets'](False,False,True)),5)
  for n in ['Image_Studio_Model_Packs.ipynb','Image_Studio_Kaggle_v4.ipynb']:
   nb=json.loads((R/n).read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code'];self.assertEqual(len(cells),1)
   code=''.join(cells[0]['source']);compile(code,n,'exec');self.assertIn('def model_assets(',code)
   self.assertEqual(cells[0]['outputs'],[])
  render=''.join(json.loads((R/'Image_Studio_Kaggle_v4.ipynb').read_text())['cells'][1]['source'])
  self.assertIn('REQUIRE_ATTACHED_MODELS = True',render)
if __name__=='__main__':unittest.main()
