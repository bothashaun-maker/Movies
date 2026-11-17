"""Offline source/control/media checks; no full weights or GPU inference."""
import ast,base64,hashlib,importlib.util,io,json,os,subprocess,sys,tempfile,time,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).parent;REPO=R.parent.parent
sys.path.insert(0,str(R));import runtime
UPSTREAM=Path(os.environ.get('AIQ_REFERENCE','/workspace/work/aiq-upscale-reference/verified.ipynb'))

class Controls(unittest.TestCase):
 def test_pinned_source_and_inference_unchanged(self):
  original=runtime.extract_engine(UPSTREAM.read_bytes());changed=runtime.prepare_engine(original)
  self.assertEqual(runtime.inference_identity(original),runtime.inference_identity(changed))
  values={}
  for n in ast.parse(changed).body:
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
    try:values[n.targets[0].id]=ast.literal_eval(n.value)
    except:pass
  self.assertEqual(values['SIZE_CHOICES'][0],'2x upscale')
  with self.assertRaisesRegex(RuntimeError,'identity differs'):runtime.extract_engine(UPSTREAM.read_bytes()+b' ')
 def test_notebooks_and_payloads_match(self):
  import nbformat
  for name in ('JJP_FlashVSR_T4.ipynb','JJP_FlashVSR_Models.ipynb'):
   nb=nbformat.read(REPO/name,as_version=4);nbformat.validate(nb);self.assertEqual(sum(c.cell_type=='code' for c in nb.cells),1)
   source=nb.cells[1].source;compile(source,name,'exec')
   call=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and ast.unparse(n.func)=='base64.b64decode')
   with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0])))) as z:
    for file in z.namelist():
     if file.endswith('.py'):compile(z.read(file),file,'exec')
    self.assertEqual(z.read('flash_runtime.py').decode(),(R/'runtime.py').read_text())
   if 'Models' in name:self.assertLess(source.index('api=authenticated_api()'),source.index('cache=choose_scratch'))
 def test_single_complete_dataset_and_no_mixing(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c={'assets':[{'name':'a'},{'name':'b'}]}
   for name in ('a','b'):
    p=root/name;p.mkdir();(p/'manifest.json').write_text(json.dumps(c));(p/name).touch()
   with self.assertRaisesRegex(RuntimeError,'ONE complete'):runtime.find_dataset(c,root)
   (root/'a/b').touch();self.assertEqual(runtime.find_dataset(c,root),root/'a')
 def test_wrong_or_busy_gpus_fail_before_any_model_activity(self):
  for data in ['Tesla T4, 14000, 15000\n','Tesla T4, 100, 15000\nTesla T4, 14000, 15000\n']:
   with patch.object(runtime.subprocess,'check_output',return_value=data):
    with self.assertRaises(RuntimeError):runtime.check_gpus()
 def test_budget_limits_are_stops_not_assumed_speed(self):
  self.assertIsNone(runtime.budget_reason(10,None,360,480))
  self.assertIsNotNone(runtime.budget_reason(361,None,360,480))
  self.assertIsNotNone(runtime.budget_reason(700,{'active':True,'elapsed':481},360,480))
  self.assertIsNone(runtime.budget_reason(900,{'active':False,'elapsed':900},360,480))
 def test_stop_only_exact_owned_engine(self):
  import psutil
  with tempfile.TemporaryDirectory() as d:
   work=Path(d);script=work/'run_flashvsr.py';script.write_text('import time;time.sleep(60)')
   owned=subprocess.Popen([sys.executable,str(script)]);other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
   try:
    (work/'engine-process.json').write_text(json.dumps({'pid':owned.pid,'created':psutil.Process(owned.pid).create_time()}))
    runtime.stop_owned(work);owned.wait(timeout=5);self.assertIsNone(other.poll())
    (work/'engine-process.json').write_text(json.dumps({'pid':other.pid,'created':psutil.Process(other.pid).create_time()}))
    with self.assertRaisesRegex(RuntimeError,'identity differs'):runtime.stop_owned(work)
    self.assertIsNone(other.poll())
   finally:
    for p in (owned,other):
     if p.poll() is None:p.terminate();p.wait(timeout=5)
 def test_pinned_catalog_complete_and_distinct(self):
  c=json.loads((R/'assets.json').read_text());a=c['assets']
  self.assertEqual(len(a),5);self.assertEqual(len({x['name'] for x in a}),5)
  self.assertEqual(sum(x['bytes'] for x in a),13843927661)
  for x in a:self.assertRegex(x['sha256'],r'^[0-9a-f]{64}$');self.assertIn(c['revision'],x['source'])

class NativeCPU(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  os.environ.update(FLASHVSR_ENGINE_TEST='1',FLASHVSR_REPO=os.environ.get('FLASHVSR_REPO','/workspace/work/flashvsr-validation'),FLASHVSR_WORK_DIR='/workspace/work/native-flash-tests')
  source=runtime.extract_engine(UPSTREAM.read_bytes());source=runtime.prepare_engine(source)
  tree=ast.parse(source);prefix=[]
  for node in tree.body:
   if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='T_LOAD' for n in node.targets):break
   prefix.append(node)
  cls.engine={};exec(compile(ast.Module(body=prefix,type_ignores=[]),'actual-TeamAIQ-definitions','exec'),cls.engine)
 def test_native_sparse_attention_matches_selected_dense_sdpa(self):
  import torch
  torch.manual_seed(4)
  q=torch.randn(1,2,3,128,16);k=torch.randn(1,2,4,128,16);v=torch.randn_like(k)
  mask=torch.tensor([[[[1,0,1,0],[0,1,0,1],[1,1,1,1]],[[1,1,0,0],[0,1,1,0],[0,0,0,1]]]],dtype=torch.bool)
  expected=torch.empty_like(q)
  for h in range(2):
   for b in range(3):
    kk=k[0,h,mask[0,h,b]].reshape(1,-1,16);vv=v[0,h,mask[0,h,b]].reshape(1,-1,16)
    expected[0,h,b]=torch.nn.functional.scaled_dot_product_attention(q[0,h,b][None],kk,vv)[0]
  for method in ('sdpa','matmul'):
   result=self.engine['torch_sparse_attention'](q,k,v,mask,method=method,budget_bytes=100000)
   torch.testing.assert_close(result,expected,atol=2e-5,rtol=2e-5)
  result=self.engine['varlen_sparse_attention'](q,k,v,mask,budget_bytes=100000)
  torch.testing.assert_close(result,expected,atol=2e-5,rtol=2e-5)
 def test_native_geometry_padding_and_media_writer(self):
  import numpy as np
  self.assertEqual(self.engine['output_size'](960,544,'2x upscale'),(1920,1088))
  self.assertEqual(self.engine['model_size'](1920,1088),(1152,1920))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'native.mp4';writer=self.engine['VideoWriter'](str(p),128,128,24,crf=14,preset='veryfast')
   writer.push(np.zeros((3,128,128,3),dtype=np.uint8));writer.close()
   info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)]))['streams'][0]
   self.assertEqual((info['width'],info['height'],int(info['nb_read_frames'])),(128,128,3))

if __name__=='__main__':unittest.main(verbosity=2)
