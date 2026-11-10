import unittest,tempfile,hashlib
from pathlib import Path
from startup_models import prepare_models
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.inp=self.root/'input';self.inp.mkdir();self.out=self.root/'out';self.log=self.root/'calls'
  self.assets=[dict(pack=p,name=n,bytes=len(n),sha256=hashlib.sha256(n.encode()).hexdigest()) for p,n in [('create_edit','core'),('finish','upscale'),('finish','face')]]
  self.source=f'''from pathlib import Path
import hashlib
log=Path({str(self.log)!r})
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_pack(pack,output_root,input_root):
 log.write_text('build')
 root=Path(output_root)/pack;root.mkdir(parents=True,exist_ok=True)
 for n in ['upscale','face']:
  p=root/n
  if not p.exists():p.write_bytes(n.encode())
def publish_pack(root):log.write_text(log.read_text()+' publish')
def kaggle_error_text(e):return str(e)
'''
 def tearDown(self):self.tmp.cleanup()
 def save(self,name,root=None):p=(root or self.inp)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());return p
 def run_models(self,**kwargs):return prepare_models(self.assets,self.source,[self.inp],self.out,**kwargs)
 def test_all_attached_no_builder_or_upload(self):
  for a in self.assets:self.save(a['name'])
  self.assertEqual(len(self.run_models()),3);self.assertFalse(self.log.exists())
 def test_missing_core_stops_before_download(self):
  with self.assertRaisesRegex(RuntimeError,'create_edit'):self.run_models()
  self.assertFalse(self.log.exists())
 def test_exact_missing_finish_failure_now_prepares_and_publishes(self):
  self.save('core');result=self.run_models();self.assertEqual(set(result),{'core','upscale','face'});self.assertEqual(self.log.read_text(),'build publish')
 def test_publication_failure_retains_models_and_continues(self):
  self.save('core');self.source+='\ndef publish_pack(root):raise RuntimeError("Dataset upload unavailable")\n'
  result=self.run_models();self.assertEqual(len(result),3);self.assertTrue((self.out/'finish/face').is_file())
 def test_explicit_disable_keeps_strict_behavior(self):
  self.save('core')
  with self.assertRaisesRegex(RuntimeError,'AUTO_PREPARE_FINISH_PACK'):self.run_models(auto_finish=False)
  self.assertFalse(self.log.exists())
 def test_reuses_already_found_finishing_model_without_copying_bytes(self):
  self.save('core');saved=self.save('upscale');result=self.run_models()
  self.assertEqual(saved.stat().st_ino,(self.out/'finish/upscale').stat().st_ino);self.assertEqual(len(result),3)
if __name__=='__main__':unittest.main()
