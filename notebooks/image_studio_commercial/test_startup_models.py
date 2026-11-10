"""One complete Dataset is sufficient; renderer never prepares/downloads models."""
import hashlib,tempfile,unittest,json,ast
from pathlib import Path
from startup_models import prepare_models
import asset_builder as builder
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.dataset=self.root/'one-dataset';self.dataset.mkdir()
 def tearDown(self):self.tmp.cleanup()
 def test_all_eight_models_selected_and_manifest_valid(self):
  selected=builder.selected_assets('all');self.assertEqual(len(selected),8)
  self.assertEqual({n for _,n,_,_,_ in selected},{a['name'] for a in builder.CATALOG['assets']})
  manifest=dict(profile='commercial-v7',pack='all',complete=True,files=[dict(path=n,bytes=z,sha256=d) for _,n,_,d,z in selected]);self.assertTrue(builder.validate_manifest(manifest))
  self.assertEqual(sum(a[4] for a in selected),20532327539)
 def test_exactly_one_dataset_resolves_every_model(self):
  assets=[]
  for i,a in enumerate(builder.CATALOG['assets']):
   data=('test model '+str(i)).encode();(self.dataset/a['name']).write_bytes(data);assets.append({**a,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  found=prepare_models(assets,[self.dataset]);self.assertEqual(len(found),8);self.assertTrue(all(p.parent==self.dataset for p in found.values()))
 def test_missing_models_fail_without_preparation(self):
  assets=[dict(name='missing.pth',bytes=1,sha256='x')]
  with self.assertRaisesRegex(RuntimeError,'ALL MODELS Dataset'):prepare_models(assets,[self.dataset])
  self.assertEqual(list(self.dataset.iterdir()),[])
 def test_invalid_hash_is_not_accepted(self):
  p=self.dataset/'model';p.write_bytes(b'bad')
  with self.assertRaises(RuntimeError):prepare_models([dict(name='model',bytes=3,sha256=hashlib.sha256(b'yes').hexdigest())],[self.dataset])
 def test_launcher_cannot_prepare_or_download_model_weights(self):
  source=(R/'launcher.py').read_text();self.assertNotIn('PACK_BUILDER_SOURCE',source);self.assertNotIn('AUTO_PREPARE_FINISH',source)
  tree=ast.parse(source)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='download':self.assertTrue(isinstance(n.args[0],ast.Constant) and any(t in n.args[0].value for t in ('cloudflared','libglvnd')))
 def test_builder_creates_one_dataset_with_all_eight_reusing_saved_files(self):
  from unittest.mock import patch
  catalog={'assets':[]}
  for i,a in enumerate(builder.CATALOG['assets']):
   data=('model'+str(i)).encode();(self.dataset/a['name']).write_bytes(data);catalog['assets'].append({**a,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  class API:
   config_values={'username':'testowner'}
   calls=0
   def dataset_create_new(self,root,**kwargs):
    self.calls+=1;self.models={p.name for p in Path(root).iterdir() if p.suffix in ('.safetensors','.pth','.task')};self.metadata=json.loads((Path(root)/'dataset-metadata.json').read_text());assert kwargs['public'] is False
    return {'status':'success'}
  api=API()
  with patch.object(builder,'CATALOG',catalog),patch('urllib.request.urlopen',side_effect=AssertionError('No network downloads allowed')):
   m=builder.build_pack('all',self.root/'out',self.dataset);self.assertEqual(len(m['files']),8)
   builder.save_private_dataset(self.root/'out/all',api=api)
  self.assertEqual(api.calls,1);self.assertEqual(len(api.models),8);self.assertIn('-all-',api.metadata['id'])
 def test_default_builder_is_all_and_keeps_working_private_create(self):
  self.assertEqual(builder.PACK,'all');s=(R/'asset_builder.py').read_text();self.assertIn('public=False',s);self.assertNotIn('api.dataset_status(',s)
if __name__=='__main__':unittest.main()
