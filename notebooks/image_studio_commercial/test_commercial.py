import unittest,json,tempfile,sys,hashlib,types,os
from pathlib import Path
from PIL import Image,PngImagePlugin
import numpy as np
R=Path(__file__).parent;sys.path.insert(0,str(R))
import asset_builder as a
from audit_outputs import audit
class CommercialTests(unittest.TestCase):
 def test_allowlist_and_dependency_chain(self):
  catalog=json.loads((R/'catalog.json').read_text());self.assertEqual(len(catalog['assets']),8)
  self.assertEqual({x['license'] for x in catalog['assets']},{'Apache-2.0','BSD-3-Clause','MIT'})
  self.assertFalse(any(t in json.dumps(catalog).lower() for t in ['inswapper','buffalo','flux-3','juggernaut','arcface']))
  for pack in ['create_edit','finish']:
   records=[dict(path=n,bytes=z,sha256=d) for _,n,_,d,z in a.selected_assets(pack)]
   m=dict(profile='commercial-v7',pack=pack,complete=True,files=records);self.assertTrue(a.validate_manifest(m));m['files'].append(dict(path='inswapper_128.onnx',bytes=1,sha256='a'))
   with self.assertRaises(RuntimeError):a.validate_manifest(m)
 def test_no_extra_weights_or_incomplete_upload(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'inswapper_128.onnx').write_bytes(b'x')
   with self.assertRaises(RuntimeError):a.reject_extra_models(p,'finish')
   (p/'manifest.json').write_text(json.dumps(dict(profile='commercial-v7',complete=False)))
   with self.assertRaises(RuntimeError):a.save_private_dataset(p,api=object())
 def test_reuse_without_network_and_licenses(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);inp=root/'input';inp.mkdir();data=b'test verified model';(inp/'face_landmarker.task').write_bytes(data)
   old=a.selected_assets;oldtransfer=a.transfer
   a.selected_assets=lambda pack:[('mediapipe','face_landmarker.task','https://unused',hashlib.sha256(data).hexdigest(),len(data))]
   calls=[]
   a.transfer=lambda url,target,digest,size:calls.append(Path(target).read_bytes())
   try:
    m=a.build_pack('finish',root/'out',inp);self.assertEqual(calls,[data]);self.assertTrue(m['complete'])
    self.assertFalse((root/'out/finish/face_landmarker.task').is_symlink());self.assertTrue((root/'out/finish/realesrgan-BSD.txt').is_file())
   finally:a.selected_assets=old;a.transfer=oldtransfer
 def test_dataset_submission_is_private_and_reuses_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);data=b'checked';name='face_landmarker.task';(root/name).write_bytes(data);digest=hashlib.sha256(data).hexdigest()
   old=a.selected_assets;a.selected_assets=lambda pack:[('mediapipe',name,'https://unused',digest,len(data))]
   class API:
    config_values={'username':'testowner'}
    calls=[]
    def dataset_create_new(self,*args,**kwargs):self.calls.append((args,kwargs));return {}
   api=API()
   try:
    (root/'manifest.json').write_text(json.dumps(dict(profile='commercial-v7',pack='finish',complete=True,files=[dict(path=name,bytes=len(data),sha256=digest)])))
    for n,c in a.LICENSE_FILES.items():(root/n).write_text(c)
    url=a.save_private_dataset(root,api=api);self.assertIn('testowner/image-studio-commercial-finish-',url);self.assertFalse(api.calls[0][1]['public'])
    self.assertEqual(a.save_private_dataset(root,api=api),url);self.assertEqual(len(api.calls),1)
   finally:a.selected_assets=old
 def test_readonly_output_triage(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'image.png';meta=PngImagePlugin.PngInfo();meta.add_text('prompt','StudioFaceSwap inswapper_128.onnx');Image.new('RGB',(8,8)).save(p,pnginfo=meta);before=p.read_bytes()
   self.assertEqual(audit(p)['status'],'restricted-face-pipeline-review');self.assertEqual(before,p.read_bytes())
   Image.new('RGB',(8,8)).save(p);self.assertEqual(audit(p)['status'],'unknown-or-incomplete-provenance')
 def test_isolated_runtime(self):
  s=(R/'launcher.py').read_text();self.assertIn('image-studio-commercial-v7',s);self.assertIn('--disable-all-custom-nodes',s)
  self.assertNotIn('import insightface',s);self.assertNotIn('hf_hub_download',s)
 def test_native_duet_three_references(self):
  workflows=json.loads((R/'workflows.json').read_text())
  for name in ['Duet Identity Edit','Approved Duet Identity Edit']:
   types=[n['type'] for n in workflows[name]['nodes']];self.assertEqual(types.count('VAEEncode'),3,name);self.assertEqual(types.count('ReferenceLatent'),6,name)
 def test_embedded_pack_notices_and_catalog(self):
  nb=json.loads((R/'Image_Studio_Commercial_Model_Packs.ipynb').read_text());code=''.join(nb['cells'][1]['source']);compile(code,'pack','exec')
  self.assertIn('MediaPipe model license evidence',code);self.assertIn('ec3d4e733a771f61',code);self.assertNotIn('inswapper_128.onnx',code)
if __name__=='__main__':unittest.main()
