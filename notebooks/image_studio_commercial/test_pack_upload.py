"""Regression tests for Dataset finalisation after successful multi-GB uploads."""
import contextlib,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import asset_builder as a
class HTTPFailure(Exception):
 def __init__(self,status,message='request failed'):super().__init__(message);self.code=status
class API:
 config_values={'username':'testowner'}
 def __init__(self,result=None,existing=False):self.result=result if result is not None else {'status':'success','error':''};self.existing=existing;self.uploads=[];self.downloads=[];self.remote=None
 def dataset_status(self,handle):
  if not self.existing:raise HTTPFailure(404)
  return self.existing if isinstance(self.existing,str) else 'ready'
 def dataset_download_file(self,handle,name,path,**kwargs):self.downloads.append(name);(Path(path)/name).write_text(json.dumps(self.remote))
 def dataset_create_new(self,path,**kwargs):
  root=Path(path);meta=json.loads((root/'dataset-metadata.json').read_text())
  self.uploads.append((set(p.name for p in root.iterdir()),meta,kwargs))
  assert (root/'face_landmarker.task').stat().st_ino==self.model.stat().st_ino,'large model was copied instead of hardlinked'
  return self.result
@contextlib.contextmanager
def pack():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'create_edit';root.mkdir();data=b'tiny verified test model';name='face_landmarker.task';(root/name).write_bytes(data)
  sha=hashlib.sha256(data).hexdigest();record=dict(path=name,bytes=len(data),sha256=sha)
  m=dict(profile='commercial-v7',pack='create_edit',complete=True,files=[record]);(root/'manifest.json').write_text(json.dumps(m))
  (root/'license.txt').write_text('retained notice')
  with patch.object(a,'selected_assets',return_value=[('mediapipe',name,'https://unused',sha,len(data))]),patch.object(a,'LICENSE_FILES',{'license.txt':'retained notice'}):yield root,m
class Tests(unittest.TestCase):
 def test_create_edit_slug_is_valid_before_upload(self):
  with pack() as (root,m):
   api=API();api.model=root/'face_landmarker.task';a.save_private_dataset(root,api=api)
   files,metadata,kwargs=api.uploads[0];handle=metadata['id'];self.assertNotIn('_',handle.split('/')[1]);self.assertIn('create-edit',handle);self.assertFalse(kwargs['public'])
   self.assertEqual(files,{'face_landmarker.task','manifest.json','dataset-metadata.json','license.txt'})
 def test_upload_excludes_internal_state(self):
  with pack() as (root,m):
   for name in ['.builder.lock','manifest.previous.json','dataset-error.json']: (root/name).write_text('internal')
   api=API();api.model=root/'face_landmarker.task';a.save_private_dataset(root,api=api)
   self.assertTrue(all(n not in api.uploads[0][0] for n in ['.builder.lock','manifest.previous.json','dataset-error.json']))
 def test_rejected_creation_keeps_reason_and_models_without_receipt(self):
  with pack() as (root,m):
   api=API({'status':'error','error':'Invalid slug'});api.model=root/'face_landmarker.task'
   with self.assertRaisesRegex(RuntimeError,'Invalid slug'):a.publish_pack(root,api=api)
   self.assertTrue(api.model.is_file());self.assertFalse((root/'dataset-upload.json').exists());self.assertIn('Invalid slug',(root/'dataset-error.json').read_text())
 def test_receipt_after_success_skips_second_upload(self):
  with pack() as (root,m):
   api=API();api.model=root/'face_landmarker.task';first=a.save_private_dataset(root,api=api);self.assertEqual(first,a.save_private_dataset(root,api=api));self.assertEqual(len(api.uploads),1)
 def test_forbidden_lookup_does_not_block_direct_private_create(self):
  with pack() as (root,m):
   api=API();api.model=root/'face_landmarker.task'
   def forbidden_lookup(handle):raise AssertionError('Builder must not call datasets.get before create')
   api.dataset_status=forbidden_lookup
   a.save_private_dataset(root,api=api)
   self.assertEqual(len(api.uploads),1);self.assertFalse(api.uploads[0][2]['public']);self.assertEqual(api.downloads,[])
 def test_existing_dataset_is_not_overwritten_or_false_success(self):
  with pack() as (root,m):
   api=API({'status':'error','error':'The requested slug is already in use by a dataset.'});api.model=root/'face_landmarker.task'
   with self.assertRaisesRegex(RuntimeError,'already in use'):a.save_private_dataset(root,api=api)
   self.assertFalse((root/'dataset-upload.json').exists());self.assertTrue(api.model.is_file())
 def test_actual_creation_permission_error_preserves_reason(self):
  with pack() as (root,m):
   api=API()
   def reject(*args,**kwargs):raise HTTPFailure(403,"Permission 'datasets.create' was denied")
   api.dataset_create_new=reject
   with self.assertRaisesRegex(RuntimeError,'datasets.create'):a.publish_pack(root,api=api)
   self.assertFalse((root/'dataset-upload.json').exists());self.assertTrue((root/'face_landmarker.task').is_file())
 def test_direct_create_handles_sdk_response_object(self):
  with pack() as (root,m):
   from types import SimpleNamespace
   api=API(SimpleNamespace(status='success',error=None));api.model=root/'face_landmarker.task';a.save_private_dataset(root,api=api)
   self.assertTrue(json.loads((root/'dataset-upload.json').read_text())['requested_private'])
 def test_error_redacts_secrets(self):
  with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'supersecret','KAGGLE_KEY':'legacysecret'}):
   text=a.kaggle_error_text('Server rejected supersecret legacysecret Bearer abcdefgh token=xyz')
   for secret in ['supersecret','legacysecret','abcdefgh','xyz']:self.assertNotIn(secret,text)
if __name__=='__main__':unittest.main()
