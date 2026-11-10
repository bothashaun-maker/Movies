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
 def test_remote_dataset_without_local_receipt_reuses_manifest_only(self):
  with pack() as (root,m):
   api=API(existing=True);api.remote=m;api.model=root/'face_landmarker.task';a.save_private_dataset(root,api=api)
   self.assertEqual(api.uploads,[]);self.assertEqual(api.downloads,['manifest.json']);self.assertTrue(json.loads((root/'dataset-upload.json').read_text())['verified_remote_manifest'])
 def test_processing_dataset_is_not_reuploaded(self):
  with pack() as (root,m):
   api=API(existing='processing')
   with self.assertRaisesRegex(RuntimeError,'processing'):a.save_private_dataset(root,api=api)
   self.assertEqual(api.uploads,[])
 def test_existing_wrong_manifest_is_not_overwritten(self):
  with pack() as (root,m):
   api=API(existing=True);api.remote={**m,'files':[]}
   with self.assertRaises(RuntimeError):a.save_private_dataset(root,api=api)
   self.assertEqual(api.uploads,[])
 def test_permission_failure_does_not_start_upload(self):
  with pack() as (root,m):
   api=API();api.dataset_status=lambda handle:(_ for _ in ()).throw(HTTPFailure(403,'quota or access denied'))
   with self.assertRaisesRegex(RuntimeError,'quota or access denied'):a.save_private_dataset(root,api=api)
   self.assertEqual(api.uploads,[])
 def test_error_redacts_secrets(self):
  with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'supersecret','KAGGLE_KEY':'legacysecret'}):
   text=a.kaggle_error_text('Server rejected supersecret legacysecret Bearer abcdefgh token=xyz')
   for secret in ['supersecret','legacysecret','abcdefgh','xyz']:self.assertNotIn(secret,text)
if __name__=='__main__':unittest.main()
