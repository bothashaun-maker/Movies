"""Exercise actual Kaggle 2.0.2 creation/uploader method bodies with offline transport."""
import contextlib,hashlib,json,os,tempfile,types,unittest,typing
from pathlib import Path
from unittest.mock import patch
import asset_builder as builder
R=Path(__file__).parent
class HTTPFailure(Exception):pass
@contextlib.contextmanager
def upload_context():yield object()
class API:
 CONFIG_NAME_USER='username';config_values={'username':'testowner'}
 DATASET_METADATA_FILE='dataset-metadata.json';OLD_DATASET_METADATA_FILE='old.json';DATASET_COVER_IMAGE_FILES=[];KERNEL_METADATA_FILE='kernel.json';MODEL_METADATA_FILE='model.json';MODEL_INSTANCE_METADATA_FILE='instance.json'
 def __init__(self):self.calls=0;self.files={};self.observations=[];self.fail_name=None;self.cache=None
 def get_dataset_metadata_file(self,folder):return str(Path(folder)/'dataset-metadata.json')
 def get_or_fail(self,d,k):return d[k]
 def get_or_default(self,d,k,v):return d.get(k,v)
 def dataset_status(self,ref):raise HTTPFailure('403 datasets.get was denied for a new Dataset')
 def _upload_file_or_folder(self,folder,name,*args):
  if name==self.fail_name:raise RuntimeError('simulated upload interruption')
  p=Path(folder)/name;self.files[name]=p.read_bytes()
  if name.endswith(('.safetensors','.pth','.task')):self.observations.append(sorted(p.name for p in self.cache.glob('*') if p.is_file()))
  return types.SimpleNamespace(token=name)
 def _new_file(self,f):return f
 def with_retry(self,f):return f
 @contextlib.contextmanager
 def build_kaggle_client(self):
  def create(request):self.calls+=1;self.request=request;return types.SimpleNamespace(status='success',error='')
  yield types.SimpleNamespace(datasets=types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(create_dataset=create)))
# Execute the original SDK method bodies, not a duplicate made-up upload algorithm.
scope=dict(os=os,json=json,List=typing.List,Dict=typing.Dict,cast=lambda typ,value:value,ApiCreateDatasetResponse=types.SimpleNamespace,ApiCreateDatasetRequest=types.SimpleNamespace,HTTPError=HTTPFailure,ResumableUploadContext=upload_context,ApiBlobType=types.SimpleNamespace(DATASET='dataset'))
exec(compile((R/'testfixtures/kaggle_2_0_2_upload_reference.py').read_text(),'SDK 2.0.2 fixture','exec'),scope)
API.dataset_create_new=scope['dataset_create_new'];API.upload_files=scope['upload_files']
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.inp=self.root/'input';self.inp.mkdir();self.out=self.root/'packs';self.api=API();self.api.cache=self.out/'.all-models-download-cache'
  self.catalog={'assets':[]};self.content={}
  for i,a in enumerate(builder.CATALOG['assets']):
   b=('model'+str(i)).encode();self.content[a['name']]=b;self.catalog['assets'].append({**a,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 def tearDown(self):self.tmp.cleanup()
 def run_upload(self,free=512*1024**2+6):
  def transfer(url,target,digest,size):
   target=Path(target);target.write_bytes(self.content[target.name]);self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),digest)
  with patch.object(builder,'CATALOG',self.catalog),patch.object(builder,'LICENSE_FILES',{'license.txt':'notice'}),patch.object(builder,'transfer',side_effect=transfer),patch('shutil.disk_usage',return_value=types.SimpleNamespace(free=free)):
   return builder.create_complete_dataset(self.out,self.inp,api=self.api)
 def test_all_models_one_real_sdk_create_with_less_disk_than_total(self):
  url=self.run_upload();self.assertIn('-all-',url);self.assertEqual(self.api.calls,1);self.assertTrue(self.api.request.is_private)
  self.assertEqual(set(self.content),set(self.api.files)&set(self.content));self.assertEqual(len(self.api.request.files),10)
  self.assertTrue(all(len(files)<=1 for files in self.api.observations));self.assertEqual(list(self.api.cache.iterdir()),[])
  self.assertTrue(json.loads(self.api.files['manifest.json'])['complete'])
 def test_existing_inputs_never_copied_or_deleted(self):
  for n,b in self.content.items():(self.inp/n).write_bytes(b)
  with patch('shutil.copyfile',side_effect=AssertionError('No copies')),patch('urllib.request.urlopen',side_effect=AssertionError('No downloads')):self.run_upload(free=17*1024**2)
  self.assertEqual(len(list(self.inp.iterdir())),8);self.assertTrue(all((self.inp/n).read_bytes()==b for n,b in self.content.items()))
 def test_existing_working_pack_is_reused(self):
  existing=self.out/'create_edit';existing.mkdir(parents=True)
  for n,b in self.content.items():(existing/n).write_bytes(b)
  self.run_upload(free=17*1024**2);self.assertEqual(len(list(existing.iterdir())),8)
 def test_failure_retains_current_download_and_restores_uploader(self):
  self.api.fail_name=self.catalog['assets'][1]['name'];original=self.api.upload_files
  with self.assertRaisesRegex(RuntimeError,'simulated upload interruption'):self.run_upload()
  self.assertEqual(self.api.calls,0);self.assertTrue((self.api.cache/self.api.fail_name).exists());self.assertEqual(self.api.upload_files,original)
  self.assertFalse((self.out/'all/dataset-upload.json').exists())
 def test_receipt_prevents_second_dataset_creation(self):
  self.run_upload();self.run_upload();self.assertEqual(self.api.calls,1)
 def test_not_enough_for_one_model_stops_without_upload(self):
  with self.assertRaisesRegex(RuntimeError,'largest missing model'):self.run_upload(free=0)
  self.assertEqual(self.api.calls,0);self.assertEqual(self.api.files,{})
if __name__=='__main__':unittest.main()
