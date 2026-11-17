from contextlib import nullcontext
from pathlib import Path
import hashlib,importlib.util,io,json,os,socket,sys,tempfile,types,unittest
from unittest.mock import patch
R=Path(__file__).parent
sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('flash_dataset',R/'dataset.py')
dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
def asset(name,data):return dict(name=name,file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source='https://invalid.example/'+name)
@unittest.skipUnless(importlib.util.find_spec('kaggle'),'real Kaggle SDK required')
class NativeSdk(unittest.TestCase):
    def exercise(self,reject=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input';inp.mkdir();(inp/'a').write_bytes(b'a')
            env={'KAGGLE_API_TOKEN':'','KAGGLE_USERNAME':'jjp-sdk-test','KAGGLE_KEY':'not-a-real-key','KAGGLE_CONFIG_DIR':str(root/'config')}
            with patch.dict(os.environ,env),patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden in SDK test')):
                from flash_check_dataset import check_runtime
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


if __name__=='__main__':unittest.main(verbosity=2)
