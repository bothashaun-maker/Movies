import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

spec=importlib.util.spec_from_file_location('assets',Path(__file__).with_name('assets.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class FakeResponse:
    def __init__(self,data,status=200,headers=None):
        import io
        self.data=io.BytesIO(data);self.status=status;self.headers=headers or {}
    def read(self,*args):return self.data.read(*args)
    def __enter__(self):return self
    def __exit__(self,*args):pass

class AssetChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.data=b'actual model fixture'*128
        self.asset={'name':'fixture.bin','folder':'models','url':'https://example.invalid/model','bytes':len(self.data),'sha256':hashlib.sha256(self.data).hexdigest()}
    def tearDown(self):self.tmp.cleanup()
    def test_resume_and_checksum(self):
        dest=self.root/'model.bin';dest.with_suffix('.bin.partial').write_bytes(self.data[:30])
        with mock.patch.object(a.urllib.request,'urlopen',return_value=FakeResponse(self.data[30:],206,{'Content-Range':f'bytes 30-{len(self.data)-1}/{len(self.data)}'})) as get:
            a.fetch(self.asset,dest)
            self.assertEqual(get.call_args.args[0].get_header('Range'),'bytes=30-')
        self.assertTrue(a.valid(dest,self.asset))
        with mock.patch.object(a.urllib.request,'urlopen',side_effect=AssertionError('unexpected download')):
            a.fetch(self.asset,dest)
    def test_server_ignores_range_restarts(self):
        dest=self.root/'model.bin';dest.with_suffix('.bin.partial').write_bytes(self.data[:30])
        with mock.patch.object(a.urllib.request,'urlopen',return_value=FakeResponse(self.data)):
            a.fetch(self.asset,dest)
        self.assertEqual(dest.read_bytes(),self.data)
    def test_corrupt_completed_file_preserved(self):
        dest=self.root/'model.bin';dest.write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError,'preserved'):a.fetch(self.asset,dest)
        self.assertEqual(dest.read_bytes(),b'bad')
    def test_pack_copies_attached_weight_and_marks_complete(self):
        inp=self.root/'input';inp.mkdir();(inp/'fixture.bin').write_bytes(self.data)
        out=self.root/'output'
        with mock.patch.object(a,'ASSETS',[self.asset]),mock.patch.object(a.urllib.request,'urlopen',side_effect=AssertionError('unexpected download')):
            a.build_pack(out,inp)
        model=out/'models/fixture.bin'
        self.assertFalse(model.is_symlink());self.assertEqual(model.read_bytes(),self.data)
        self.assertTrue(json.loads((out/'manifest.json').read_text())['complete'])
    def test_failed_pack_not_complete(self):
        with mock.patch.object(a,'ASSETS',[self.asset]),mock.patch.object(a.urllib.request,'urlopen',side_effect=urllib.error.URLError('offline')):
            with self.assertRaises(urllib.error.URLError):a.build_pack(self.root/'out',self.root/'input')
        self.assertFalse((self.root/'out/manifest.json').exists())

class DatasetChecks(unittest.TestCase):
    setUp=AssetChecks.setUp
    tearDown=AssetChecks.tearDown
    def ready(self):
        root=self.root/'pack'; model=root/'models/fixture.bin'
        model.parent.mkdir(parents=True);model.write_bytes(self.data)
        (root/'manifest.json').write_text(json.dumps({'complete':True,'assets':[self.asset]}))
        return root
    def test_automatic_private_upload_contains_real_flat_files(self):
        root=self.ready()
        class API:
            config_values={'username':'test-user'}
            calls=[]
            def dataset_create_new(api,folder,**kwargs):
                api.calls.append(kwargs)
                self.assertEqual((Path(folder)/'fixture.bin').read_bytes(),self.data)
                self.assertFalse((Path(folder)/'fixture.bin').is_symlink())
                self.assertEqual((Path(folder)/'fixture.bin').stat().st_ino,(root/'models/fixture.bin').stat().st_ino)
                return {'error':None}
        api=API()
        with mock.patch.object(a,'ASSETS',[self.asset]):
            url=a.save_private_dataset(root,api=api)
            self.assertEqual(a.save_private_dataset(root,api=api),url)
        self.assertEqual(len(api.calls),1)
        self.assertIs(api.calls[0]['public'],False)
        self.assertEqual(api.calls[0]['dir_mode'],'skip')
        self.assertTrue((root/'dataset-upload.json').exists())
    def test_upload_failure_keeps_models_without_success_receipt(self):
        root=self.ready()
        class API:
            config_values={'username':'test-user'}
            def dataset_create_new(self,*args,**kwargs):return {'error':'failed'}
        with mock.patch.object(a,'ASSETS',[self.asset]):
            with self.assertRaisesRegex(RuntimeError,'not confirmed'):a.save_private_dataset(root,api=API())
        self.assertEqual((root/'models/fixture.bin').read_bytes(),self.data)
        self.assertFalse((root/'dataset-upload.json').exists())
    def test_corrupt_model_is_not_uploaded(self):
        root=self.ready();(root/'models/fixture.bin').write_bytes(b'bad')
        api=mock.Mock()
        with mock.patch.object(a,'ASSETS',[self.asset]):
            with self.assertRaisesRegex(RuntimeError,'verification'):a.save_private_dataset(root,owner='test-user',api=api)
        api.dataset_create_new.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
