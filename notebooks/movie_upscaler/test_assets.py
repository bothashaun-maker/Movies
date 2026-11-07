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

if __name__=='__main__':unittest.main(verbosity=2)
