"""Exercise upload contracts without credentials, network or model downloads."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock

R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h3_next_dataset',R/'notebooks/h3_next/dataset.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NextDataset(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'models';self.root.mkdir()
        self.data=b'verified model fixture';p=self.root/'projections/model.bin';p.parent.mkdir();p.write_bytes(self.data)
        self.assets=[{'file':'projections/model.bin','size':len(self.data),'sha256':hashlib.sha256(self.data).hexdigest()}]
    def tearDown(self):self.tmp.cleanup()
    def test_private_flat_hardlinks_and_receipt_reuse(self):
        api=Mock();api.config_values={'username':'example'}
        def submit(folder,**kwargs):
            path=Path(folder)/'model.bin'
            self.assertEqual(path.read_bytes(),self.data)
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_ino,(self.root/'projections/model.bin').stat().st_ino)
            self.assertFalse(kwargs['public']);self.assertEqual(kwargs['dir_mode'],'skip')
            manifest=json.loads((Path(folder)/'manifest.json').read_text())
            self.assertEqual(manifest['assets'],self.assets)
            return {'error':None}
        api.dataset_create_new.side_effect=submit
        url=m.save_private_dataset(self.root,'molmo',self.assets,api=api)
        self.assertIn('/example/h3-next-molmo-',url)
        self.assertEqual(m.save_private_dataset(self.root,'molmo',self.assets,api=api),url)
        self.assertEqual(api.dataset_create_new.call_count,1)
    def test_upload_failure_has_no_receipt_and_retains_models(self):
        for response in [None,{'error':'failed'}]:
            api=Mock();api.dataset_create_new.return_value=response
            with self.assertRaisesRegex(RuntimeError,'not confirmed'):
                m.save_private_dataset(self.root,'molmo',self.assets,owner='example',api=api)
            self.assertFalse((self.root.parent/'h3-next-molmo-dataset-upload.json').exists())
            self.assertEqual((self.root/'projections/model.bin').read_bytes(),self.data)
    def test_corruption_and_symlinks_block_upload(self):
        p=self.root/'projections/model.bin';api=Mock()
        p.write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError,'verification'):m.save_private_dataset(self.root,'molmo',self.assets,owner='example',api=api)
        p.unlink();other=self.root/'real.bin';other.write_bytes(self.data);p.symlink_to(other)
        with self.assertRaisesRegex(RuntimeError,'verification'):m.save_private_dataset(self.root,'molmo',self.assets,owner='example',api=api)
        api.dataset_create_new.assert_not_called()
    def test_duplicate_names_block_upload(self):
        api=Mock()
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            m.save_private_dataset(self.root,'molmo',self.assets*2,owner='example',api=api)
        api.dataset_create_new.assert_not_called()
    def test_authentication_preflight_and_generated_order(self):
        api=Mock();api.config_values={}
        with self.assertRaisesRegex(RuntimeError,'No model downloads'):m.dataset_session(api=api)
        self.assertEqual(m.dataset_session('example',api),(api,'example'))
        nb=json.loads((R/'H3_Next_Model_Packs.ipynb').read_text())
        code=''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
        self.assertIn('AUTO_CREATE_DATASET = True',code)
        self.assertLess(code.index('dataset_client,DATASET_OWNER=dataset_session'),code.index('for asset in selected:'))
        self.assertIn('save_private_dataset(OUTPUT,PACK,selected,DATASET_OWNER,dataset_client)',code)
