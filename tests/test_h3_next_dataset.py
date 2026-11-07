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

class PackStorage(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.roots=[self.base/'working/models',self.base/'temp/models',self.base/'other/models']
        self.asset={'file':'large.bin','size':20967647456,'sha256':'unused'}
    def tearDown(self):self.tmp.cleanup()
    def usage(self,values):
        from types import SimpleNamespace
        return lambda p:SimpleNamespace(free=values[self.roots.index(Path(p))])
    def test_reported_1950_gib_failure_selects_roomy_temp(self):
        from unittest.mock import patch
        with patch('shutil.disk_usage',side_effect=self.usage([int(19.5*1024**3),50*1024**3,8*1024**3])):
            out=m.select_pack_storage('singularity',[self.asset],candidates=self.roots)
        self.assertEqual(out,self.roots[1])
    def test_free_space_is_not_added_across_paths(self):
        from unittest.mock import patch
        with patch('shutil.disk_usage',side_effect=self.usage([int(19.5*1024**3)]*3)):
            with self.assertRaisesRegex(RuntimeError,'No single writable location'):
                m.select_pack_storage('singularity',[self.asset],candidates=self.roots)
    def test_partial_download_counts_only_remaining_bytes(self):
        from unittest.mock import patch
        self.roots[0].mkdir(parents=True)
        (self.roots[0]/'large.bin.part').write_bytes(b'x'*80)
        asset=dict(self.asset,size=100)
        with patch('shutil.disk_usage',side_effect=self.usage([30,1000,1000])):
            out=m.select_pack_storage('singularity',[asset],candidates=self.roots,headroom=10)
        self.assertEqual(out,self.roots[0])
        self.assertEqual((out/'large.bin.part').stat().st_size,80)
    def test_other_pack_is_preserved_and_skipped(self):
        from unittest.mock import patch
        self.roots[0].mkdir(parents=True)
        manifest=self.roots[0].parent/'h3-pack-selection.json'
        manifest.write_text(json.dumps({'pack':'molmo'}))
        with patch('shutil.disk_usage',side_effect=self.usage([100*1024**3,50*1024**3,30*1024**3])):
            out=m.select_pack_storage('singularity',[self.asset],candidates=self.roots)
        self.assertEqual(out,self.roots[1]);self.assertEqual(json.loads(manifest.read_text())['pack'],'molmo')
    def test_generated_preflight_and_working_only_fallback(self):
        from unittest.mock import patch
        # Redirect the default working path into our fixture; temp paths must not be tried.
        def redirect(value):
            return self.roots[0] if str(value)=='/kaggle/working/models' else Path(value)
        with patch.object(m,'Path',side_effect=redirect), patch('shutil.disk_usage',side_effect=self.usage([int(19.5*1024**3),50*1024**3,50*1024**3])) as disk:
            with self.assertRaisesRegex(RuntimeError,'Notebook Outputs require working storage'):
                m.select_pack_storage('singularity',[self.asset],automatic=False)
            self.assertEqual(disk.call_count,1)
        nb=json.loads((R/'H3_Next_Model_Packs.ipynb').read_text())
        code=''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
        self.assertIn('OUTPUT=select_pack_storage(PACK,selected,AUTO_CREATE_DATASET)',code)
        self.assertNotIn('assert free>=needed',code)
        self.assertLess(code.index('dataset_client,DATASET_OWNER=dataset_session'),code.index('OUTPUT=select_pack_storage'))
