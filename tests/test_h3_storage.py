"""Storage regressions using real links/files and measured-volume fixtures."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
GIB=2**30


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        base=self.base
        devices={'scratch':22,'working':11,'tmp':11,'inputs':33}
        self.devices=devices
        class VolumePath(type(Path())):
            def stat(path,**kwargs):
                result=super().stat(**kwargs)
                values=list(result)
                for name,device in devices.items():
                    if path.is_relative_to(base/name):values[2]=device;break
                return os.stat_result(values)
        def paths(value):
            mapping={'/kaggle/input':base/'inputs',
                     '/kaggle/temp/minimax_h3_models':base/'scratch',
                     '/tmp/minimax_h3_models':base/'tmp'}
            return VolumePath(mapping.get(str(value),value))
        root=paths(base/'working');root.mkdir()
        self.free={'working':int(11.57*GIB),'scratch':50*GIB,'tmp':int(11.57*GIB)}
        def disk_usage(path):
            part=next(name for name in self.free if Path(path).is_relative_to(base/name))
            return SimpleNamespace(free=self.free[part])
        self.disk_patch=patch.object(shutil,'disk_usage',side_effect=disk_usage)
        self.disk_patch.start();self.addCleanup(self.disk_patch.stop)
        self.nb=json.loads((REPO/'minimax_h3_kaggle_dual_t4.ipynb').read_text())
        tree=ast.parse(''.join(self.nb['cells'][2]['source']))
        funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
        self.scope=dict(ROOT=root,COMFY=root/'ComfyUI',Path=paths,os=os,shutil=shutil,
                        hashlib=hashlib,json=json,time=time,MODEL_FAMILY='ref2va',USE_BRIDGE=False,
                        MODEL_DATASET='',MODEL_STORAGE='auto',ALLOW_MODEL_DOWNLOADS=True,MODELS_READY=False,
                        child_env=os.environ.copy())
        exec(compile(funcs,'model_storage_functions','exec'),self.scope)

    def asset(self,name,size):
        return dict(repo='Comfy-Org/MiniMax-H3',revision='pinned',file=name,size=size,sha256='unused')

    def test_separate_50gib_scratch_holds_41gib_payload(self):
        self.scope['ASSETS']=[self.asset('diffusion_models/model.safetensors',int(41.38*GIB))]
        plan=self.scope['model_storage_plan'](package_reserve=6*GIB)
        self.assertEqual(plan['store'],self.base/'scratch')
        self.assertEqual(plan['device'],22)
        self.assertTrue(plan['fits'])
        self.assertTrue(plan['locations'][0][1].is_relative_to(self.base/'scratch'))

    def test_shared_small_directories_are_not_summed(self):
        self.devices['scratch']=11
        self.free.update(working=20*GIB,scratch=20*GIB,tmp=20*GIB)
        self.scope['ASSETS']=[self.asset('diffusion_models/model.safetensors',int(41.38*GIB))]
        with self.assertRaisesRegex(RuntimeError,'No measured storage'):self.scope['model_storage_plan']()
        self.assertTrue((self.base/'working/storage-report.json').is_file())

    def test_nested_attached_inputs_are_linked_and_verified(self):
        assets=[];sources=[]
        for index,name in enumerate(('diffusion_models/a.safetensors','text_encoders/b.safetensors')):
            data=(b'model '+str(index).encode())*3
            a=self.asset(name,len(data));a['sha256']=hashlib.sha256(data).hexdigest();assets.append(a)
            source=self.base/'inputs/datasets/owner/title/versions/1/models'/name
            source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(data);sources.append(source)
        self.scope['ASSETS']=assets
        exec(''.join(self.nb['cells'][8]['source']),self.scope)
        self.assertTrue(self.scope['MODELS_READY'])
        for a,source in zip(assets,sources):
            target=self.scope['asset_target'](a)
            self.assertTrue(target.is_symlink());self.assertEqual(target.resolve(),source)
        self.scope['require_models_ready']()
        sources[0].unlink()
        with self.assertRaisesRegex(RuntimeError,'missing or changed'):self.scope['require_models_ready']()

    def test_download_staging_and_destination_stay_on_scratch(self):
        data=b'checkpoint bytes';a=self.asset('diffusion_models/a.safetensors',len(data))
        a['sha256']=hashlib.sha256(data).hexdigest();self.scope.update(ASSETS=[a],PY='python')
        def run(args,**kwargs):
            stage=args[-1];staged=stage/a['file'];staged.parent.mkdir(parents=True)
            self.assertTrue(stage.is_relative_to(self.base/'scratch'))
            staged.write_bytes(data)
        self.scope['run']=run
        exec(''.join(self.nb['cells'][8]['source']),self.scope)
        target=self.scope['asset_target'](a)
        self.assertTrue(target.is_symlink())
        self.assertTrue(target.resolve().is_relative_to(self.base/'scratch'))
        self.assertEqual(target.read_bytes(),data)
        self.assertEqual(list((self.base/'scratch/.staging').iterdir()),[])

    def test_later_cells_stop_before_server_or_queue_on_setup_failure(self):
        self.scope.update(ASSETS=[],RUN_GENERATION=True)
        for index in (10,12,14):
            with self.subTest(cell=index),self.assertRaisesRegex(RuntimeError,'Model setup has not completed'):
                exec(''.join(self.nb['cells'][index]['source']),self.scope)

    def test_all_notebooks_compile_and_preparation_shards_fit_20gib(self):
        for path in [REPO/'minimax_h3_kaggle_dual_t4.ipynb',*(REPO/'notebooks').glob('minimax_h3_prepare_*.ipynb')]:
            n=json.loads(path.read_text())
            for i,c in enumerate(n['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'{path.name}:{i}','exec')
            if 'prepare_' in path.name:
                constants=''.join(n['cells'][1]['source']).split('OUTPUT.mkdir',1)[0]
                scope={};exec(constants,scope)
                self.assertLess(sum(a['size'] for a in scope['selected']),20*GIB)


if __name__=='__main__':unittest.main()
