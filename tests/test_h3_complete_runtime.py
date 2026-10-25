"""Actual pinned ComfyUI validation and checkpoint executor tests, CPU only."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from h3_notebook_helpers import notebook_stages
from unittest.mock import patch

if not os.environ.get('COMFY_ROOT') or not os.environ.get('H3_MGPU_ROOT'):
    raise unittest.SkipTest('Set COMFY_ROOT and H3_MGPU_ROOT for pinned runtime checks.')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(os.environ['COMFY_ROOT'])))
saved_argv=list(sys.argv);sys.argv=[sys.argv[0],'--cpu','--disable-metadata']
import comfy.options
comfy.options.enable_args_parsing()
import nodes,execution,folder_paths,torch
from comfy.nested_tensor import NestedTensor
sys.argv=saved_argv


def load(path,name,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


class CompleteRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False,init_api_nodes=False))
        cls.cp=load(ROOT/'notebooks/h3_complete/checkpoint_nodes.py','h3_complete_checkpoint')
        cls.mgpu=load(Path(os.environ['H3_MGPU_ROOT'])/'__init__.py','h3_complete_mgpu',True)
        # Device enum only; no model loading or GPU operations run in this CPU test.
        sys.modules['h3_complete_mgpu.wrappers'].get_device_list=lambda:['cpu','cuda:0','cuda:1']
        nodes.NODE_CLASS_MAPPINGS.update(cls.cp.NODE_CLASS_MAPPINGS)
        nodes.NODE_CLASS_MAPPINGS.update(cls.mgpu.NODE_CLASS_MAPPINGS)

    def test_notebook_actual_graph_validation(self):
        import nbformat
        nb=nbformat.read(ROOT/'minimax_h3_kaggle_complete.ipynb',as_version=4)
        nbformat.validate(nb)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);scope={}
            for cell in nb.cells:
                if cell.cell_type=='code':compile(cell.source,cell.id,'exec')
            shipped=notebook_stages(nb)
            for stage in ('settings','catalog','workflow'):
                source=shipped[stage].replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(temp)!r})')
                exec(source,scope)
            for asset in scope['ASSETS']:
                if asset['file'].endswith('.zip'):continue
                relative=Path(asset.get('dest') or asset['file'])
                folder=temp/'models'/relative.parts[0];folder.mkdir(parents=True,exist_ok=True)
                (folder/relative.name).touch()  # filenames for schema only; never opened as weights
                folder_paths.add_model_folder_path(relative.parts[0],str(folder))
            (scope['COMFY']/'input').mkdir(parents=True)
            folder_paths.set_input_directory(str(scope['COMFY']/'input'))
            folder_paths.set_output_directory(str(scope['COMFY']/'output'))
            from PIL import Image
            image=temp/'reference.png';Image.new('RGB',(80,120)).save(image)
            for family in ('ref2va','fl2va'):
                for mode in ('text','image'):
                    scope.update(MODEL_FAMILY=family,MODE=mode,IMAGE_PATH=str(image))
                    key,directory,sample,decode=scope['prepare_job']()
                    catalog={}
                    for node in [*sample.values(),*decode.values()]:
                        kind=node['class_type'];cls=nodes.NODE_CLASS_MAPPINGS[kind]
                        catalog[kind]=cls.GET_NODE_INFO_V1() if hasattr(cls,'GET_NODE_INFO_V1') else {'input':cls.INPUT_TYPES()}
                    scope['check_node_catalog'](catalog,[sample,decode])
                    valid=asyncio.run(execution.validate_prompt('sample',sample,None))
                    self.assertTrue(valid[0],valid)
                    self.assertEqual(valid[2],['19'])
                    self.cp.H3CheckpointBeforeDecode().save({'samples':NestedTensor([
                        torch.zeros(1,24,37,48,32),torch.zeros(1,32,2,200)])},key)
                    valid=asyncio.run(execution.validate_prompt('decode',decode,None))
                    self.assertTrue(valid[0],valid)
                    self.assertEqual(valid[2],['16'])
                    print('Validated separate sampling/decode graphs:',family,mode)
            # Optional bridge schema checked against its pinned, verified source.
            bridge=os.environ.get('H3_BRIDGE_ROOT')
            if bridge:
                module=load(Path(bridge)/'__init__.py','h3_complete_bridge',True)
                nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                scope.update(USE_BRIDGE=True,MODEL_FAMILY='fl2va')
                key,directory,sample,decode=scope['prepare_job']()
                valid=asyncio.run(execution.validate_prompt('bridge',sample,None))
                self.assertTrue(valid[0],valid)
                print('Validated supported FL2VA bridge graph')
            sample['9']['inputs']['model']=['missing',0]
            self.assertFalse(asyncio.run(execution.validate_prompt('negative',sample,None))[0])

    def test_actual_export_is_found_and_verified_by_controller(self):
        import subprocess
        from comfy_extras.nodes_video import CreateVideo,SaveVideo
        class Fixture:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{}}
            RETURN_TYPES=('IMAGE','AUDIO');FUNCTION='make'
            def make(self):
                return (torch.rand(22,32,32,3),{'waveform':torch.zeros(1,2,32000),'sample_rate':32000})
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);scope={}
            for name in ('settings','graphs','runtime'):
                source=(ROOT/'notebooks/h3_complete'/(name+'.py')).read_text()
                source=source.replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(temp)!r})')
                exec(source,scope)
            output=scope['COMFY']/'output';output.mkdir(parents=True)
            scope.update(PY=Path(sys.executable),DURATION=22/24,
                         canvas_size=lambda *args:(32,32),run=lambda args:subprocess.run([str(x) for x in args],check=True))
            with patch.object(folder_paths,'get_output_directory',return_value=str(output)),patch.dict(nodes.NODE_CLASS_MAPPINGS,{'H3TestMedia':Fixture}):
                graph={'1':{'class_type':'H3TestMedia','inputs':{}},
                       '15':{'class_type':'CreateVideo','inputs':{'images':['1',0],'audio':['1',1],'fps':24.0,'bit_depth':8}},
                       '16':{'class_type':'SaveVideo','inputs':{'video':['15',0],'filename_prefix':'fixture/video','format':'mp4','format.codec':'h264'}}}
                server=types.SimpleNamespace(client_id=None,last_node_id=None,send_sync=lambda *a:None)
                runner=execution.PromptExecutor(server,cache_args={'ram':2,'ram_inactive':0})
                runner.execute(graph,'export',{},['16'])
                self.assertTrue(runner.success,runner.status_messages)
                paths=scope['video_paths'](runner.history_result)
                self.assertEqual(len(paths),1,runner.history_result)
                report=scope['verify_video'](paths[0],temp/'report.json')
                self.assertEqual(report['frames'],22);self.assertGreater(report['audio_samples'],0)

    def test_checkpoint_is_executed_as_terminal_output(self):
        class Fixture:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{}}
            RETURN_TYPES=('LATENT',);FUNCTION='make'
            def make(self):
                return ({'samples':NestedTensor([torch.ones(1,24,2,2,2),torch.ones(1,32,2,10)])},)
        with tempfile.TemporaryDirectory() as temp,patch.object(folder_paths,'get_output_directory',return_value=temp),patch.dict(nodes.NODE_CLASS_MAPPINGS,{'H3TestLatents':Fixture}):
            graph={'1':{'class_type':'H3TestLatents','inputs':{}},
                   '19':{'class_type':'H3CheckpointBeforeDecode','inputs':{'samples':['1',0],'checkpoint_id':'boundary'}}}
            valid=asyncio.run(execution.validate_prompt('boundary',graph,None))
            self.assertTrue(valid[0],valid);self.assertEqual(valid[2],['19'])
            server=types.SimpleNamespace(client_id=None,last_node_id=None,send_sync=lambda *a:None)
            runner=execution.PromptExecutor(server,cache_args={'ram':2,'ram_inactive':0})
            runner.execute(graph,'boundary',{},valid[2])
            self.assertTrue(runner.success,runner.status_messages)
            restored=self.cp.H3LoadDecodeCheckpoint().load('boundary')[0]['samples'].unbind()
            self.assertEqual(len(restored),2)
            self.assertTrue(torch.equal(restored[0],torch.ones(1,24,2,2,2)))
            with self.assertRaises(ValueError):self.cp.checkpoint_path('../escape')
            restored[0][0,0,0,0,0]=float('nan')
            with self.assertRaisesRegex(ValueError,'non-finite'):
                self.cp.pack_latents({'samples':NestedTensor(list(restored))})


if __name__=='__main__':unittest.main()
