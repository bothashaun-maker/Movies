import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from h3_notebook_helpers import notebook_stages

ROOT=Path(__file__).resolve().parents[1]

class BrowserWorkflow(unittest.TestCase):
    def test_installed_canvas_and_isolated_worker_are_embedded(self):
        nb=json.loads((ROOT/'minimax_h3_kaggle_complete.ipynb').read_text())
        tree=ast.parse(notebook_stages(nb)['install'])
        first=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='browser_node_source' for t in n.targets))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);comfy=root/'ComfyUI'
            scope=dict(ROOT=root,COMFY=comfy,PY=Path(sys.executable),PORT=9190,json=json,
                       MODEL_FAMILY='fl2va',USE_BRIDGE=True,BRIDGE_ALPHA=.1,
                       PROMPT='Fixture prompt',SEED=7,MEGAPIXELS=.4,DURATION=5)
            exec(compile(ast.Module(body=tree.body[first:],type_ignores=[]),'install_browser','exec'),scope)
            canvas=json.loads((comfy/'user/default/workflows/MiniMax H3 - Kaggle T4.json').read_text())
            self.assertEqual(len(canvas['nodes']),17)
            self.assertEqual(len(canvas['links']),19)
            self.assertNotIn('H3KaggleRender',[n['type'] for n in canvas['nodes']])
            self.assertEqual(next(n for n in canvas['nodes'] if n['id']==7)['widgets_values'][0],'Fixture prompt')
            worker=(root/'h3_browser_worker.py').read_text()
            compile(worker,'shipped_worker','exec')
            # Execute definitions/config only: no GPU worker submission in this audit.
            wtree=ast.parse(worker)
            stop=next(n for n in wtree.body if isinstance(n,ast.FunctionDef) and n.name=='stop_owned_servers')
            self.assertTrue(all(isinstance(n,ast.Pass) for n in stop.body))
            self.assertIn('PORT=9190;',worker)
            self.assertIn('PORT+=1',worker)
            self.assertIn("MODEL_FAMILY='fl2va'",worker)
            self.assertIn(repr(str(root)),worker)
            self.assertTrue((comfy/'custom_nodes/H3KaggleBrowser/__init__.py').is_file())

    def test_edited_browser_graph_is_split_without_hidden_presets(self):
        scope={};exec((ROOT/'notebooks/h3_complete/browser_job.py').read_text(),scope)
        graph=json.loads((ROOT/'tests/fixtures/h3_browser_prompt.json').read_text())
        graph['7']['inputs']['prompt']='Browser edit is authoritative'
        graph['8']['inputs']['noise_seed']=12345
        sample,decode=scope['partition_browser_graph'](graph,'16','fixture')
        self.assertEqual(sample['7']['inputs']['prompt'],'Browser edit is authoritative')
        self.assertEqual(sample['8']['inputs']['noise_seed'],12345)
        self.assertFalse({'VAEDecode','VAEDecodeAudio'} & {n['class_type'] for n in sample.values()})
        self.assertFalse({'UNETLoaderH3TensorParallel','CLIPLoaderMultiGPU','SamplerCustomAdvanced'} & {n['class_type'] for n in decode.values()})
        self.assertEqual(decode['16']['class_type'],'SaveVideo')
        self.assertNotIn('18',sample)  # Disconnected optional image is not executed.
        graph['7']['inputs']['first_frame']=['18',0]
        sample,_=scope['partition_browser_graph'](graph,'16','image')
        self.assertIn('18',sample)
        graph['14']['inputs']['samples']=['12',1]
        with self.assertRaisesRegex(ValueError,'same joint sampler'):
            scope['partition_browser_graph'](graph,'16','bad')

    @unittest.skipUnless(os.environ.get('COMFY_ROOT'),'Requires pinned ComfyUI runtime')
    def test_lazy_output_does_not_execute_model_nodes_in_ui_server(self):
        from unittest.mock import patch
        import types,asyncio
        sys.path.insert(0,os.environ['COMFY_ROOT'])
        import nodes,execution
        spec=importlib.util.spec_from_file_location('h3_browser_audit',ROOT/'notebooks/h3_complete/browser_node.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class ModelMustNotRun:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{}}
            RETURN_TYPES=('VIDEO',);FUNCTION='run'
            def run(self):raise AssertionError('GPU graph executed in UI process')
        graph={'1':{'class_type':'H3TestLazyVideo','inputs':{}},'16':{'class_type':'H3IsolatedSaveVideo','inputs':{'video':['1',0],'filename_prefix':'test'}}}
        calls=[]
        def isolated(self,filename_prefix,video=None,prompt=None,unique_id=None):
            calls.append((prompt,unique_id,video));return {'ui':{'images':[]}}
        with patch.dict(nodes.NODE_CLASS_MAPPINGS,{**module.NODE_CLASS_MAPPINGS,'H3TestLazyVideo':ModelMustNotRun}),patch.object(module.H3IsolatedSaveVideo,'render',isolated):
            valid=asyncio.run(execution.validate_prompt('lazy',graph,None))
            self.assertTrue(valid[0],valid)
            server=types.SimpleNamespace(client_id=None,last_node_id=None,send_sync=lambda *a:None)
            runner=execution.PromptExecutor(server,cache_args={'ram':2,'ram_inactive':0})
            runner.execute(graph,'lazy',{},valid[2]);self.assertTrue(runner.success,runner.status_messages)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0][0],graph);self.assertEqual(calls[0][1],'16')
            self.assertIsNone(calls[0][2])

if __name__=='__main__':unittest.main()
