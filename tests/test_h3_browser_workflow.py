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
            self.assertEqual(canvas['nodes'][0]['type'],'H3KaggleRender')
            self.assertEqual(canvas['nodes'][0]['widgets_values'][0],'Fixture prompt')
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

    @unittest.skipUnless(os.environ.get('COMFY_ROOT'),'Requires pinned ComfyUI runtime')
    def test_actual_browser_node_schema_accepts_workflow_inputs(self):
        import asyncio
        sys.path.insert(0,os.environ['COMFY_ROOT'])
        import nodes,execution
        spec=importlib.util.spec_from_file_location('h3_browser_audit',ROOT/'notebooks/h3_complete/browser_node.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        canvas=json.loads((ROOT/'workflows/minimax_h3_kaggle_t4.json').read_text())
        values=canvas['nodes'][0]['widgets_values']
        graph={'1':{'class_type':'H3KaggleRender','inputs':dict(prompt=values[0],seed=values[1],megapixels=values[3],duration=values[4],aspect=values[5])}}
        valid=asyncio.run(execution.validate_prompt('browser-schema',graph,None))
        self.assertTrue(valid[0],valid);self.assertEqual(valid[2],['1'])

if __name__=='__main__':unittest.main()
