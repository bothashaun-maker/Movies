"""Regression checks for the separate experimental H3 Next notebook."""
import ast,copy,hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
import nbformat
from h3_notebook_helpers import notebook_stages
R=Path(__file__).resolve().parents[1];N=R/'notebooks/h3_next'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def assignments(source):
    result={}
    for s in ast.parse(source).body:
        if isinstance(s,ast.Assign):
            for target in s.targets:
                if isinstance(target,ast.Name):
                    try:result[target.id]=ast.literal_eval(s.value)
                    except (ValueError,TypeError):pass
    return result

class NextNotebook(unittest.TestCase):
    def test_notebooks_compile_and_pack_coverage(self):
        nb=nbformat.read(R/'H3_Next_T4.ipynb',4);nbformat.validate(nb)
        self.assertEqual(sum(c.cell_type=='code' for c in nb.cells),1)
        stages=notebook_stages(nb)
        for name,s in stages.items():compile(s,name,'exec')
        embedded=assignments(stages['install'])
        for key in ['plugin_source','worker_source']:compile(embedded[key],key,'exec')
        packs_nb=nbformat.read(R/'H3_Next_Model_Packs.ipynb',4);nbformat.validate(packs_nb)
        source=next(c.source for c in packs_nb.cells if c.cell_type=='code');compile(source,'packs','exec')
        packs=assignments(source)['PACKS'];self.assertEqual(set(packs),{'singularity','denomix','molmo','support'})
        prepared={a['sha256'] for group in packs.values() for a in group}
        catalog=assignments(stages['catalog'])['ASSETS']
        self.assertEqual(prepared,{a['sha256'] for a in catalog})
        for pack in packs.values():self.assertLess(sum(a['size'] for a in pack),20*1024**3)
        self.assertIn("Path('/kaggle/working/h3_next_t4')",stages['settings'])

    def test_install_reader_patch_produces_valid_python(self):
        stages=notebook_stages(nbformat.read(R/'H3_Next_T4.ipynb',4))
        tree=ast.parse(stages['install'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tp_source.py';p.write_text('from collections import OrderedDict\ndef open_weights(source):\n    return safe_open(source, framework="pt", device="cpu")\n')
            body=[];active=False
            for s in tree.body:
                if isinstance(s,ast.Assign) and isinstance(s.value,ast.BinOp) and ast.unparse(s.value)=="MGPU / 'tp_source.py'":active=True
                if active and isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='pinned_checkout':break
                if active:body.append(s)
            self.assertTrue(body)
            exec(compile(ast.Module(body=body,type_ignores=[]),'patch','exec'),{'MGPU':Path(tmp)})
            compile(p.read_text(),'patched_reader','exec')
            self.assertIn('from .h3_next_weights import H3NextWeights',p.read_text())

    def test_molmo_patch_is_repeatable_and_preserves_int8(self):
        mod=module(N/'patch_molmo.py','next_patch')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'quantization/convrot_runtime.py';p.parent.mkdir()
            p.write_text('import torch\nfrom torch import nn\ndef load_model(device):\n    weight=torch.empty(1,dtype=torch.bfloat16)\n    quant=torch.empty(1,dtype=torch.int8)\n    return model.eval().to(device)\n')
            mod.patch_molmo(root);once=p.read_text();mod.patch_molmo(root)
            self.assertEqual(once,p.read_text());compile(once,'molmo','exec')
            self.assertIn('dtype=torch.int8',once);self.assertIn('value.dtype == torch.bfloat16',once)
            self.assertIn("model.config._attn_implementation = 'sdpa'",once)

    def test_real_prefixed_weight_reader(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:self.skipTest('torch and safetensors required')
        mod=module(N/'weight_reader.py','next_weights')
        with tempfile.TemporaryDirectory() as tmp:
            for prefix in ['', 'model.diffusion_model.', 'diffusion_model.']:
                p=Path(tmp)/'model.safetensors'
                tensors={prefix+'blocks.0.attn.qkv_proj.weight':torch.arange(8,dtype=torch.int8).reshape(4,2),prefix+'blocks.0.attn.qkv_proj.weight_scale':torch.ones(4,1),prefix+'blocks.0.attn.qkv_proj.comfy_quant':torch.tensor(list(b'{"format":"int8_tensorwise","convrot":true}'),dtype=torch.uint8)}
                save_file(tensors,str(p));before=hashlib.sha256(p.read_bytes()).hexdigest()
                with mod.H3NextWeights(str(p)) as f:
                    self.assertIn('blocks.0.attn.qkv_proj.weight',f.keys())
                    self.assertEqual(f.get_slice('blocks.0.attn.qkv_proj.weight').get_shape(),[4,2])
                    self.assertEqual(f.get_tensor('blocks.0.attn.qkv_proj.weight_scale').shape,(4,))
                    self.assertTrue(torch.equal(f.get_tensor('blocks.0.attn.qkv_proj.weight'),tensors[prefix+'blocks.0.attn.qkv_proj.weight']))
                self.assertEqual(before,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_reference_preflight_and_optional_assets(self):
        stages=notebook_stages(nbformat.read(R/'H3_Next_T4.ipynb',4));assets=assignments(stages['catalog'])['ASSETS']
        scope={'Path':Path,'json':json,'SUITE_ASSETS':assets,'suite_required_assets':lambda graph,require_diffusion=True:[], 'suite_preflight':lambda graph:None}
        exec((N/'job.py').read_text(),scope)
        graph={'1':{'class_type':'H3NextReference','inputs':{'width':512,'height':768,'seconds':5.,'prompt':'A woman walks.'}},'2':{'class_type':'H3NextMolmo','inputs':{}},'3':{'class_type':'H3NextMotion','inputs':{'strength':0.}}}
        scope['suite_preflight'](graph)
        selected=scope['suite_required_assets'](graph);self.assertEqual({a['pack'] for a in selected},{'molmo'})
        graph['3']['inputs']['strength']=.5;self.assertTrue(any(a['file']=='dials/h3_motion.safetensors' for a in scope['suite_required_assets'](graph)))
        for key,value in [('width',1344),('seconds',10),('prompt','x'*5001)]:
            bad=copy.deepcopy(graph);bad['1']['inputs'][key]=value
            with self.assertRaises(ValueError):scope['suite_preflight'](bad)

    def test_three_canvases_have_real_nodes_and_no_extra_pass(self):
        for p in (R/'workflows/H3_Next').glob('*.json'):
            graph=json.loads(p.read_text());ns={n['id']:n for n in graph['nodes']}
            self.assertGreater(len(ns),15)
            kinds=[n['type'] for n in ns.values()]
            self.assertEqual(kinds.count('SamplerCustomAdvanced'),1)
            self.assertEqual(kinds.count('H3SuiteSaveVideo'),1)
            self.assertNotIn('BlockSparseAttention',kinds)
            self.assertIn('H3NextMolmo',kinds)
            self.assertEqual(ns[20]['widgets_values'],[0.])
            for lid,src,slot,dst,index,typ in graph['links']:
                self.assertIn(lid,ns[src]['outputs'][slot]['links']);self.assertEqual(ns[dst]['inputs'][index]['link'],lid)

    def test_three_process_partition_preserves_reference(self):
        scope={'Path':Path,'json':json}
        for source in ['h3_complete/browser_job.py','h3_suite/preflight.py','h3_suite/job.py']:
            exec((R/'notebooks'/source).read_text(),scope)
        scope['suite_required_assets']=lambda *a,**k:[]
        exec((N/'job.py').read_text(),scope)
        for path in (R/'tests/fixtures/h3_next').glob('*.json'):
            for reference in [False,True]:
                graph=json.loads(path.read_text())
                if reference:graph['7']['inputs']['reference_image']=['18',0]
                else:graph.pop('18',None)
                output=next(k for k,n in graph.items() if n['class_type']=='H3SuiteSaveVideo')
                prepare,sample,decode=scope['next_partition'](graph,output,'fixture')
                kinds=lambda g:{n['class_type'] for n in g.values()}
                self.assertIn('H3NextMolmo',kinds(prepare))
                self.assertEqual('LoadImage' in kinds(prepare),reference)
                self.assertNotIn('H3SuiteUNETLoader',kinds(prepare))
                self.assertIn('H3NextLoadPreparation',kinds(sample))
                self.assertNotIn('H3NextMolmo',kinds(sample))
                self.assertNotIn('H3SuiteVAELoader',kinds(sample))
                self.assertNotIn('H3SuiteUNETLoader',kinds(decode))
