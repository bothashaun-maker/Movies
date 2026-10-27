import copy,json,os,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'notebooks/h3_suite'

class Collections(unittest.TestCase):
    def scope(self):
        scope={'Path':Path,'json':json}
        for p in (ROOT/'notebooks/h3_complete/browser_job.py',S/'job.py'):exec(p.read_text(),scope)
        return scope

    def test_all_seven_real_browser_graphs_preserve_worker_boundary(self):
        scope=self.scope()
        files=list((ROOT/'tests/fixtures/h3_collections').glob('*.json'))
        self.assertEqual(len(files),7)
        for file in files:
            with self.subTest(workflow=file.name):
                graph=json.loads(file.read_text());output=next(k for k,n in graph.items() if n['class_type']=='H3SuiteSaveVideo')
                sample,decode=scope['suite_partition'](graph,output,'test')
                cp=sample['h3_suite_checkpoint']['inputs']
                external='Audio' in file.stem
                self.assertEqual('external_audio' in cp,external)
                self.assertEqual(any(n['class_type']=='H3SuiteLoadAudio' for n in decode.values()),external)
                self.assertFalse(any(n['class_type'] in ('KSampler','SamplerCustomAdvanced','MiniMaxH3DirectorCS','H3SuiteUNETLoader','H3SuiteCLIPLoader','PixaromaH3AudioSync') for n in decode.values()))
                self.assertEqual(len([n for n in sample.values() if n['class_type']=='H3SuiteUNETLoader']),1)

    def test_director_reference_switch_selects_only_correct_family(self):
        scope=self.scope()
        graph=json.loads((ROOT/'tests/fixtures/h3_collections/Director - Three Step.json').read_text())
        node=next(n for n in graph.values() if n['class_type']=='MiniMaxH3DirectorCS')
        state=json.loads(node['inputs']['timeline_data']);state['reference_mode']='ON';node['inputs']['timeline_data']=json.dumps(state)
        sample,_=scope['suite_partition'](graph,'403','test')
        loaders=[n for n in sample.values() if n['class_type']=='H3SuiteUNETLoader']
        self.assertEqual(len(loaders),1);self.assertIn('ref2va',loaders[0]['inputs']['unet_name'])

    def test_notebooks_have_distinct_collections_and_one_run_cell(self):
        import nbformat
        for name,count in [('Current_Working_Workflow',None),('Three_Step_Workflow',None),('Pixa_5_Workflows',5),('Director',2)]:
            nb=nbformat.read(ROOT/(name+'.ipynb'),as_version=4);nbformat.validate(nb)
            codes=[c.source for c in nb.cells if c.cell_type=='code'];self.assertEqual(len(codes),1);compile(codes[0],name,'exec')
            if count is not None:self.assertEqual(len(list((ROOT/'workflows'/name).glob('*.json'))),count)

    @unittest.skipUnless(os.environ.get('COMFY_ROOT'),'Requires pinned CPU runtime')
    def test_external_audio_roundtrips_without_resampling(self):
        from unittest.mock import patch
        from test_h3_complete_runtime import torch,folder_paths,NestedTensor
        scope={'__name__':'suite_audio_fixture'}
        exec((ROOT/'notebooks/h3_complete/checkpoint_nodes.py').read_text(),scope)
        scope['CHECKPOINT_NODE_MAPPINGS']=scope['NODE_CLASS_MAPPINGS'].copy()
        # Supply only the base class; sidecar methods do not need the UI process.
        scope['H3IsolatedSaveVideo']=type('H3IsolatedSaveVideo',(),{})
        exec(S.joinpath('nodes.py').read_text(),scope)
        with tempfile.TemporaryDirectory() as tmp,patch.object(folder_paths,'get_output_directory',return_value=tmp):
            latent={'samples':NestedTensor([torch.zeros(1,24,7,2,2),torch.zeros(1,32,2,50)])}
            original={'waveform':torch.randn(1,2,32000),'sample_rate':32000}
            scope['H3SuiteCheckpoint']().save(latent,'audio',original)
            restored=scope['H3SuiteLoadAudio']().load('audio')[0]
            self.assertEqual(restored['sample_rate'],original['sample_rate'])
            self.assertTrue(torch.equal(restored['waveform'],original['waveform']))

    @unittest.skipUnless(os.environ.get('COMFY_ROOT'),'Requires pinned CPU runtime')
    def test_suite_save_keeps_image_and_audio_dependencies_out_of_ui_process(self):
        import asyncio,types
        from unittest.mock import patch
        from test_h3_complete_runtime import nodes,execution
        scope={'__name__':'suite_lazy_fixture'}
        exec((ROOT/'notebooks/h3_complete/checkpoint_nodes.py').read_text(),scope)
        scope['CHECKPOINT_NODE_MAPPINGS']=scope['NODE_CLASS_MAPPINGS'].copy()
        exec((ROOT/'notebooks/h3_complete/browser_node.py').read_text(),scope)
        exec(S.joinpath('nodes.py').read_text(),scope)
        cls=scope['H3SuiteSaveVideo']
        class MustNotRun:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{}}
            RETURN_TYPES=('IMAGE','AUDIO');FUNCTION='run'
            def run(self):raise AssertionError('GPU dependency executed in browser server')
        graph={'1':{'class_type':'SuiteBomb','inputs':{}},'2':{'class_type':'H3SuiteSaveVideo','inputs':{'images':['1',0],'audio':['1',1],'filename_prefix':'fixture','fps':24}}}
        calls=[]
        def render(self,**kw):calls.append(kw);return {'ui':{'images':[]}}
        with patch.dict(nodes.NODE_CLASS_MAPPINGS,{'SuiteBomb':MustNotRun,'H3SuiteSaveVideo':cls}),patch.object(cls,'render',render):
            valid=asyncio.run(execution.validate_prompt('suite-lazy',graph,None));self.assertTrue(valid[0],valid)
            server=types.SimpleNamespace(client_id=None,last_node_id=None,send_sync=lambda *a:None)
            runner=execution.PromptExecutor(server,cache_args={'ram':2,'ram_inactive':0});runner.execute(graph,'suite-lazy',{},valid[2])
            self.assertTrue(runner.success,runner.status_messages)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0]['prompt'],graph)

    def test_cache_reclaims_only_exact_tracked_download_paths(self):
        from test_h3_complete import scope_at
        from unittest.mock import Mock
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scope=scope_at(root);exec(S.joinpath('assets.py').read_text(),scope)
            store=root/'cache';scope['MODEL_STORAGE']=str(store)
            asset={'repo':'fixture/model','revision':'pin','file':'diffusion_models/old.safetensors','size':3,'sha256':hashlib.sha256(b'old').hexdigest()}
            scope['SUITE_ASSETS']=[asset]
            owned=scope['stored_asset'](store,asset);owned.parent.mkdir(parents=True);owned.write_bytes(b'old')
            user=root/'user-upload.safetensors';user.write_bytes(b'old')
            ledger={asset['sha256']:{'path':str(user),'store':str(store),'sha256':asset['sha256'],'size':3,'mtime_ns':user.stat().st_mtime_ns}}
            path=root/'suite-download-cache.json';path.write_text(json.dumps(ledger))
            scope['model_storage_plan']=Mock(side_effect=[RuntimeError('space'),{'locations':[],'store':store}])
            scope['suite_prepare_assets']([])
            self.assertTrue(user.exists());self.assertTrue(owned.exists())
            ledger[asset['sha256']].update(path=str(owned),mtime_ns=owned.stat().st_mtime_ns);path.write_text(json.dumps(ledger))
            scope['model_storage_plan']=Mock(side_effect=[RuntimeError('space'),{'locations':[],'store':store}])
            scope['suite_prepare_assets']([])
            self.assertFalse(owned.exists());self.assertTrue(user.exists())
