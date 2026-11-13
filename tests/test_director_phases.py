"""Director phase boundaries, bounded storage, and recovery without GPU weights."""
import ast,copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];S=R/'notebooks/h3_suite'

def scope():
    s={'Path':Path,'json':json,'hashlib':hashlib}
    for p in [R/'notebooks/h3_complete/browser_job.py',S/'job.py',S/'director_job.py']:exec(p.read_text(),s)
    return s

class DirectorPhases(unittest.TestCase):
    def test_interrupted_model_copy_resumes_and_checks_whole_hash(self):
        from test_h3_complete import scope_at
        with tempfile.TemporaryDirectory() as tmp:
            s=scope_at(Path(tmp));exec((S/'assets.py').read_text(),s)
            root=Path(tmp);source=root/'source.bin';source.write_bytes(b'abcdefgh')
            target=root/'local.bin';target.with_suffix('.bin.stage-partial').write_bytes(b'abcd')
            asset={'size':8,'sha256':hashlib.sha256(b'abcdefgh').hexdigest()}
            s['suite_stage_attached'](source,target,asset)
            self.assertEqual(target.read_bytes(),b'abcdefgh')
            self.assertFalse(target.with_suffix('.bin.stage-partial').exists())

    @unittest.skipUnless(os.environ.get('DIRECTOR_TEST_ROOT') and os.environ.get('COMFY_ROOT'),'Pinned Director and CPU runtime required')
    def test_real_pinned_director_prepares_without_a_model_and_validates_worker_schemas(self):
        import importlib.util,shutil,sys,types,asyncio
        from test_h3_complete_runtime import torch,load,nodes,execution,folder_paths
        from server import PromptServer
        from aiohttp import web
        PromptServer.instance=types.SimpleNamespace(routes=web.RouteTableDef())
        patcher={};exec((S/'director_prepare_patch.py').read_text(),patcher)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for p in Path(os.environ['DIRECTOR_TEST_ROOT']).glob('*.py'):shutil.copyfile(p,root/p.name)
            path=root/'minimax_director.py';patcher['patch_director_preparation'](path)
            once=path.read_text();patcher['patch_director_preparation'](path);self.assertEqual(once,path.read_text())
            pkg=types.ModuleType('director_actual');pkg.__path__=[str(root)];sys.modules[pkg.__name__]=pkg
            module=load(path,'director_actual.minimax_director');cls=module.MiniMaxH3Director
            class Clip:
                def tokenize(self,prompt,**kwargs):return prompt
                def encode_from_tokens_scheduled(self,tokens):return [[torch.ones(1,4,8),{}]]
            class Vae:
                def encode(self,image):return torch.zeros(1,24,1,2,2)
            data=json.loads((R/'tests/fixtures/h3_collections/Director - Three Step.json').read_text())['5']['inputs']
            data={k:v for k,v in data.items() if k not in ('model','model_ref2va','clip','vae','audio_vae')}
            for mode in ('OFF','ON'):
                state=json.loads(data['timeline_data']);state['reference_mode']=mode;data['timeline_data']=json.dumps(state)
                kw=dict(data,clip=Clip(),vae=Vae())
                if mode=='ON':kw['ref_images']=torch.zeros(1,32,32,3)
                import inspect
                out=tuple(cls.prepare(**{k:v for k,v in kw.items() if k in inspect.signature(cls.prepare).parameters}))
                self.assertIsNone(out[0]);self.assertEqual(len(out),10)
                self.assertEqual(out[7],124);self.assertEqual(out[2]['samples'].unbind()[0].shape[2],37)
            nodes.NODE_CLASS_MAPPINGS['MiniMaxH3DirectorCS']=cls
            s={'__name__':'director_schema_fixture','copy':copy,'comfy_nodes':nodes,'NODE_CLASS_MAPPINGS':{}}
            exec((S/'director_nodes.py').read_text(),s)
            nodes.NODE_CLASS_MAPPINGS.update(s['NODE_CLASS_MAPPINGS'])
            # This wrapper must inherit all actual timeline/reference inputs while
            # excluding both lazy MODEL sockets.
            schema=s['H3DirectorPrepare'].INPUT_TYPES()
            all_inputs={**schema.get('required',{}),**schema.get('optional',{})}
            self.assertNotIn('model',all_inputs);self.assertNotIn('model_ref2va',all_inputs)
            self.assertIn('ref_images',all_inputs);self.assertIn('timeline_data',all_inputs)
            # Serialize the actual native reference conditioning, not just a mock
            # conditioning tree. This exercises nested reference metadata too.
            cp=load(R/'notebooks/h3_complete/checkpoint_nodes.py','director_native_checkpoint')
            s.update(pack_latents=cp.pack_latents,checkpoint_path=cp.checkpoint_path,torch=torch,
                     suite_save_tensors=__import__('safetensors.torch',fromlist=['save_file']).save_file)
            with patch.object(folder_paths,'get_output_directory',return_value=str(root/'output')):
                s['H3DirectorPrepare']().save('native',**kw)
                self.assertTrue(s['director_prepared_path']('native').is_file())

    def test_all_director_branches_separate_models_and_preserve_outputs(self):
        s=scope()
        for file in (R/'tests/fixtures/h3_collections').glob('Director*.json'):
            for refs in ['OFF','ON']:
                g=json.loads(file.read_text());i=g['5']['inputs'];state=json.loads(i['timeline_data']);state['reference_mode']=refs;i['timeline_data']=json.dumps(state)
                prep,sample,decode=s['director_partition'](g,'403','test')
                self.assertIn('H3DirectorPrepare',{n['class_type'] for n in prep.values()})
                self.assertFalse({'H3SuiteUNETLoader','H3SuitePDDApply','H3SuiteLoraLoader'} & {n['class_type'] for n in prep.values()})
                self.assertFalse({'H3SuiteCLIPLoader','H3SuiteVAELoader','MiniMaxH3DirectorCS'} & {n['class_type'] for n in sample.values()})
                loaders=[n['inputs']['unet_name'] for n in sample.values() if n['class_type']=='H3SuiteUNETLoader']
                self.assertEqual(len(loaders),1);self.assertIn('ref2va' if refs=='ON' else 'fl2va',loaders[0])
                self.assertEqual(sample['5']['class_type'],'H3DirectorResume')
                self.assertNotIn('H3DirectorResume',{n['class_type'] for n in decode.values()})
                # Native audio replacement must keep the original Director audio output.
                g['403']['inputs']['audio']=['5',3]
                _,sample,decode=s['director_partition'](g,'403','test')
                self.assertEqual(sample['h3_suite_checkpoint']['inputs']['external_audio'],['5',3])
                self.assertIn('H3SuiteLoadAudio',{n['class_type'] for n in decode.values()})

    def test_sampling_failure_reuses_preparation_and_finishes_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);comfy=root/'Comfy';(comfy/'input').mkdir(parents=True)
            (comfy/'output/h3_latents').mkdir(parents=True)
            s=scope();g=json.loads((R/'tests/fixtures/h3_collections/Director - Three Step.json').read_text())
            geometry=False;fail=True;events=[]
            @contextmanager
            def process(phase,*args):
                events.append(('start',phase))
                try:yield None,None
                finally:events.append(('exit',phase))
            def execute(graph,proc,log,directory,phase):
                nonlocal geometry,fail
                if phase=='preparation':
                    key=graph['5']['inputs']['checkpoint_id'];(comfy/'output/h3_latents'/(key+'.director.safetensors')).touch()
                if phase=='sampling':
                    if fail:fail=False;raise RuntimeError('fixture sampling failure')
                    geometry=True
                return {}
            out=root/'clip.mp4';out.touch()
            s.update(ROOT=root,COMFY=comfy,TIMEOUT_MINUTES=90,DIRECTOR_CODE_ID='test',COMFY_COMMIT='pin',MGPU_COMMIT='pin',suite_preflight=lambda g:None,
                     suite_required_assets=lambda *a,**k:[],director_prepare_assets=lambda a:None,
                     suite_geometry=lambda k:geometry,validate_checkpoint=lambda k:None,comfy_process=process,execute_graph=execute,
                     video_paths=lambda e:[out],verify_video=lambda *a:None)
            with self.assertRaisesRegex(RuntimeError,'fixture'):s['run_browser_job']({'graph':g,'output_id':'403'})
            self.assertEqual(s['run_browser_job']({'graph':g,'output_id':'403'}),out)
            self.assertEqual(events,[('start','preparation'),('exit','preparation'),('start','sampling'),('exit','sampling'),('start','sampling'),('exit','sampling'),('start','decoding'),('exit','decoding')])
            self.assertEqual(s['TIMEOUT_MINUTES'],90)

    def test_phase_deadlines_restore_controller_settings_after_failure(self):
        s=scope();s['TIMEOUT_MINUTES']=90;observed=[]
        def execute(*args):
            observed.append(s['TIMEOUT_MINUTES']);raise TimeoutError('fixture')
        s['execute_graph']=execute
        for phase in ('preparation','sampling','decoding'):
            with self.assertRaises(TimeoutError):s['director_execute']({},None,None,None,phase)
            self.assertEqual(s['TIMEOUT_MINUTES'],90)
        self.assertEqual(observed,[5,10,5])

    def test_phase_budget_releases_encoder_before_diffusion_and_preserves_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            from test_h3_complete import scope_at
            root=Path(tmp);s=scope_at(root)
            for f in ['assets.py','director_storage.py']:exec((S/f).read_text(),s)
            store=root/'cache';store.mkdir()
            encoder={'repo':'fixture','revision':'pin','file':'text_encoders/encoder.bin','sha256':'enc','size':8}
            model=dict(encoder,file='diffusion_models/model.bin',sha256='model',size=9)
            p=s['stored_asset'](store,encoder);p.parent.mkdir(parents=True);p.write_bytes(b'12345678')
            unrelated=store/'user.txt';unrelated.write_text('preserve')
            ledger={'enc':{'path':str(p),'store':str(store),'mtime_ns':p.stat().st_mtime_ns}}
            s['SUITE_ASSETS']=[encoder,model]
            s['director_phase_budget'](store,[model],ledger,limit=10)
            self.assertFalse(p.exists());self.assertTrue(unrelated.exists())
            with self.assertRaisesRegex(RuntimeError,'24 GiB'):s['director_phase_budget'](store,[model],ledger,limit=5)

    @unittest.skipUnless(os.environ.get('COMFY_ROOT'),'Pinned CPU Comfy runtime required')
    def test_preparation_roundtrip_preserves_conditioning_references_audio_and_latent(self):
        from test_h3_complete_runtime import torch,NestedTensor,folder_paths
        s={'__name__':'director_checkpoint_fixture','copy':copy}
        exec((R/'notebooks/h3_complete/checkpoint_nodes.py').read_text(),s)
        s.update(comfy_nodes=type('N',(),{})(),suite_save_tensors=__import__('safetensors.torch',fromlist=['save_file']).save_file)
        exec((S/'director_nodes.py').read_text(),s)
        cond=[[torch.ones(1,4,8),{'minimax_refs':[{'latent':torch.ones(1,24,2,2,2)}],'tuple':(1,2)}]]
        latent={'samples':NestedTensor([torch.zeros(1,24,37,2,2),torch.zeros(1,32,2,200)])}
        audio={'waveform':torch.ones(1,2,800),'sample_rate':16000}
        class Director:
            @classmethod
            def prepare(cls,**kw):return (None,cond,latent,audio,24.,32,32,124,'prompt','retake')
        s['comfy_nodes'].NODE_CLASS_MAPPINGS={'MiniMaxH3DirectorCS':Director}
        with tempfile.TemporaryDirectory() as tmp,patch.object(folder_paths,'get_output_directory',return_value=tmp):
            s['H3DirectorPrepare']().save('fixture')
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
            with patch.object(MiniMaxH3SigmaShift,'execute',return_value=('shifted',)):
                out=s['H3DirectorResume']().load(object(),12.,3.,'fixture')
            self.assertEqual(len(out),10);self.assertEqual(out[0],'shifted');self.assertEqual(out[8:9],('prompt',))
            self.assertTrue(torch.equal(out[1][0][1]['minimax_refs'][0]['latent'],cond[0][1]['minimax_refs'][0]['latent']))
            self.assertEqual(out[1][0][1]['tuple'],(1,2))
            self.assertTrue(torch.equal(out[3]['waveform'],audio['waveform']))
            self.assertTrue(torch.equal(out[2]['samples'].unbind()[0],latent['samples'].unbind()[0]))
