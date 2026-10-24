import sys,importlib.util,tempfile,pathlib,json,types,unittest
from unittest.mock import patch
import os
ROOT=pathlib.Path(__file__).resolve().parents[1]
if not os.environ.get('COMFY_ROOT'):
 raise unittest.SkipTest('Set COMFY_ROOT to the pinned ComfyUI checkout for runtime audit tests.')
COMFY=pathlib.Path(os.environ['COMFY_ROOT'])
saved_argv=list(sys.argv)
sys.path.insert(0,str(COMFY));sys.argv=[sys.argv[0],'--cpu','--disable-metadata']
import comfy.options
comfy.options.enable_args_parsing()
import torch,folder_paths
spec=importlib.util.spec_from_file_location('h3_checkpoint_nodes',ROOT/'patches/h3_checkpoint_nodes.py')
cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_video import CreateVideo,SaveVideo
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
import av
sys.argv=saved_argv

class Audit(unittest.TestCase):
 def test_joint_checkpoint_roundtrip_and_validation(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(folder_paths,'get_output_directory',return_value=tmp),patch.object(cp,'release_sampling_memory') as release:
   tensors=[torch.randn(1,24,37,48,32),torch.randn(1,32,2,200)]
   cp.H3CheckpointBeforeDecode().save({'samples':NestedTensor(tensors)},'test')
   restored=cp.H3LoadDecodeCheckpoint().load('test')[0]['samples'].unbind()
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(tensors,restored)))
   self.assertEqual(release.call_count,2)
   self.assertLess(cp.checkpoint_path('test').stat().st_size,10*2**20)
   with self.assertRaises(ValueError):cp.checkpoint_path('../escape')
   tensors[0][0,0,0,0,0]=float('nan')
   with self.assertRaisesRegex(ValueError,'non-finite'):cp.pack_latents({'samples':NestedTensor(tensors)})
 def test_release_clears_both_tp_caches(self):
  instance=types.SimpleNamespace(shards=['gpu'],lora={'gpu':1},lora_sig=(1,))
  fake=types.SimpleNamespace(_SHARD_CACHE={'weights':instance.shards},_INSTANCES={'weights':instance})
  def release(drop_shards=False):
   self.assertTrue(drop_shards);instance.shards=None
  fake.release_gpu=release
  with patch.dict(sys.modules,{'audit.h3_tensor_parallel':fake}),patch.object(cp.comfy.model_management,'unload_all_models'),patch.object(cp.comfy.model_management,'soft_empty_cache'):
   cp.release_sampling_memory()
  self.assertFalse(fake._SHARD_CACHE);self.assertIsNone(instance.shards);self.assertEqual(instance.lora,{})
 def test_native_temporal_output_geometry(self):
  # Exercise real chunk planning/blending, with synthetic decoded clip pixels.
  vae=MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE);torch.nn.Module.__init__(vae)
  vae.clip_length=17;vae.vae_ratio=16;vae.vae_ratio_t=4;vae.token_drop=3;vae.tokens_chunk_size=5
  vae.token_overlap=2;vae.frame_pre_padding=3;vae.frame_overlap=5
  vae.decoder=types.SimpleNamespace(out_channels=3)
  vae._adaptive_decode=lambda z:torch.zeros((1,3,z.shape[2]*4,z.shape[3]*16,z.shape[4]*16))
  vae._finalize_pixels=lambda x:x
  for latent_t,expected in [(2,5),(7,22),(37,124)]:
   out=vae.decode_temporal(torch.zeros(1,24,latent_t,2,2))
   self.assertEqual(tuple(out.shape),(1,3,expected,32,32))
 def test_create_save_and_decode_real_mp4_with_audio(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(folder_paths,'get_output_directory',return_value=tmp):
   images=torch.rand(24,32,32,3)
   audio={'waveform':torch.sin(torch.arange(32000)[None,None,:]*.03).expand(1,2,-1).contiguous(),'sample_rate':32000}
   video=CreateVideo.execute(images,24.0,audio=audio,bit_depth=8).result[0]
   SaveVideo.execute(video,'audit/clip',{'format':'mp4','codec':{'codec':'h264'}})
   files=list(pathlib.Path(tmp).rglob('*.mp4'));self.assertEqual(len(files),1)
   with av.open(str(files[0])) as f:self.assertEqual(sum(1 for _ in f.decode(video=0)),24)
   with av.open(str(files[0])) as f:self.assertIsNotNone(next(f.decode(audio=0),None))

 def test_actual_executor_export_graph(self):
  import nodes,execution
  class Fixture:
   @classmethod
   def INPUT_TYPES(cls):return {'required':{}}
   RETURN_TYPES=('IMAGE','AUDIO');FUNCTION='make'
   def make(self):return (torch.rand(24,32,32,3),{'waveform':torch.zeros(1,2,32000),'sample_rate':32000})
  with tempfile.TemporaryDirectory() as tmp,patch.object(folder_paths,'get_output_directory',return_value=tmp),patch.dict(nodes.NODE_CLASS_MAPPINGS,{'H3AuditFixture':Fixture,'CreateVideo':CreateVideo,'SaveVideo':SaveVideo}):
   graph={'1':{'class_type':'H3AuditFixture','inputs':{}},'15':{'class_type':'CreateVideo','inputs':{'images':['1',0],'audio':['1',1],'fps':24.0,'bit_depth':8}},'16':{'class_type':'SaveVideo','inputs':{'video':['15',0],'filename_prefix':'test/video','format':'mp4','format.codec':'h264'}}}
   server=types.SimpleNamespace(client_id=None,last_node_id=None,send_sync=lambda *a:None)
   runner=execution.PromptExecutor(server,cache_args={'ram':2,'ram_inactive':0})
   runner.execute(graph,'audit',{},['16'])
   self.assertTrue(runner.success,runner.status_messages)
   files=list(pathlib.Path(tmp).rglob('*.mp4'));self.assertEqual(len(files),1)
   with av.open(str(files[0])) as f:self.assertEqual(sum(1 for _ in f.decode(video=0)),24)
   with av.open(str(files[0])) as f:self.assertIsNotNone(next(f.decode(audio=0),None))

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit))
 raise SystemExit(not result.wasSuccessful())
