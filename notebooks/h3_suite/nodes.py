"""Visible suite loaders and lazy terminal: model work stays in child processes."""
import copy
import nodes as comfy_nodes
from safetensors.torch import save_file as suite_save_tensors, load_file as suite_load_tensors

def suite_config():return json.loads(Path(__file__).with_name('config.json').read_text())
def suite_choices(category):
    return sorted({Path(a.get('dest') or a['file']).name for a in suite_config()['assets']
                   if (a.get('dest') or a['file']).split('/')[0]==category})

def delegate(kind,kwargs):
    cls=comfy_nodes.NODE_CLASS_MAPPINGS[kind]
    return getattr(cls(),cls.FUNCTION)(**kwargs)

class H3SuiteUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'unet_name':(suite_choices('diffusion_models'),)}}
    RETURN_TYPES=('MODEL',);FUNCTION='load';CATEGORY='H3 Suite/Loaders'
    def load(self,unet_name):return delegate('UNETLoaderH3TensorParallel',dict(unet_name=unet_name))
class H3SuiteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'clip_name':(suite_choices('text_encoders'),),'type':(['minimax'],),'device':(['cuda:0','cuda:1','cpu'],)}}
    RETURN_TYPES=('CLIP',);FUNCTION='load';CATEGORY='H3 Suite/Loaders'
    def load(self,**kw):return delegate('CLIPLoaderMultiGPU',kw)
class H3SuiteVAELoader:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'vae_name':(suite_choices('vae'),),'device':(['cuda:0','cuda:1','cpu'],)}}
    RETURN_TYPES=('VAE',);FUNCTION='load';CATEGORY='H3 Suite/Loaders'
    def load(self,**kw):return delegate('VAELoaderMultiGPU',kw)
class H3SuiteLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'lora_name':(suite_choices('loras'),),'strength_model':('FLOAT',{'default':1.0,'min':-20,'max':20,'step':0.01})}}
    RETURN_TYPES=('MODEL',);FUNCTION='load';CATEGORY='H3 Suite/Loaders'
    def load(self,**kw):return delegate('LoraLoaderModelOnly',kw)
class H3SuitePDDApply:
    @classmethod
    def INPUT_TYPES(cls):
        schema=copy.deepcopy(comfy_nodes.NODE_CLASS_MAPPINGS['MiniMaxH3PDDAccApply'].INPUT_TYPES())
        for k,v in schema['required'].items():
            if isinstance(v[0],list) and ('file' in k or 'lora' in k):schema['required'][k]=(suite_choices('pdd_acc'),*v[1:])
        return schema
    RETURN_TYPES=('MODEL','SIGMAS','STRING');FUNCTION='load';CATEGORY='H3 Suite/Loaders'
    def load(self,**kw):return delegate('MiniMaxH3PDDAccApply',kw)
class H3SuiteSaveVideo(H3IsolatedSaveVideo):
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'images':('IMAGE',{'lazy':True}),'audio':('AUDIO',{'lazy':True}),'filename_prefix':('STRING',{'default':'H3_Suite/clip'}),'fps':('FLOAT',{'default':24.0,'min':24,'max':24})},'hidden':{'prompt':'PROMPT','unique_id':'UNIQUE_ID'}}
    DESCRIPTION='Runs this visible workflow in isolated sampling/decode workers. Downloads only missing selected weights.'
    def check_lazy_status(self,**kwargs):return []
    def render(self,filename_prefix,fps=24,images=None,audio=None,prompt=None,unique_id=None):
        return super().render(filename_prefix,prompt=prompt,unique_id=unique_id)
class H3SuiteCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'samples':('LATENT',),'checkpoint_id':('STRING',{'default':'clip'})},'optional':{'external_audio':('AUDIO',)}}
    RETURN_TYPES=('LATENT',);OUTPUT_NODE=True;FUNCTION='save';CATEGORY='H3 Suite/Recovery'
    def save(self,samples,checkpoint_id,external_audio=None):
        # Validate limits from actual latent geometry before permitting decode.
        tensors=pack_latents(samples);v=tensors['video'];frames=(v.shape[2]-2)//5*17+5
        if v.shape[-2]*v.shape[-1]*256>430000 or frames>124:raise ValueError('T4 limit: at most 0.43 MP and 124 frames.')
        if external_audio is not None:
            dest=checkpoint_path(checkpoint_id).with_suffix('.audio.safetensors');tmp=dest.with_suffix('.partial')
            suite_save_tensors({'waveform':external_audio['waveform'].detach().cpu().contiguous()},str(tmp),metadata={'sample_rate':str(external_audio['sample_rate'])});tmp.replace(dest)
        return H3CheckpointBeforeDecode().save(samples,checkpoint_id)
class H3SuiteLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'checkpoint_id':('STRING',{'default':'clip'})}}
    RETURN_TYPES=('AUDIO',);FUNCTION='load';CATEGORY='H3 Suite/Recovery'
    def load(self,checkpoint_id):
        from safetensors import safe_open
        path=checkpoint_path(checkpoint_id).with_suffix('.audio.safetensors')
        with safe_open(str(path),framework='pt',device='cpu') as f:return ({'waveform':f.get_tensor('waveform'),'sample_rate':int(f.metadata()['sample_rate'])},)

NODE_CLASS_MAPPINGS.update({c.__name__:c for c in (H3SuiteUNETLoader,H3SuiteCLIPLoader,H3SuiteVAELoader,H3SuiteLoraLoader,H3SuitePDDApply,H3SuiteSaveVideo,H3SuiteCheckpoint,H3SuiteLoadAudio)})
NODE_CLASS_MAPPINGS.update(CHECKPOINT_NODE_MAPPINGS)
