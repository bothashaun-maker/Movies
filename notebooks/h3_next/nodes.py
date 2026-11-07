"""Next-generation H3 controls; heavy work executes only in isolated workers."""
import importlib

class H3NextMolmo:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{}}
    RETURN_TYPES=('CLIP',);FUNCTION='load';CATEGORY='H3 Next/Loaders'
    DESCRIPTION='Pinned INT8 Molmo2 with an experimental FP16 T4 adaptation. No automatic model downloads.'
    def load(self):
        module=importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS['MolmoH3EncoderLoader'].__module__)
        root=Path(folder_paths.models_dir)/'text_encoders/Molmo2-8B-Minimax-H3'
        # Use our exact verified package, never the upstream loader's floating-main download.
        config=suite_config()
        for a in config['assets']:
            if a.get('pack')=='molmo':
                p=Path(folder_paths.models_dir)/a['dest']
                if not p.is_file() or p.stat().st_size!=a['size']:raise RuntimeError('Molmo package incomplete; rerun notebook setup: '+str(p))
        return (module.MolmoProjectedCLIP(str(root),module.resolve_projector(str(root))),)

class H3NextReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'clip':('CLIP',),'vae':('VAE',),'prompt':('STRING',{'multiline':True,'default':''}),
          'width':('INT',{'default':512,'min':32,'max':1024,'step':32}),
          'height':('INT',{'default':768,'min':32,'max':1024,'step':32}),
          'seconds':('FLOAT',{'default':5.,'min':1.,'max':5.,'step':.5})},
          'optional':{'reference_image':('IMAGE',)}}
    RETURN_TYPES=('CONDITIONING','LATENT');RETURN_NAMES=('positive','latent');FUNCTION='encode';CATEGORY='H3 Next/Conditioning'
    DESCRIPTION='Text or one optional reference image. Use <Picture 1> in the prompt. Frames snap to the H3 grid at 24 fps.'
    def encode(self,clip,vae,prompt,width,height,seconds,reference_image=None):
        frames=round(seconds*24);frames+=(5-frames%17)%17
        if width%32 or height%32 or width*height>430000 or not 5<=frames<=124:raise ValueError('Use at most 0.43 MP and five seconds on this T4 preset.')
        if len(prompt)>5000:raise ValueError('Keep the prompt within 5,000 characters.')
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        refs={} if reference_image is None else {'ref_image_1':reference_image}
        result=MiniMaxH3ReferenceToVideo.execute(clip=clip,vae=vae,prompt=prompt,width=width,height=height,length=frames,ref_image_size='match',ref_images=refs)
        # Abort on numerical overflow instead of producing a silent bad render on SM75.
        for tensor,metadata in result[0]:
            if not torch.isfinite(tensor).all():raise RuntimeError('Molmo conditioning is non-finite on this device; stop and retain diagnostics.')
        return tuple(result)

class H3NextMotion:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'strength':('FLOAT',{'default':0.,'min':-2.,'max':2.,'step':.05})}}
    RETURN_TYPES=('MODEL',);FUNCTION='apply';CATEGORY='H3 Next/Controls'
    DESCRIPTION='Optional movement control. Zero is off. This changes motion in the scene, not rendering speed.'
    def apply(self,model,strength):
        if strength==0:return (model,)
        dial=comfy_nodes.NODE_CLASS_MAPPINGS['DialsLoadDial'].execute(dial_name='dials/h3_motion.safetensors')[0]
        return tuple(comfy_nodes.NODE_CLASS_MAPPINGS['DialsApplyDial'].execute(model=model,dial=dial,strength=strength))

NODE_CLASS_MAPPINGS.update({c.__name__:c for c in (H3NextMolmo,H3NextReference,H3NextMotion)})
NODE_DISPLAY_NAME_MAPPINGS.update({'H3NextMolmo':'Molmo2 — T4 encoder (experimental)', 'H3NextReference':'Prompt, reference image and clip length', 'H3NextMotion':'Motion control — zero is off'})

# Conditioning is also isolated: the ~9 GB Molmo host model must not remain
# resident alongside a ~20 GB diffusion checkpoint in Kaggle's 30 GB RAM.
def next_prepared_path(key):return checkpoint_path(key).with_suffix('.prepared.safetensors')

def next_pack_tree(value,tensors):
    if torch.is_tensor(value):
        key='cond_'+str(len(tensors));tensor=value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():raise ValueError('Non-finite conditioning.')
        tensors[key]=tensor;return {'tensor':key}
    if isinstance(value,dict):return {'dict':[[k,next_pack_tree(v,tensors)] for k,v in value.items()]}
    if isinstance(value,(list,tuple)):return {'list':[next_pack_tree(v,tensors) for v in value]}
    if value is None or isinstance(value,(str,int,float,bool)):return {'value':value}
    raise TypeError('Unsupported conditioning checkpoint value: '+type(value).__name__)

def next_unpack_tree(tree,tensors):
    if 'tensor' in tree:return tensors[tree['tensor']]
    if 'dict' in tree:return {k:next_unpack_tree(v,tensors) for k,v in tree['dict']}
    if 'list' in tree:return [next_unpack_tree(v,tensors) for v in tree['list']]
    return tree['value']

class H3NextSavePreparation:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'conditioning':('CONDITIONING',),'latent':('LATENT',),'checkpoint_id':('STRING',)}}
    RETURN_TYPES=();FUNCTION='save';OUTPUT_NODE=True;CATEGORY='H3 Next/Recovery'
    def save(self,conditioning,latent,checkpoint_id):
        tensors=pack_latents(latent);tree=next_pack_tree(conditioning,tensors)
        dest=next_prepared_path(checkpoint_id);tmp=dest.with_suffix('.partial')
        suite_save_tensors(tensors,str(tmp),metadata={'conditioning':json.dumps(tree),'format':'h3-next-prepared-v1'});tmp.replace(dest)
        return ()

class H3NextLoadPreparation:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'checkpoint_id':('STRING',)}}
    RETURN_TYPES=('CONDITIONING','LATENT');FUNCTION='load';CATEGORY='H3 Next/Recovery'
    @classmethod
    def VALIDATE_INPUTS(cls,checkpoint_id):return next_prepared_path(checkpoint_id).is_file() or 'Prepared conditioning checkpoint is missing.'
    def load(self,checkpoint_id):
        from safetensors import safe_open
        path=next_prepared_path(checkpoint_id)
        with safe_open(str(path),framework='pt',device='cpu') as f:
            metadata=f.metadata();assert metadata['format']=='h3-next-prepared-v1'
            tensors={k:f.get_tensor(k) for k in f.keys()}
        return next_unpack_tree(json.loads(metadata['conditioning']),tensors),{'samples':NestedTensor([tensors['video'],tensors['audio']])}

NODE_CLASS_MAPPINGS.update({c.__name__:c for c in (H3NextSavePreparation,H3NextLoadPreparation)})
