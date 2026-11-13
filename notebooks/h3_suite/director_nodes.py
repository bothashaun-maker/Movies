"""Persist Director's actual conditioning between preparation and sampling."""
import json
def director_prepared_path(key):return checkpoint_path(key).with_suffix('.director.safetensors')

def director_pack(value,tensors):
    if torch.is_tensor(value):
        key='payload_'+str(len(tensors));t=value.detach().cpu().contiguous()
        if not torch.isfinite(t).all():raise ValueError('Non-finite Director conditioning.')
        tensors[key]=t;return {'tensor':key}
    if isinstance(value,dict):return {'dict':[[k,director_pack(v,tensors)] for k,v in value.items()]}
    if isinstance(value,tuple):return {'tuple':[director_pack(v,tensors) for v in value]}
    if isinstance(value,list):return {'list':[director_pack(v,tensors) for v in value]}
    if value is None or isinstance(value,(str,int,float,bool)):return {'value':value}
    raise TypeError('Unsupported Director checkpoint type: '+type(value).__name__)

def director_unpack(tree,tensors):
    if 'tensor' in tree:return tensors[tree['tensor']]
    if 'dict' in tree:return {k:director_unpack(v,tensors) for k,v in tree['dict']}
    if 'tuple' in tree:return tuple(director_unpack(v,tensors) for v in tree['tuple'])
    if 'list' in tree:return [director_unpack(v,tensors) for v in tree['list']]
    return tree['value']

class H3DirectorPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        s=copy.deepcopy(comfy_nodes.NODE_CLASS_MAPPINGS['MiniMaxH3DirectorCS'].INPUT_TYPES())
        for group in ('required','optional'):
            for name in ('model','model_ref2va'):s.get(group,{}).pop(name,None)
        s.setdefault('required',{})['checkpoint_id']=('STRING',)
        return s
    RETURN_TYPES=();OUTPUT_NODE=True;FUNCTION='save';CATEGORY='H3 Suite/Recovery'
    def save(self,checkpoint_id,**kwargs):
        import inspect
        prepare=comfy_nodes.NODE_CLASS_MAPPINGS['MiniMaxH3DirectorCS'].prepare
        parameters=inspect.signature(prepare).parameters
        # V3 Director's frontend-only widgets are serialized in submitted graphs,
        # but are not execute() arguments. Native V3 filters these automatically.
        out=tuple(prepare(**{k:v for k,v in kwargs.items() if k in parameters}))
        tensors=pack_latents(out[2])
        payload=director_pack([out[1],*out[3:]],tensors)
        path=director_prepared_path(checkpoint_id);tmp=path.with_suffix('.partial')
        suite_save_tensors(tensors,str(tmp),metadata={'format':'director-prepared-v1','payload':json.dumps(payload)})
        tmp.replace(path);return ()

class H3DirectorResume:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'shift_video':('FLOAT',),'shift_audio':('FLOAT',),'checkpoint_id':('STRING',)}}
    RETURN_TYPES=('MODEL','CONDITIONING','LATENT','AUDIO','FLOAT','INT','INT','INT','STRING','STRING')
    FUNCTION='load';CATEGORY='H3 Suite/Recovery'
    def load(self,model,shift_video,shift_audio,checkpoint_id):
        from safetensors import safe_open
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        with safe_open(str(director_prepared_path(checkpoint_id)),framework='pt',device='cpu') as f:
            meta=f.metadata()
            if meta.get('format')!='director-prepared-v1':raise ValueError('Invalid Director preparation checkpoint.')
            tensors={k:f.get_tensor(k) for k in f.keys()}
        payload=director_unpack(json.loads(meta['payload']),tensors)
        shifted=MiniMaxH3SigmaShift.execute(model=model,shift_video=shift_video,shift_audio=shift_audio)[0]
        latent={'samples':NestedTensor([tensors['video'],tensors['audio']])}
        return (shifted,payload[0],latent,*payload[1:])

NODE_CLASS_MAPPINGS.update({'H3DirectorPrepare':H3DirectorPrepare,'H3DirectorResume':H3DirectorResume})
