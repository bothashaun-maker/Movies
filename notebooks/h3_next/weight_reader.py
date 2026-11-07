"""Read prefixed community H3 checkpoints without copying/re-quantizing weights."""
import json
from safetensors import safe_open

class H3NextWeights:
    def __init__(self,path):
        self.handle=safe_open(path,framework='pt',device='cpu')
        keys=set(self.handle.keys())
        self.prefix=next((p for p in ('','model.diffusion_model.','diffusion_model.') if p+'blocks.0.attn.qkv_proj.weight' in keys),None)
        if self.prefix is None:raise ValueError('Unknown H3 checkpoint key layout.')
    def __enter__(self):return self
    def __exit__(self,*args):return self.handle.__exit__(*args)
    def keys(self):return [k[len(self.prefix):] for k in self.handle.keys() if k.startswith(self.prefix)]
    def get_tensor(self,name):
        tensor=self.handle.get_tensor(self.prefix+name)
        return tensor.reshape(-1) if name.endswith('.weight_scale') else tensor
    def get_slice(self,name):return self.handle.get_slice(self.prefix+name)
    def metadata(self):
        meta=dict(self.handle.metadata() or {})
        if '_quantization_metadata' in meta and self.prefix:
            q=json.loads(meta['_quantization_metadata']);q['layers']={k.removeprefix(self.prefix):v for k,v in q['layers'].items()};meta['_quantization_metadata']=json.dumps(q)
        return meta
