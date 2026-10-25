"""Persist sampled H3 video/audio before decoding; allow decode-only recovery."""
from pathlib import Path
import re
import torch
from safetensors.torch import save_file,load_file
import folder_paths
from comfy.nested_tensor import NestedTensor


def checkpoint_path(checkpoint_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',checkpoint_id):
        raise ValueError('Checkpoint ID must contain only letters, digits, underscore or hyphen.')
    root=Path(folder_paths.get_output_directory())/'h3_latents'
    root.mkdir(parents=True,exist_ok=True)
    return root/(checkpoint_id+'.safetensors')


def pack_latents(samples):
    latent=samples['samples']
    if not latent.is_nested:raise ValueError('Expected H3 joint video/audio latents.')
    parts=latent.unbind()
    if len(parts)!=2 or parts[0].ndim!=5 or parts[1].ndim!=4:
        raise ValueError('Unexpected H3 latent layout.')
    tensors={name:value.detach().to('cpu').contiguous() for name,value in zip(('video','audio'),parts)}
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError('Sampler produced non-finite latents; refusing to decode.')
    return tensors


class H3CheckpointBeforeDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'samples':('LATENT',),'checkpoint_id':('STRING',{'default':'clip'})}}
    RETURN_TYPES=('LATENT',)
    OUTPUT_NODE=True
    FUNCTION='save'
    CATEGORY='h3/recovery'
    def save(self,samples,checkpoint_id):
        tensors=pack_latents(samples)
        destination=checkpoint_path(checkpoint_id)
        partial=destination.with_suffix('.partial')
        save_file(tensors,str(partial),metadata={'format':'h3-joint-latent-v1'})
        partial.replace(destination)
        return ({**samples,'samples':NestedTensor([tensors['video'],tensors['audio']])},)


class H3LoadDecodeCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'checkpoint_id':('STRING',{'default':'clip'})}}
    @classmethod
    def VALIDATE_INPUTS(cls,checkpoint_id):
        try:return True if checkpoint_path(checkpoint_id).is_file() else 'Saved latent checkpoint does not exist.'
        except ValueError as error:return str(error)
    RETURN_TYPES=('LATENT',)
    FUNCTION='load'
    CATEGORY='h3/recovery'
    def load(self,checkpoint_id):
        tensors=load_file(str(checkpoint_path(checkpoint_id)),device='cpu')
        samples={'samples':NestedTensor([tensors['video'],tensors['audio']])}
        pack_latents(samples)
        return (samples,)


NODE_CLASS_MAPPINGS={'H3CheckpointBeforeDecode':H3CheckpointBeforeDecode,
                     'H3LoadDecodeCheckpoint':H3LoadDecodeCheckpoint}
