"""Read both ComfyUI quantization layouts without rewriting checkpoint weights."""
import json


def read_h3_quant_layers(weights):
    metadata=weights.metadata() or {}
    if '_quantization_metadata' in metadata:
        layers=json.loads(metadata['_quantization_metadata'])['layers']
    else:
        layers={}
        for name in weights.keys():
            if name.endswith('.comfy_quant'):
                data=weights.get_tensor(name)
                layers[name[:-len('.comfy_quant')]]=json.loads(bytes(data.tolist()).decode('utf-8'))
    if not layers:
        raise ValueError('Checkpoint contains neither legacy nor per-layer quantization metadata.')
    return layers


def validate_h3_quant_layers(weights):
    layers=read_h3_quant_layers(weights)
    for suffix in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2'):
        name='blocks.0.'+suffix
        conf=layers.get(name,{})
        if conf.get('format')!='int8_tensorwise' or not conf.get('convrot'):
            raise ValueError(f'H3 tensor parallel requires INT8 ConvRot: {name}: {conf}')
    return layers
