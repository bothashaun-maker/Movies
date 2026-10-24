# Paste this file into a new cell in the SAME Kaggle notebook session.
# Run after the reported metadata error, then rerun steps 5, 6 and 7.
from pathlib import Path
assert 'COMFY' in globals() and 'MODELS_READY' in globals() and MODELS_READY, 'Use the existing session with verified weights.'
MGPU=COMFY/'custom_nodes/ComfyUI-MultiGPU-Kaggle'
# Stop only the server process launched by this notebook, releasing its GPU allocations.
if 'server' in globals() and server.poll() is None:
    server.terminate()
    try:server.wait(timeout=15)
    except __import__('subprocess').TimeoutExpired:
        server.kill();server.wait(timeout=15)
# Compatibility with current Comfy-Org per-layer .comfy_quant metadata.
compat_source='"""Read both ComfyUI quantization layouts without rewriting checkpoint weights."""\nimport json\n\n\ndef read_h3_quant_layers(weights):\n    metadata=weights.metadata() or {}\n    if \'_quantization_metadata\' in metadata:\n        layers=json.loads(metadata[\'_quantization_metadata\'])[\'layers\']\n    else:\n        layers={}\n        for name in weights.keys():\n            if name.endswith(\'.comfy_quant\'):\n                data=weights.get_tensor(name)\n                layers[name[:-len(\'.comfy_quant\')]]=json.loads(bytes(data.tolist()).decode(\'utf-8\'))\n    if not layers:\n        raise ValueError(\'Checkpoint contains neither legacy nor per-layer quantization metadata.\')\n    return layers\n\n\ndef validate_h3_quant_layers(weights):\n    layers=read_h3_quant_layers(weights)\n    for suffix in (\'attn.qkv_proj\',\'attn.out_proj\',\'mlp.fc1\',\'mlp.fc2\'):\n        name=\'blocks.0.\'+suffix\n        conf=layers.get(name,{})\n        if conf.get(\'format\')!=\'int8_tensorwise\' or not conf.get(\'convrot\'):\n            raise ValueError(f\'H3 tensor parallel requires INT8 ConvRot: {name}: {conf}\')\n    return layers\n'
(MGPU/'h3_quant_compat.py').write_text(compat_source)
tp_path=MGPU/'h3_tensor_parallel.py'
tp_source=tp_path.read_text()
old='layers = json.loads(f.metadata()["_quantization_metadata"])["layers"]'
new='layers = read_h3_quant_layers(f)'
if old in tp_source:
    assert tp_source.count(old)==1,'Unexpected TP loader source; refusing an ambiguous patch.'
    tp_source=tp_source.replace(old,new)
    tp_source=tp_source.replace('from .tp_source import open_weights',
        'from .tp_source import open_weights\nfrom .h3_quant_compat import read_h3_quant_layers')
    tp_path.write_text(tp_source)
else:
    assert new in tp_source and 'from .h3_quant_compat import read_h3_quant_layers' in tp_source,'Unexpected TP loader revision.'
print('Installed H3 quantization metadata compatibility reader.')


print('Patch installed. Rerun steps 5, 6 and 7. Skip step 4: existing verified weights are retained.')
