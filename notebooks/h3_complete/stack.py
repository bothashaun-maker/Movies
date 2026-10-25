def pinned_checkout(url,commit,target):
    if not target.exists():
        target.mkdir(parents=True)
        run(['git','init',target]);run(['git','-C',target,'remote','add','origin',url])
    head=subprocess.run(['git','-C',target,'rev-parse','HEAD'],capture_output=True,text=True)
    if head.returncode==0:assert head.stdout.strip()==commit,'Existing checkout differs; choose a fresh ROOT.'
    else:
        run(['git','-C',target,'fetch','--depth','1','origin',commit])
        run(['git','-C',target,'checkout','--detach','FETCH_HEAD'])

pinned_checkout('https://github.com/comfyanonymous/ComfyUI.git',COMFY_COMMIT,COMFY)
MGPU=COMFY/'custom_nodes/ComfyUI-MultiGPU-Kaggle'
pinned_checkout('https://github.com/MrKK4/ComfyUI-MultiGPU-Kaggle.git',MGPU_COMMIT,MGPU)
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

constraints=ROOT/'constraints.txt'
constraints.write_text('torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\ntorchaudio==2.10.0+cu130\ncomfy-kitchen==0.2.35\ncomfy-aimdo==0.5.5\n')
run([PY,'-m','pip','install','-c',constraints,'-r',COMFY/'requirements.txt',
    'huggingface-hub==0.36.2','transformers==4.57.6'])
run([PY,'-m','pip','check'])
(ROOT/'installed-packages.txt').write_text(subprocess.check_output([PY,'-m','pip','freeze'],text=True,env=child_env))
kernel_probe="""import torch
import comfy_kitchen.backends.cuda as ck
with torch.inference_mode():
 for i in range(2):
  with torch.cuda.device(i):
   x=torch.randn(256,256,device=f'cuda:{i}',dtype=torch.float16)
   w=torch.randint(-8,8,(256,256),device=x.device,dtype=torch.int8)
   scales=torch.full((256,),.02,device=x.device,dtype=torch.float32)
   y=ck.int8_linear(x,w,scales,convrot=True,convrot_groupsize=256)
   torch.cuda.synchronize(i)
   assert y.shape==(256,256) and torch.isfinite(y).all()
   print('Native ConvRot INT8 kernel passed on GPU',i)
"""
run([PY,'-c',kernel_probe])
