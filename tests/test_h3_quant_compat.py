import importlib.util
import json
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h3_quant_compat',REPO/'patches/h3_quant_compat.py')
compat=importlib.util.module_from_spec(spec);spec.loader.exec_module(compat)

class Tensor:
    def __init__(self,value):self.value=value
    def tolist(self):return list(json.dumps(self.value).encode())

class Weights:
    def __init__(self,legacy=False,empty=False):
        self.layers={} if empty else {'blocks.0.'+s:{'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}
                    for s in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')}
        self.legacy=legacy
    def metadata(self):return {'_quantization_metadata':json.dumps({'layers':self.layers})} if self.legacy else None
    def keys(self):return [k+'.comfy_quant' for k in self.layers]+['blocks.0.attn.qkv_proj.weight']
    def get_tensor(self,k):
        if not k.endswith('.comfy_quant'):raise AssertionError('Do not load model weights during metadata inspection')
        return Tensor(self.layers[k[:-len('.comfy_quant')]])

class QuantTests(unittest.TestCase):
    def test_old_and_new_layouts_agree(self):
        self.assertEqual(compat.validate_h3_quant_layers(Weights()),compat.validate_h3_quant_layers(Weights(legacy=True)))
    def test_missing_metadata_fails_clearly(self):
        with self.assertRaisesRegex(ValueError,'neither legacy nor per-layer'):compat.read_h3_quant_layers(Weights(empty=True))
    def test_wrong_format_is_rejected(self):
        weights=Weights();weights.layers['blocks.0.mlp.fc1']['format']='fp8'
        with self.assertRaisesRegex(ValueError,'requires INT8 ConvRot'):compat.validate_h3_quant_layers(weights)
    def test_notebook_embeds_exact_patch_and_patch_is_idempotent(self):
        n=json.loads((REPO/'minimax_h3_kaggle_dual_t4.ipynb').read_text())
        source=''.join(n['cells'][6]['source'])
        start=source.index('# Compatibility with current');end=source.index("constraints=ROOT/'constraints.txt'")
        import tempfile
        upstream=(REPO.parent/'artifacts/github_research/MultiGPU-Kaggle/h3_tensor_parallel.py').read_text() if (REPO.parent/'artifacts/github_research/MultiGPU-Kaggle/h3_tensor_parallel.py').exists() else 'from .tp_source import open_weights\ndef load(f):\n    layers = json.loads(f.metadata()["_quantization_metadata"])["layers"]\n'
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp);(target/'h3_tensor_parallel.py').write_text(upstream)
            exec(source[start:end],{'MGPU':target})
            first=(target/'h3_tensor_parallel.py').read_text()
            exec(source[start:end],{'MGPU':target})
            self.assertEqual(first,(target/'h3_tensor_parallel.py').read_text())
            self.assertEqual((target/'h3_quant_compat.py').read_text(),(REPO/'patches/h3_quant_compat.py').read_text())
            compile(first,'patched_loader','exec')

if __name__=='__main__':unittest.main()
