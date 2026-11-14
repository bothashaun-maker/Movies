"""Compile the actual T4 Sage kernel without a GPU: 3.6 fails, 3.2 uses MMA.

Select the isolated compiler through PYTHONPATH; never modify the video env.
This does not execute a kernel or load any model.
"""
from contextlib import redirect_stderr
import io
import importlib.metadata
import json
import inspect
import re

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sageattention.attn_qk_int8_block_varlen import _attn_fwd

assert triton.__version__ in ('3.6.0', '3.2.0'), 'Use the exact base or isolated Turing compiler'
assert importlib.metadata.version('sageattention') == '1.0.6'
params = _attn_fwd.arg_names
signature = dict(zip(params[:10], ['*i8', '*i8', '*fp16', '*i32', '*i32', '*fp32', '*fp32', '*i32', '*i32', '*fp16']))
signature.update({name: 'i32' for name in params[10:18]})
constants = dict(H=2, num_kv_groups=1, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, STAGE=1)
key = 'constexprs' if 'constexprs' in inspect.signature(ASTSource).parameters else 'constants'
source = ASTSource(_attn_fwd, signature=signature, **{key: constants})
compiled = None
with redirect_stderr(io.StringIO()):
    try:
        compiled = triton.compile(source, target=GPUTarget('cuda', 75, 32), options={'num_warps': 8, 'num_stages': 4})
    except RuntimeError as error:
        assert triton.__version__ == '3.6.0', str(error)
        assert 'PassManager::run failed' in str(error), str(error)
    else:
        assert triton.__version__ == '3.2.0', 'Expected the reproduced 3.6 failure'
instructions = [] if compiled is None else sorted(set(re.findall(r'mma\.[^\s;]+', compiled.asm['ptx'])))
if compiled is not None:
    assert any('s8.s8' in op for op in instructions), 'INT8 Tensor Core instructions missing'
    assert any('f16.f16' in op for op in instructions), 'FP16 Tensor Core instructions missing'
print(json.dumps({'compiled': compiled is not None, 'target': 'sm_75', 'triton': triton.__version__,
                  'mma': instructions, 'sageattention': '1.0.6', 'gpu_executed': False, 'models_loaded': False}))
