"""Reproduce the actual SM75 Sage compiler failure without a GPU.

Run with the isolated Triton 3.6.0 compiler on PYTHONPATH, SageAttention 1.0.6
and CPU Torch installed. This does not execute a kernel or load any model.
"""
from contextlib import redirect_stderr
import io
import importlib.metadata
import json

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sageattention.attn_qk_int8_block_varlen import _attn_fwd

assert triton.__version__ == '3.6.0', 'Use the exact compiler from the shipped GPU lock'
assert importlib.metadata.version('sageattention') == '1.0.6'
params = _attn_fwd.arg_names
signature = dict(zip(params[:10], ['*i8', '*i8', '*fp16', '*i32', '*i32', '*fp32', '*fp32', '*i32', '*i32', '*fp16']))
signature.update({name: 'i32' for name in params[10:18]})
constants = dict(H=2, num_kv_groups=1, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, STAGE=1)
source = ASTSource(_attn_fwd, signature=signature, constexprs=constants)
with redirect_stderr(io.StringIO()):
    try:
        triton.compile(source, target=GPUTarget('cuda', 75, 32), options={'num_warps': 8, 'num_stages': 4})
    except RuntimeError as error:
        assert 'PassManager::run failed' in str(error), str(error)
    else:
        raise AssertionError('Expected the reproduced T4 compiler failure; reassess the speed gate')
print(json.dumps({'reproduced': True, 'target': 'sm_75', 'triton': triton.__version__,
                  'sageattention': '1.0.6', 'gpu_executed': False, 'models_loaded': False}))
