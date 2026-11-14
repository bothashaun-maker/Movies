"""Executed in the isolated runtime with CUDA_VISIBLE_DEVICES=0 before GUI launch."""
import importlib.metadata
import matplotlib
assert matplotlib.get_backend().lower() == 'agg', 'Standalone WanGP requires the Agg Matplotlib backend'
import torch

assert torch.__version__ == '2.10.0+cu128', torch.__version__
assert torch.cuda.is_available(), 'CUDA runtime cannot see the T4'
assert torch.cuda.device_count() == 1, 'Only GPU 0 must be visible to this process'
assert torch.cuda.get_device_capability() == (7, 5), 'This build targets T4 / SM75'
assert 'sm_75' in torch.cuda.get_arch_list(), 'Installed PyTorch wheel lacks SM75 kernels'
x = torch.randn(256, 256, dtype=torch.float16, device='cuda')
assert torch.isfinite(x @ x).all()
q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device='cuda')
assert torch.isfinite(torch.nn.functional.scaled_dot_product_attention(q, q, q)).all()
volume = torch.randn(1, 8, 4, 16, 16, dtype=torch.float16, device='cuda')
kernel = torch.randn(8, 8, 3, 3, 3, dtype=torch.float16, device='cuda')
assert torch.isfinite(torch.nn.functional.conv3d(volume, kernel, padding=1)).all()
torch.cuda.synchronize()
from models.minimax_h3.minimax_h3_handler import family_handler
import gradio
definition = family_handler.query_model_def('minimax_h3_ref2va_pruned',
                                          {'pdd': True, 'name': 'MiniMax H3 Ref2VA Pruned PDD 8-Step 20B'})
assert definition['dtype'] == 'fp16'
assert definition['reference_video_max_size'] == (160, 288)
print('PASS: T4 FP16 matmul, FP16 SDPA, FP16 Conv3D, native H3 handler and Gradio imports')
print('Versions:', {n: importlib.metadata.version(n) for n in ('torch', 'gradio', 'transformers', 'optimum-quanto')})
print('This is a kernel/import probe, not a model inference benchmark.')
