"""H3 FP16 branches with FP32 residuals; no model/checkpoint replacement.

Range analysis: https://github.com/deepbeepmeep/WanGP/issues/2156 and
https://github.com/Amduraznak/minimax-h3-fp16-fix (MIT, Amduraznak).
This implementation uses WanGP's native projection/LoRA/offload calls and
bounded row chunks. Power-of-two gains preserve the linear operation up to
ordinary FP16 rounding; they do not make it bit-identical to FP32.
"""
import functools
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common import atomic_json

ROWS = 512


def require_finite(value, stage):
    if value is not None and not bool(torch.isfinite(value).all()):
        report = {'stage': stage, 'dtype': str(value.dtype), 'shape': list(value.shape)}
        if os.environ.get('JJP_OUTPUT'):
            atomic_json(Path(os.environ['JJP_OUTPUT'])/'numerical-error.json', report)
        raise FloatingPointError(f'H3 non-finite values at {stage}; stopped before exporting a black clip. '
                                 f'Details: {report}')
    return value


def normalized_input(norm, hidden, shift=None, scale=None, segments=None):
    """Normalize FP32 residuals, then cast only bounded branch inputs to FP16."""
    out = torch.empty_like(hidden, dtype=torch.float16)
    for start, stop, row in segments or [(0, hidden.shape[0], 0)]:
        for first in range(start, stop, ROWS):
            last = min(first + ROWS, stop)
            value = norm(hidden[first:last].float()).float()
            if scale is not None:
                value.mul_(1.0 + scale[row].float()).add_(shift[row].float())
            out[first:last].copy_(value)
    return require_finite(out, 'normalized FP16 branch input')


def project_add(layer, inputs, residual=None, gate=None, segments=None, gain=64.0):
    """Linear(x/gain), gain restored in FP32; keeps native linear/LoRA hooks."""
    out = torch.empty((*inputs.shape[:-1], layer.out_features), device=inputs.device,
                      dtype=torch.float32) if residual is None else residual
    for start, stop, row in segments or [(0, inputs.shape[0], 0)]:
        factor = gain if gate is None else gate[row].float() * gain
        for first in range(start, stop, ROWS):
            last = min(first + ROWS, stop)
            value = layer((inputs[first:last] / gain).to(torch.float16)).float()
            # Native H3 out_proj/fc2 have no bias; retain the linear identity for
            # ordinary biased layers too, without multiplying their bias by gain.
            bias = getattr(layer, 'bias', None)
            if bias is not None:
                value.sub_(bias.float())
            if residual is None:
                out[first:last].copy_(value * factor)
            else:
                out[first:last].add_(value * factor)
            if bias is not None:
                out[first:last].add_(bias.float() if gate is None else bias.float() * gate[row].float())
    return out


def safe_mlp(layer, x_list, residual=None, gate=None, segments=None):
    x = x_list.pop()
    out = torch.empty_like(x, dtype=torch.float32) if residual is None else residual
    for start, stop, row in segments or [(0, x.shape[0], 0)]:
        factor = 256.0 if gate is None else gate[row].float() * 256.0
        for first in range(start, stop, ROWS):
            last = min(first + ROWS, stop)
            packed = layer.fc1(x[first:last].to(torch.float16))
            gate_value, up = packed.chunk(2, dim=-1)
            activation = F.silu(gate_value.float()).mul_(up.float()).div_(256.0)
            del packed, gate_value, up
            value = layer.fc2(activation.to(torch.float16)).float()
            del activation
            if residual is None:
                out[first:last].copy_(value * factor)
            else:
                out[first:last].add_(value * factor)
    return out


def install(namespace):
    """Install before model construction/loading, so MMGP sees the dtype locks."""
    Model = namespace['MiniMaxH3Model']
    if getattr(Model, '_jjp_safe_fp16_installed', False):
        return
    Refiner, Block = namespace['RefinerBlock'], namespace['DiTBlock']
    Attention, MLP = namespace['Attention'], namespace['MLP']
    original_init, original_preprocess = Model.__init__, Model.preprocess_text_embeds
    original_refiner, original_block = Refiner.forward, Block.forward
    original_attention, original_projection = Attention.forward, Attention._out_projection
    original_mlp = MLP.forward

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._jjp_safe_fp16 = self.dtype == torch.float16
        if not self._jjp_safe_fp16:
            return
        self.condition_proj.to(dtype=torch.float32)
        self.condition_proj._lock_dtype = torch.float32
        self.token_refiner._lock_dtype = torch.float32
        modules = [*self.token_refiner.blocks, *self.blocks]
        for block in modules:
            block._jjp_safe_fp16 = True
            block._lock_dtype = torch.float32
            block.attn._jjp_safe_fp16 = block.mlp._jjp_safe_fp16 = True
            # MMGP must not downcast a borrowed FP32 residual passed by keyword.
            block.attn._lock_dtype = block.mlp._lock_dtype = torch.float32
            for norm in (block.norm1, block.norm2):
                norm.to(dtype=torch.float32)
                norm._lock_dtype = torch.float32
        for norm in (self.token_refiner.final_norm, self.final_layer.norm):
            norm.to(dtype=torch.float32)
            norm._lock_dtype = torch.float32
        print('JJP H3 safe FP16: FP32 condition projection/residuals; scaled FP16 branches; finite checks ON.', flush=True)

    @functools.wraps(original_preprocess)
    def preprocess(self, text_states):
        if not getattr(self, '_jjp_safe_fp16', False):
            return original_preprocess(self, text_states)
        require_finite(text_states, 'Qwen conditioning')
        return require_finite(original_preprocess(self, text_states.float()), 'condition projection / token refiner')

    @functools.wraps(original_attention)
    def attention(self, x_list, *args, **kwargs):
        if getattr(self, '_jjp_safe_fp16', False):
            x_list[0] = x_list[0].to(torch.float16)
        return original_attention(self, x_list, *args, **kwargs)

    @functools.wraps(original_projection)
    def projection(self, attention_list, seq_len, residual, gate, segments):
        if not getattr(self, '_jjp_safe_fp16', False):
            return original_projection(self, attention_list, seq_len, residual, gate, segments)
        return project_add(self.out_proj, attention_list.pop().reshape(seq_len, -1), residual, gate, segments)

    @functools.wraps(original_mlp)
    def mlp(self, x_list, residual=None, gate=None, segments=None):
        if not getattr(self, '_jjp_safe_fp16', False):
            return original_mlp(self, x_list, residual, gate, segments)
        return safe_mlp(self, x_list, residual, gate, segments)

    @functools.wraps(original_refiner)
    def refiner(self, x_list):
        if not getattr(self, '_jjp_safe_fp16', False):
            return original_refiner(self, x_list)
        hidden = x_list.pop().float()
        hidden = self.attn([normalized_input(self.norm1, hidden)], residual=hidden)
        return require_finite(self.mlp([normalized_input(self.norm2, hidden)], residual=hidden), 'token refiner residual')

    @functools.wraps(original_block)
    def block(self, x_list, temb, segments, rope, residual_signature_elements=0):
        if not getattr(self, '_jjp_safe_fp16', False):
            return original_block(self, x_list, temb, segments, rope, residual_signature_elements)
        if residual_signature_elements:
            raise ValueError('Safe FP16 motion test requires step skipping/cache OFF.')
        hidden = x_list.pop().float()
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_proj(temb)
        inputs = normalized_input(self.norm1, hidden, shift_a, scale_a, segments)
        hidden = self.attn([inputs], rope=rope, residual=hidden, gate=gate_a, segments=segments)
        del inputs
        inputs = normalized_input(self.norm2, hidden, shift_m, scale_m, segments)
        hidden = self.mlp([inputs], residual=hidden, gate=gate_m, segments=segments)
        return require_finite(hidden, 'DiT residual')

    Model.__init__, Model.preprocess_text_embeds = init, preprocess
    Refiner.forward, Block.forward = refiner, block
    Attention.forward, Attention._out_projection, MLP.forward = attention, projection, mlp
    Model._jjp_safe_fp16_installed = True


def self_test(device):
    """Small real overflow case for the actual runtime device; no model weights."""
    linear = torch.nn.Linear(4, 4, bias=False, device=device, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.fill_(128.0)
        x = torch.full((2, 4), 256.0, device=device, dtype=torch.float16)
        assert not torch.isfinite(linear(x)).all(), 'Expected FP16 output-range overflow in control case'
        result = project_add(linear, x)
        reference = F.linear(x.float(), linear.weight.float())
        torch.testing.assert_close(result, reference)
        assert result.dtype == torch.float32
    print('PASS: safe FP16 overflow control/projection on', device, flush=True)
