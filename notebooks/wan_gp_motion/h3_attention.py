"""Optional, probed SageAttention 1 for H3 only. Qwen/VAE retain SDPA."""
from contextlib import contextmanager
from pathlib import Path
import functools
import importlib.metadata
import json
import os
import sys

from common import atomic_json, bounded


def speed_preflight(mode):
    """Reject the reproduced SM75 compiler failure before any costly setup."""
    if mode not in ('auto', 'sdpa'):
        raise ValueError("SPEED_MODE must be 'auto' or 'sdpa'.")
    if mode == 'auto':
        raise RuntimeError(
            'Sage speed mode is blocked for this pinned T4/Triton 3.6/SageAttention 1.0.6 stack. '
            'The T4 target compiler fails in the INT8 attention dot product (arith.extf / PassManager::run failed). '
            'This was reproduced offline. No model staging, package installation, GPU probe or GUI was started. '
            "SPEED_MODE='sdpa' explicitly selects the existing slow working backend; it is not a speed fix. "
            'Do not repeat GPU speed tests with this stack.'
        )


@contextmanager
def attention_scope(state, mode):
    missing = object()
    previous = {key: state.get(key, missing) for key in ('_attention', '_sage_v_scaling')}
    state.update(_attention=mode, _sage_v_scaling=True)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                state.pop(key, None)
            else:
                state[key] = value


def h3_only(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        if os.environ.get('JJP_H3_ATTENTION', 'sdpa') != 'sage':
            return function(*args, **kwargs)
        from mmgp import offload
        with attention_scope(offload.shared_state, 'sage'):
            return function(*args, **kwargs)
    run._jjp_h3_attention = True
    return run


def probe(output):
    """Actual-device compatibility/precision check; never a render benchmark."""
    import torch
    from mmgp import offload
    from shared.attention import pay_attention, get_supported_attention_modes
    assert importlib.metadata.version('sageattention') == '1.0.6'
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5)
    assert 'sage' in get_supported_attention_modes(), 'Native SageAttention 1 unavailable on this runtime'
    generator = torch.Generator(device='cuda').manual_seed(8675309)
    errors = []
    with torch.inference_mode():
        for amplitude in (1.0, 4096.0):
            q, k, v = [torch.randn((1, 257, 2, 128), device='cuda', dtype=torch.float16,
                                  generator=generator) for _ in range(3)]
            v.mul_(amplitude)
            with attention_scope(offload.shared_state, 'sdpa'):
                reference = pay_attention([q.clone(), k.clone(), v.clone()])
            with attention_scope(offload.shared_state, 'sage'):
                actual = pay_attention([q.clone(), k.clone(), v.clone()])
            torch.cuda.synchronize()
            assert bool(torch.isfinite(actual).all()), 'SageAttention produced non-finite output'
            error = float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
            assert error <= 0.05, f'SageAttention relative L2 error {error:.5f} exceeds probe limit'
            errors.append(error)
        # Production H3 has split q/k/v and uses attention_kit's staged Sage
        # route, not just pay_attention. Exercise that real class as well.
        from models.minimax_h3.transformer import Attention
        attention = Attention(256, 2, 128, 1e-5, dtype=torch.float16, device='cuda')
        for name in ('q_proj', 'k_proj', 'v_proj'):
            setattr(attention, name, torch.nn.Linear(256, 256, bias=False, dtype=torch.float16, device='cuda'))
        del attention.qkv_proj
        attention._jjp_safe_fp16 = True
        inputs = torch.randn((257, 256), device='cuda', dtype=torch.float16, generator=generator)
        rope = torch.zeros((1, 257, 1, 64, 2), device='cuda', dtype=torch.float32)
        rope[..., 0] = 1.0
        with attention_scope(offload.shared_state, 'sdpa'):
            reference = attention([inputs.clone()], rope=rope)
        with attention_scope(offload.shared_state, 'sage'):
            actual = attention([inputs.clone()], rope=rope)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(actual).all()), 'Staged H3 SageAttention produced non-finite output'
        error = float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
        assert error <= 0.05, f'Staged H3 SageAttention relative L2 error {error:.5f} exceeds probe limit'
        errors.append(error)
    atomic_json(output, {'selected': 'sage', 'package': 'sageattention==1.0.6',
                         'relative_l2_errors': errors, 'compute_capability': [7, 5],
                         'staged_h3_attention': True,
                         'torch': torch.__version__, 'production_render_tested': False})
    print('PASS: native H3 attention path, FP16/head128, ordinary and large V inputs. '
          'Relative L2 errors:', errors, flush=True)


def choose(runtime, output, python, uv, app, env, mode):
    if mode not in ('auto', 'sdpa'):
        raise ValueError("SPEED_MODE must be 'auto' or 'sdpa'.")
    runtime, output = Path(runtime), Path(output)
    receipt = output/'attention-probe.json'
    # Do not consume a prior run's success if installation or this probe fails.
    atomic_json(receipt, {'selected': 'sdpa', 'requested': mode})
    if mode == 'sdpa':
        return 'sdpa'
    try:
        bounded([uv, 'pip', 'install', '--python', python, '--no-deps', '--require-hashes',
                 '-r', runtime/'sage-requirements.lock'], output/'sage-install.log', 90, env=env)
        bounded([python, runtime/'h3_attention.py', receipt], output/'attention-probe.log', 120, cwd=app, env=env)
        report = json.loads(receipt.read_text())
        if report.get('selected') != 'sage':
            raise RuntimeError('SageAttention probe did not report success')
        return 'sage'
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        atomic_json(receipt, {'selected': 'sdpa', 'requested': mode, 'reason': str(error)})
        print('Optional SageAttention check failed; retaining working SDPA. See attention-probe.json.', flush=True)
        return 'sdpa'


if __name__ == '__main__':
    probe(Path(sys.argv[1]))
