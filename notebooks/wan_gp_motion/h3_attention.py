"""Optional, probed SageAttention 1 for H3 only. Qwen/VAE retain SDPA."""
from contextlib import contextmanager
from pathlib import Path
import functools
import importlib.metadata
import json
import math
import os
import statistics
import sys

from common import atomic_json, bounded, sha256


def speed_preflight(mode):
    """Reject the reproduced SM75 compiler failure before any costly setup."""
    if mode not in ('auto', 'sdpa', 'sage'):
        raise ValueError("SPEED_MODE must be 'sage' or 'sdpa'. Legacy 'auto' is blocked.")
    if mode == 'auto':
        raise RuntimeError(
            'Sage speed mode is blocked for this pinned T4/Triton 3.6/SageAttention 1.0.6 stack. '
            'The T4 target compiler fails in the INT8 attention dot product (arith.extf / PassManager::run failed). '
            'This was reproduced offline. No model staging, package installation, GPU probe or GUI was started. '
            "SPEED_MODE='sdpa' explicitly selects the existing slow working backend; it is not a speed fix. "
            'Do not repeat GPU speed tests with this stack.'
        )


def prepare_turing_runtime(runtime, output, python, uv, env):
    """Use an isolated import directory; leave the working base environment intact.

    Torch 2.10's CUDA wheel normally pins Triton 3.6. WanGP documents 3.2 for
    Turing. This explicit exception is scoped to these child processes, with
    torch.compile disabled, not a resolver downgrade of the working runtime.
    """
    lock = Path(runtime)/'turing-attention.lock'
    target = Path(python).parent.parent/'jjp-turing-attention'
    marker = target/'jjp-ready'
    identity = sha256(lock)
    complete = all((target/name).is_dir() for name in
                   ('triton-3.2.0.dist-info', 'sageattention-1.0.6.dist-info', 'triton', 'sageattention'))
    if not complete or not marker.exists() or marker.read_text() != identity:
        bounded([uv, 'pip', 'install', '--python', python, '--target', target,
                 '--no-deps', '--require-hashes', '-r', lock],
                Path(output)/'turing-attention-install.log', 300, env=env)
        marker.write_text(identity)
    env['PYTHONPATH'] = str(target)+os.pathsep+env.get('PYTHONPATH', '')
    env['JJP_TURING_ATTENTION'] = '1'
    return target


def validate_benchmark(report):
    """Accept measured kernel acceleration only; this is not a clip-time estimate."""
    sdpa, sage = float(report['sdpa_ms']), float(report['sage_ms'])
    if not (math.isfinite(sdpa) and math.isfinite(sage) and 0 < sage <= 0.90*sdpa):
        raise RuntimeError('Sage attention did not beat SDPA by 10% in the bounded device check. '
                           'No model staging or render was started. See attention-probe.json.')


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
    turing = os.environ.get('JJP_TURING_ATTENTION') == '1'
    if turing:
        assert importlib.metadata.version('triton') == '3.2.0', 'Turing compiler isolation failed'
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
        benchmark = None
        if turing:
            # Every path is warmed up. Time alternating paths using CUDA events,
            # excluding kernel compilation and host launch timing. Small enough
            # to run before model staging, not a full-length H3 benchmark.
            q, k, v = [torch.randn((1, 2048, 8, 128), device='cuda', dtype=torch.float16,
                                  generator=generator) for _ in range(3)]
            for backend in ('sdpa', 'sage'):
                with attention_scope(offload.shared_state, backend):
                    result = pay_attention([q.clone(), k.clone(), v.clone()])
                    assert bool(torch.isfinite(result).all())
                    if backend == 'sdpa':
                        reference = result
                    else:
                        error = float((result.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
                        assert error <= 0.05, f'Benchmark-shape relative L2 error {error:.5f}'
                        errors.append(error)
                    del result
            del reference
            times = {'sdpa': [], 'sage': []}
            for iteration in range(5):
                for backend in (('sdpa', 'sage') if iteration % 2 == 0 else ('sage', 'sdpa')):
                    inputs = [q.clone(), k.clone(), v.clone()]
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with attention_scope(offload.shared_state, backend):
                        start.record()
                        result = pay_attention(inputs)
                        end.record()
                    end.synchronize()
                    times[backend].append(start.elapsed_time(end))
                    del result
            benchmark = {backend+'_ms': statistics.median(values) for backend, values in times.items()}
            benchmark.update(shape=[1, 2048, 8, 128], samples_per_backend=5,
                             gpu='Tesla T4', scope='attention kernel only; excludes model/offload/VAE')
    report = {'selected': 'sage', 'package': 'sageattention==1.0.6',
                         'relative_l2_errors': errors, 'compute_capability': [7, 5],
                         'staged_h3_attention': True,
                         'triton': importlib.metadata.version('triton'),
                         'torch': torch.__version__, 'production_render_tested': False,
                         'benchmark': benchmark}
    atomic_json(output, report)
    if turing:
        validate_benchmark(benchmark)
    print('PASS: native H3 attention path, FP16/head128, ordinary and large V inputs. '
          'Relative L2 errors:', errors, flush=True)


def choose(runtime, output, python, uv, app, env, mode):
    if mode not in ('auto', 'sdpa', 'sage'):
        raise ValueError("SPEED_MODE must be 'sage' or 'sdpa'.")
    runtime, output = Path(runtime), Path(output)
    receipt = output/'attention-probe.json'
    # Do not consume a prior run's success if installation or this probe fails.
    atomic_json(receipt, {'selected': 'sdpa', 'requested': mode})
    if mode == 'sdpa':
        return 'sdpa'
    if mode == 'sage':
        try:
            prepare_turing_runtime(runtime, output, python, uv, env)
            bounded([python, runtime/'h3_attention.py', receipt], output/'attention-probe.log',
                    120, cwd=app, env=env)
            report = json.loads(receipt.read_text())
            if not isinstance(report, dict):
                raise ValueError('Malformed Turing probe receipt')
            if report.get('selected') != 'sage' or report.get('triton') != '3.2.0':
                raise RuntimeError('Turing probe did not confirm Sage/Triton 3.2.')
            validate_benchmark(report['benchmark'])
            print('T4 attention benchmark (kernel only):', report['benchmark'], flush=True)
            return 'sage'
        except (OSError, RuntimeError, TimeoutError, ValueError, KeyError, TypeError) as error:
            try:
                failed = json.loads(receipt.read_text())
                if not isinstance(failed, dict):
                    failed = {}
            except (OSError, ValueError):
                failed = {}
            failed.update(selected=None, requested=mode, reason=str(error))
            atomic_json(receipt, failed)
            raise RuntimeError('Turing Sage check failed; no automatic SDPA render or model staging. '
                               'See attention-probe.json / attention-probe.log. '+str(error)) from error
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
