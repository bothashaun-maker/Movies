"""Reviewed, narrow adaptations around the pinned AIQUEST engine."""
import ast
import hashlib
import json
from pathlib import Path

UPSTREAM_BLOB = '1fe403e80116e4b261a004e57b17c6a683f20d58'
ALLOWED_FUNCTION_CHANGES = {'_wrap_conditioner', 'encode_text_fast', 'set_denoising_steps',
                            'run_generation', 'save_mp4', '_t2v_ui'}


def extract_engine(raw):
    identity = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if identity != UPSTREAM_BLOB:
        raise RuntimeError('AIQUEST notebook identity differs from the reviewed source.')
    notebook = json.loads(raw)
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'RUN_SCRIPT_CODE' for n in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError('Pinned source has no engine.')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Reviewed integration target changed: '+old[:100])
    return source.replace(old, new, 1)


def prepare_engine(source):
    result = source
    edits = [
        ('print(f"🔧 Wan2GP fixes: {_fixed}/{len(_FIXES)} applied")',
         'print(f"🔧 Wan2GP fixes: {_fixed}/{len(_FIXES)} applied")\n'
         'if _fixed != len(_FIXES):\n    raise RuntimeError("A required pinned numerical patch did not apply. No model loaded.")'),
        ('            except Exception as retry_error:\n                print(f"  ⚠️ CPU re-encode failed ({retry_error}); non-finite values are zeroed instead")',
         '            except Exception as retry_error:\n                raise RuntimeError("Reference encoding failed; refusing to discard identity conditioning") from retry_error'),
        ('                fixed = torch.nan_to_num(fixed.float(), nan=0.0, posinf=0.0, neginf=0.0)',
         '                raise RuntimeError("Reference conditioning contains NaN/Inf after FP32 retry; no video generated")'),
        ('    return results\n\n\ndef _unload_mmgp():',
         '    if bad:\n        raise RuntimeError("Prompt encoder produced NaN/Inf; no video generated")\n    return results\n\n\ndef _unload_mmgp():'),
        ('def set_denoising_steps(num_steps):\n    schedule = STEP_SCHEDULES.get(int(num_steps), FULL_SIGMAS)',
         'def set_denoising_steps(num_steps):\n    if int(num_steps) != 8:\n        raise ValueError("This test edition uses the official eight-step distilled schedule")\n    schedule = FULL_SIGMAS'),
        ('        if read_state:  # interrupt polling, once per transformer block\n',
         '        if torch.is_tensor(latent) and not bool(torch.isfinite(latent).all()):\n'
         '            raise RuntimeError("Denoising produced NaN/Inf; refusing invalid video")\n'
         '        if read_state:  # interrupt polling, once per transformer block\n'),
        ('    _, t, h, w = video.shape\n    wav = path[:-4] + ".wav"',
         '    if not bool(torch.isfinite(video).all()):\n'
         '        raise RuntimeError("Decoded video contains NaN/Inf; export refused")\n'
         '    if audio is not None and not np.isfinite(np.asarray(audio)).all():\n'
         '        raise RuntimeError("Decoded audio contains NaN/Inf; export refused")\n'
         '    _, t, h, w = video.shape\n    wav = path[:-4] + ".wav"'),
        ('            steps = gr.Slider(4, 8, value=8, step=1, label="⚡ Denoising steps",\n'
         '                              info="8 = full quality; fewer steps are faster but softer")\n'
         '            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")\n\n'
         '    with gr.Row():\n        gen_btn = gr.Button("🎬 Generate Video",',
         '            steps = gr.Slider(4, 8, value=8, step=1, interactive=False, label="⚡ Denoising steps",\n'
         '                              info="Official eight-step distilled schedule; no reduced-step experiments")\n'
         '            gr.Markdown("Preset labels are rounded to the model grid. Default landscape: 832×512. Image inputs set their own rounded aspect ratio; exact dimensions appear in the job log.")\n'
         '            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")\n\n'
         '    with gr.Row():\n        gen_btn = gr.Button("🎬 Generate Video",'),
    ]
    for old, new in edits:
        result = replace_once(result, old, new)
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(result).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    changed = {name for name in before if before[name] != after.get(name)}
    if changed != ALLOWED_FUNCTION_CHANGES:
        raise RuntimeError('Unexpected engine changes: '+repr(changed))
    compile(result, 'run_ltx25.py', 'exec')
    return result


def build_engine(folder):
    folder = Path(folder)
    original = extract_engine((folder/'aiquest_upstream.ipynb').read_bytes())
    engine = prepare_engine(original)
    (folder/'engine.py').write_text(engine, encoding='utf-8')
    (folder/'provenance.json').write_text(json.dumps({
        'aiquest_blob': UPSTREAM_BLOB,
        'original_engine_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'engine_sha256': hashlib.sha256(engine.encode()).hexdigest(),
        'changed_functions': sorted(ALLOWED_FUNCTION_CHANGES),
        'additional_change': 'Fail clearly if a pinned upstream numerical patch is missing.',
        'unchanged': 'Tensor-parallel computation, INT8 kernels, FP16 math protections, native I2V/FL2V and media implementation.',
    }, indent=2)+'\n', encoding='utf-8')
