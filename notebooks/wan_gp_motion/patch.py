"""Small exact-source patch. Generation and conditioning remain upstream WanGP."""
from pathlib import Path
import ast


def patch_wgp(path):
    path = Path(path)
    source = path.read_text()
    if '# JJP_MOTION_GUARDS_V1' in source:
        return
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    validate, generate = functions['validate_settings'], functions['generate_media']
    assert [a.arg for a in validate.args.args] == ['state', 'model_type', 'single_prompt', 'inputs', 'silent']
    assert not any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(generate))
    lines = source.splitlines(keepends=True)
    # Insert the guard after local err() is defined; return WanGP's normal validation tuple.
    error_function = next(node for node in validate.body if isinstance(node, ast.FunctionDef) and node.name == 'err')
    insertions = [(error_function.end_lineno, '\n    try:\n        _jjp_check_shot(model_type, inputs)\n    except (ValueError, OSError, TimeoutError) as error:\n        return err(str(error))\n')]
    first_decorator = min(node.lineno for node in generate.decorator_list)
    insertions.append((first_decorator-1, '@_jjp_guarded\n'))
    # Imports must follow __future__ imports if upstream ever adds them.
    insertions.append((0, '# JJP_MOTION_GUARDS_V1\nfrom guard import guarded as _jjp_guarded, check_shot as _jjp_check_shot\n'))
    for index, text in sorted(insertions, reverse=True):
        lines.insert(index, text)
    result = ''.join(lines)
    # Upstream's header guesses compute dtype from the checkpoint filename. INT8
    # filenames have no fp16 suffix, so this otherwise misleadingly displays BF16.
    original_header = 'if "fp16" in model_filename:'
    assert result.count(original_header) == 1
    result = result.replace(original_header, 'if get_transformer_dtype(model_type, transformer_dtype_policy) == torch.float16:', 1)
    compile(result, str(path), 'exec')
    path.write_text(result)


def patch_h3_dtype(path):
    """H3's hard-coded BF16 model definition overrides --fp16; fix that before model load.

    The original video VAE is already FP16. Keep its filename selector ("bf16")
    and do not modify its learned weights, PDD arithmetic or conditioning path.
    """
    path = Path(path)
    source = path.read_text()
    marker = '# JJP_T4_H3_DTYPE_V1'
    if marker in source:
        return
    original = '"dtype": "bf16",'
    # This return belongs to the common H3 video model definition, not the image-only branch.
    anchor = '        result = {\n            "dtype": "bf16",'
    assert source.count(anchor) == 1, 'Pinned H3 model-definition layout changed'
    patched = source.replace(anchor, '        '+marker+'\n        result = {\n            "dtype": "fp16",\n            **({"resolutions": [["JJP 864x480", "864x480"]], "lock_resolutions": True}\n               if model_def.get("name") == "MiniMax H3 Ref2VA Pruned PDD 8-Step 20B" else {}),', 1)
    reference_size = '"reference_video_max_size": (768, 1344),'
    assert patched.count(reference_size) == 1, 'Pinned native reference-size control changed'
    patched = patched.replace(reference_size,
                              '"reference_video_max_size": (160, 288) if model_def.get("name") == "MiniMax H3 Ref2VA Pruned PDD 8-Step 20B" else (768, 1344),', 1)
    compile(patched, str(path), 'exec')
    path.write_text(patched)


if __name__ == '__main__':
    import sys
    patch_wgp(sys.argv[1])
