"""Narrow, repeatable source adaptation for CUDA SM75; no checkpoint rewriting."""
from pathlib import Path

def patch_molmo(root):
    path=Path(root)/'quantization/convrot_runtime.py'
    text=path.read_text()
    if '# H3_NEXT_T4_FP16_V1' in text:return
    # Native T4 tensor cores do not execute BF16. Keep INT8 weights and FP32
    # scales; convert only BF16 parameters/buffers to FP16 after assignment.
    text=text.replace('torch.bfloat16','torch.float16')
    old='return model.eval().to(device)'
    assert text.count(old)==1,'Molmo loader changed; refuse an ambiguous adaptation.'
    new='''# H3_NEXT_T4_FP16_V1
    for module in model.modules():
        for name, value in list(module._parameters.items()) + list(module._buffers.items()):
            if value is not None and value.dtype == torch.bfloat16:
                converted = value.to(torch.float16)
                if not torch.isfinite(converted).all():
                    raise ValueError('Molmo weights exceed FP16 range on T4: ' + name)
                if name in module._parameters:
                    module._parameters[name] = nn.Parameter(converted, requires_grad=False)
                else:
                    module._buffers[name] = converted
    model.config._attn_implementation = 'sdpa'
    return model.eval().to(device)'''
    path.write_text(text.replace(old,new))
