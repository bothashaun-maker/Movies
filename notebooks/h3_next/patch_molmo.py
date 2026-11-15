from pathlib import Path
MOLMO_INT8_SOURCE = '"""Molmo INT8 matmul dispatch; injected into the pinned ConvRot runtime.\n\nNo weights are converted permanently. The bounded fallback multiplies int8\nvalues in FP32 in K=256 chunks, then accumulates those exact integer partial\nsums in int32. Each partial\'s magnitude is <= 256*128*128 = 2**22.\n"""\nimport logging\n_H3_MOLMO_FALLBACK_DEVICES=set()\n\ndef _h3_int8_mm_fallback(a,weight):\n    rows,inner=a.shape;outputs=weight.shape[0]\n    result=torch.empty((rows,outputs),dtype=torch.int32,device=a.device)\n    for r in range(0,rows,256):\n        for o in range(0,outputs,1024):\n            total=torch.zeros((min(256,rows-r),min(1024,outputs-o)),dtype=torch.int32,device=a.device)\n            for k in range(0,inner,256):\n                partial=F.linear(a[r:r+256,k:k+256].float(),weight[o:o+1024,k:k+256].float())\n                total.add_(partial.to(torch.int32))\n            result[r:r+256,o:o+1024]=total\n    return result\n\ndef _h3_native_int8_mm(a,weight,turing):\n    n=a.shape[0]\n    # Keep the RHS column-major on Turing, rather than materializing a\n    # row-major transpose. Align token rows to 32, not the upstream 8.\n    multiple=32 if turing else 8\n    padded=F.pad(a,(0,0,0,max(32,((n+multiple-1)//multiple)*multiple)-n)).contiguous()\n    rhs=weight.contiguous().t() if turing else weight.t().contiguous()\n    return torch._int_mm(padded,rhs)[:n]\n\ndef _h3_molmo_int8_mm(a,weight):\n    if a.dtype!=torch.int8 or weight.dtype!=torch.int8 or a.ndim!=2 or weight.ndim!=2 or a.shape[1]!=weight.shape[1]:\n        raise ValueError(\'Molmo matmul requires [rows,K] and [outputs,K] int8 tensors.\')\n    if not a.is_cuda:return _h3_int8_mm_fallback(a,weight)\n    key=str(a.device)\n    if key not in _H3_MOLMO_FALLBACK_DEVICES:\n        try:\n            return _h3_native_int8_mm(a,weight,torch.cuda.get_device_capability(a.device)==(7,5))\n        except RuntimeError as error:\n            if \'CUBLAS_STATUS_NOT_SUPPORTED\' not in str(error):raise\n            _H3_MOLMO_FALLBACK_DEVICES.add(key)\n            logging.getLogger(__name__).warning(\'Molmo INT8 cuBLAS layout unsupported on %s; using bounded FP32 integer-partial fallback. Speed is not guaranteed.\',key)\n    return _h3_int8_mm_fallback(a,weight)\n'
def patch_molmo(root):
    path=Path(root)/'quantization/convrot_runtime.py'
    text=path.read_text()
    if '# H3_NEXT_T4_MM_V2' in text:return
    # Native T4 tensor cores do not execute BF16. Keep INT8 weights and FP32
    # scales; convert only BF16 parameters/buffers to FP16 after assignment.
    if '# H3_NEXT_T4_FP16_V1' not in text:
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
    if '# H3_NEXT_T4_FP16_V1' not in text:text=text.replace(old,new)
    old_mm="""            q = F.pad(q, (0,0,0,max(32, ((n+7)//8)*8)-n))
            y = torch._int_mm(q.contiguous(), self.qweight.t().contiguous())[:n].float()"""
    assert text.count(old_mm)==1,'Molmo INT8 forward changed; refusing an unverified patch.'
    text=text.replace(old_mm,'            y = _h3_molmo_int8_mm(q, self.qweight).float()')
    text+='\n# H3_NEXT_T4_MM_V2\n'+MOLMO_INT8_SOURCE+'\n'
    compile(text,str(path),'exec');path.write_text(text)
