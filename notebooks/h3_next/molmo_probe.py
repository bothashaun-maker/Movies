"""Run the exact patched Molmo matmul on each GPU before model downloads."""
import importlib.util,sys,torch
path=sys.argv[1]
spec=importlib.util.spec_from_file_location('h3_molmo_probe',path)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
with torch.inference_mode():
    for device_index in range(torch.cuda.device_count()):
        device=torch.device('cuda',device_index)
        # The production failure had rows=216, K=4096, outputs=6144.
        weight=torch.randint(-8,9,(6144,4096),device=device,dtype=torch.int8)
        for rows in (1,33,216):
            a=torch.randint(-8,9,(rows,4096),device=device,dtype=torch.int8)
            got=runtime._h3_molmo_int8_mm(a,weight)
            expected=torch.nn.functional.linear(a.float(),weight[:64].float()).to(torch.int32)
            torch.cuda.synchronize(device)
            if not torch.equal(got[:,:64],expected):raise RuntimeError('Molmo production-shape INT8 accuracy probe failed.')
        # Exercise the patched forward call too, including rotation, activation
        # quantization, scales, dtype conversion and the 216-token shape.
        layer=runtime.ConvRotLinear(4096,6144,256)
        layer.qweight=weight
        layer.scales=torch.full((6144,),.02,device=device,dtype=torch.float32)
        x=torch.randn((1,216,4096),device=device,dtype=torch.float16)*.1
        actual=layer(x)
        rotated=runtime.rotate(x.float(),256).reshape(216,4096)
        scales=rotated.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127
        quant=(rotated/scales).round().clamp(-127,127)
        reference=(torch.nn.functional.linear(quant,weight[:64].float())*scales*.02).to(x.dtype)
        torch.cuda.synchronize(device)
        if not torch.isfinite(actual).all() or not torch.equal(actual[0,:,:64],reference):
            raise RuntimeError('Patched Molmo forward accuracy probe failed.')
        del layer,x,actual,rotated,scales,quant,reference,weight,got,a,expected
        print('Molmo production-shape probe passed on',torch.cuda.get_device_name(device_index),'device',device_index,
              '(bounded fallback)' if str(device) in runtime._H3_MOLMO_FALLBACK_DEVICES else '(native INT8)',flush=True)
