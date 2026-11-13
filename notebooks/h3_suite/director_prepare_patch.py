"""Expose pinned Director's conditioning path without loading a diffusion model."""
def patch_director_preparation(path):
    import ast
    source=path.read_text()
    marker='# H3_DIRECTOR_PREPARATION_V1'
    if marker in source:return
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3Director')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    # Keep all upstream timeline, reference-image/video/audio and retake logic.
    # Only model selection and SigmaShift move to the sampling process.
    removed=[];body=[]
    for n in method.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('chosen_model','patched_model'):
            removed.append(n.targets[0].id);continue
        body.append(n)
    assert removed==['chosen_model','patched_model'],'Unexpected pinned Director model path.'
    result=body[-1]
    assert isinstance(result,ast.Return) and isinstance(result.value,ast.Call) and result.value.args[0].id=='patched_model'
    result.value.args[0]=ast.Constant(None)
    method.name='_h3_prepare';method.decorator_list=[];method.body=body
    code=ast.unparse(ast.fix_missing_locations(method))
    source+='\n'+marker+'\n'+code+'\nMiniMaxH3Director.prepare=classmethod(_h3_prepare)\n'
    compile(source,str(path),'exec');path.write_text(source)
