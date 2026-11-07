import ast
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
for name in ['Movie_Upscaler_Kaggle_v1.ipynb','Movie_Upscaler_Model_Packs.ipynb']:
    nb=nbformat.read(REPO/name,as_version=4)
    nbformat.validate(nb)
    code=[c.source for c in nb.cells if c.cell_type=='code']
    assert len(code)==1
    compile(code[0],name,'exec')
    assert not any(c.get('outputs') for c in nb.cells if c.cell_type=='code')
    tree=ast.parse(code[0])
    assignments={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:assignments[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    if 'Kaggle_v1' in name:
        assert assignments['WORKFLOWS']==json.loads((ROOT/'workflows.json').read_text())
        for file in ['media.py','nodes.py']:
            assert assignments['FILES'][file]==(ROOT/file).read_text()
        assert assignments['FILES']['web/open.js']==(ROOT/'open.js').read_text()
    else:
        assert 'import torch' not in code[0]
        assert assignments['AUTO_CREATE_DATASET'] is True
        assert isinstance(tree.body[-1],ast.Expr) and tree.body[-1].value.func.id=='build_and_save'
    print('Validated schema, code and embedded payload:',name)
