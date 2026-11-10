"""Embed normal workflows and the unchanged saved-model/runtime support."""
from pathlib import Path
import json
from workflow_builder import build
R=Path(__file__).parent
workflows=build();licenses={p.name:p.read_text() for p in (R/'licenses').iterdir() if p.is_file()}
source=(R/'launcher.py').read_text()
for token,value in [('STARTUP_MODELS_SOURCE = None',(R/'startup_models.py').read_text()),('WORKFLOW_INSTALL_SOURCE = None',(R/'workflow_install.py').read_text()),('FACE_NODE_SOURCE = None','FACE_NODE_SOURCE = '+repr((R/'face_nodes.py').read_text())),('CATALOG = {}','CATALOG = '+repr(json.loads((R/'catalog.json').read_text()))),('LICENSE_FILES = {}','LICENSE_FILES = '+repr(licenses)),('WORKFLOW_PAYLOAD = {}','WORKFLOW_PAYLOAD = '+repr(workflows))]:source=source.replace(token,value)
compile(source,'normal studio','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','id':'guide','metadata':{},'source':(R/'README.md').read_text().splitlines(True)},{'cell_type':'code','id':'start','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
for name in ['Image_Studio_Normal.ipynb','Image_Studio_Normal_v10.ipynb']:(R/name).write_text(json.dumps(nb,indent=1))
folder=R/'workflow_files';folder.mkdir(exist_ok=True)
for name,wf in workflows.items():(folder/(name.replace(' ','_')+'.json')).write_text(json.dumps(wf,indent=2))
print('Built',len(workflows),'normal workflows; no model/dataset changes')
