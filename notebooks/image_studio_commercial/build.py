"""Build self-contained, single-code-cell Kaggle launch and CPU pack notebooks."""
from pathlib import Path
import json
from workflow_builder import build
R=Path(__file__).parent
workflows=build();licenses={p.name:p.read_text() for p in (R/'licenses').iterdir() if p.is_file()}
pack=(R/'asset_builder.py').read_text().replace('LICENSE_FILES = {}','LICENSE_FILES = '+repr(licenses))
source=(R/'launcher.py').read_text().replace('LICENSE_FILES = {}','LICENSE_FILES = '+repr(licenses))
for token,value in [('PACK_BUILDER_SOURCE = None','PACK_BUILDER_SOURCE = '+repr(pack)),('FACE_NODE_SOURCE = None','FACE_NODE_SOURCE = '+repr((R/'face_nodes.py').read_text())),('JJP_NODE_SOURCE = None','JJP_NODE_SOURCE = '+repr((R/'jjp_nodes.py').read_text())),('JJP_PRESETS_SOURCE = None','JJP_PRESETS_SOURCE = '+repr((R/'jjp_presets.json').read_text())),('CATALOG = {}','CATALOG = '+repr(json.loads((R/'catalog.json').read_text()))),('WORKFLOW_PAYLOAD = {}','WORKFLOW_PAYLOAD = '+repr(workflows))]:source=source.replace(token,value)
def notebook(filename,intro,code,gpu):
 compile(code,filename,'exec')
 nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU' if gpu else 'none'},'cells':[{'cell_type':'markdown','id':'guide','metadata':{},'source':intro.splitlines(True)},{'cell_type':'code','id':'start','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
 (R/filename).write_text(json.dumps(nb,indent=1))
notebook('Image_Studio_Commercial_v7.ipynb',(R/'README.md').read_text(),source,True)
notebook('Image_Studio_Commercial_Model_Packs.ipynb',(R/'MODEL_PACKS.md').read_text(),pack,False)
export=R/'workflow_files';export.mkdir(exist_ok=True)
for name,wf in workflows.items():
 import re
 (export/(re.sub(r'[^A-Za-z0-9_-]+','-',name)+'.json')).write_text(json.dumps(wf,indent=2))
print('Built',len(workflows),'workflows and two notebooks')
