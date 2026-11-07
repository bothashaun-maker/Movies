from pathlib import Path
import json
R=Path(__file__).parent
source=(R/'asset_builder.py').read_text().replace('ASSET_CATALOG_SOURCE = None',(R/'assets.py').read_text())
intro=(R/'asset_intro.md').read_text()
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},'language_info':{'name':'python'},'accelerator':'none'},'cells':[{'cell_type':'markdown','id':'pack-guide','metadata':{},'source':intro.splitlines(True)},{'cell_type':'code','id':'pack-builder','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
(R/'Image_Studio_Model_Packs.ipynb').write_text(json.dumps(nb,indent=1))
print('Built CPU model pack notebook')
