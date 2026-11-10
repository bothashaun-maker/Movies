"""Resolve approved models and automatically prepare the missing finish pack."""
from pathlib import Path
import json,os,shutil

def prepare_models(assets,builder_source,roots,output_root,require_attached=True,auto_finish=True,save_dataset=True):
 namespace={'__name__':'studio_embedded_pack_builder'}
 exec(compile(builder_source,'embedded_pack_builder','exec'),namespace)
 digest=namespace['file_sha'];roots=list(map(Path,roots));resolved={}
 def scan():
  for asset in assets:
   if asset['name'] in resolved:continue
   for root in roots:
    if not root.exists():continue
    found=next((p for p in root.rglob(asset['name']) if p.is_file() and p.stat().st_size==asset['bytes'] and digest(p)==asset['sha256']),None)
    if found:resolved[asset['name']]=found;break
 scan()
 missing=[a for a in assets if a['name'] not in resolved]
 core=[a['name'] for a in missing if a['pack']!='finish']
 if core and require_attached:raise RuntimeError('Attach your saved create_edit Dataset or notebook output. Missing: '+', '.join(core)+'. No model downloads started.')
 finishing=[a for a in missing if a['pack']=='finish']
 if finishing and auto_finish:
  print('Preparing missing finish pack automatically; saved matching models will be reused.',flush=True)
  # Builder uses SHA checks/resumable transfer and the same direct private-create
  # method the user confirmed working in Model Packs v3.
  pack=Path(output_root)/'finish';pack.mkdir(parents=True,exist_ok=True)
  for asset in assets:
   source=resolved.get(asset['name']);target=pack/asset['name']
   if asset['pack']=='finish' and source and not target.exists():
    try:os.link(source.resolve(),target)
    except OSError:
     if shutil.disk_usage(pack).free<asset['bytes']+512*1024**2:raise RuntimeError('Insufficient disk to retain finishing model: '+asset['name'])
     shutil.copyfile(source,target)
  namespace['build_pack']('finish',output_root=Path(output_root),input_root=roots[0])
  pack=Path(output_root)/'finish'
  roots.append(pack);scan()
  if save_dataset:
   try:namespace['publish_pack'](pack)
   except (Exception,SystemExit) as error:
    # Locally verified models remain usable even if Dataset publication fails.
    # The builder retains a sanitised error record; never misreport persistence.
    print('Finish pack is ready locally; private Dataset submission was not confirmed. Save Version with outputs to preserve it. '+namespace['kaggle_error_text'](error),flush=True)
 remaining=[a['name'] for a in assets if a['name'] not in resolved]
 if remaining and require_attached:raise RuntimeError('Missing saved models: '+', '.join(remaining)+'. Attach the finish pack or enable AUTO_PREPARE_FINISH_PACK.')
 return resolved
