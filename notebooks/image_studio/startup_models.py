"""Resolve saved image models and repair only the optional face-pack dependency."""
from pathlib import Path
import shutil

def model_candidates(folder,name,input_root=Path('/kaggle/input'),runtime_root=Path('/kaggle/temp'),pack_root=Path('/kaggle/working/image-studio-model-packs')):
 paths=[]
 for root in [Path(input_root),Path(pack_root)]:
  if root.exists():paths.extend(root.rglob(name))
 paths.extend(Path(runtime_root)/('image-studio-v'+str(v))/'ComfyUI/models'/folder/name for v in range(1,6))
 return [p for p in paths if p.is_file()]

def prepare_startup_models(assets,builder_source,require_attached=True,prepare_faces=True,save_face_dataset=True,input_root=Path('/kaggle/input'),runtime_root=Path('/kaggle/temp'),pack_root=Path('/kaggle/working/image-studio-model-packs'),builder=None):
 """Do not initialize a GPU or fetch existing generation-model families."""
 input_root=Path(input_root);runtime_root=Path(runtime_root);pack_root=Path(pack_root)
 def present(a):return [p for p in model_candidates(a[0],a[1],input_root,runtime_root,pack_root) if p.stat().st_size==a[4]]
 missing=[a for a in assets if not present(a)]
 missing_core=[a for a in missing if a[0]!='insightface']
 missing_face=[a for a in missing if a[0]=='insightface']
 if require_attached and missing_core:
  raise RuntimeError('Attach your saved standard/FLUX3 model pack. Missing: '+', '.join(a[1] for a in missing_core)+'. No generation-model downloads started.')
 if require_attached and missing_face and not prepare_faces:
  raise RuntimeError('Attach the faceswap model pack or enable AUTO_PREPARE_FACE_PACK. Missing: '+', '.join(a[1] for a in missing_face))
 owns_manifest=(pack_root/'faceswap/manifest.json').is_file()
 if prepare_faces and (missing_face or (owns_manifest and any(a[0]=='insightface' for a in assets))):
  if builder is None:
   builder={'__name__':'image_studio_pack_support'}
   exec(compile(builder_source,'embedded_image_pack_builder','exec'),builder)
  if missing_face:
   print('Preparing missing face-swap models on CPU (up to 843 MB). Existing generation packs are reused.',flush=True)
   output=pack_root/'faceswap';output.mkdir(parents=True,exist_ok=True)
   for a in assets:
    if a[0]!='insightface':continue
    dest=output/a[1]
    if dest.exists():continue
    for candidate in present(a):
     if builder['file_sha'](candidate)==a[3]:
      if shutil.disk_usage(output).free<a[4]+512*1024**2:raise RuntimeError('Insufficient free disk for the saved face pack.')
      temporary=dest.with_name(dest.name+'.partial');shutil.copyfile(candidate,temporary);temporary.replace(dest);break
   builder['build_pack']('faceswap',output_root=pack_root,input_root=input_root)
  if save_face_dataset:
   try:builder['save_private_dataset'](pack_root/'faceswap')
   except (Exception,SystemExit) as error:
    print('Face models are ready for this session, but private Dataset submission did not complete ('+type(error).__name__+').',flush=True)
    print('Verified files remain at '+str(pack_root/'faceswap')+'. Save Version with outputs or retry the model-pack builder to keep them for future sessions.',flush=True)
  else:print('Face pack prepared. Save Version with outputs to preserve it.',flush=True)
 # Size discovery is only preflight. The renderer still verifies model SHA256 before use.
 return {a[1]:present(a) for a in assets}
