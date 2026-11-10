# CPU-only model pack builder. Run once and save the outputs on Kaggle.
PACK_BUILDER_VERSION = 'dataset-upload-v3-direct-create'
ALL_MODELS_STORAGE_VERSION = 'single-model-cache-v2'
AUTO_CREATE_DATASET = True  # Uses your existing Kaggle authentication to create a private Dataset.
DATASET_OWNER = ''  # Usually detected from authenticated Kaggle API credentials.
PACK = 'all'  # All eight reviewed models in one private Dataset (~20.53 GB / 19.12 GiB)
from pathlib import Path
import os, json, hashlib, shutil, time, urllib.request, urllib.error, fcntl
CATALOG = {'profile': 'commercial-v7', 'reviewed': '2026-11-09', 'assets': [{'pack': 'create_edit', 'folder': 'diffusion_models', 'name': 'flux-2-klein-4b.safetensors', 'bytes': 7751105712, 'sha256': 'ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343', 'license': 'Apache-2.0', 'source': 'https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/5b3abca5c0dc1ecd4eaf3340868d09063fc1c965/split_files/diffusion_models/flux-2-klein-4b.safetensors'}, {'pack': 'create_edit', 'folder': 'text_encoders', 'name': 'qwen_3_4b.safetensors', 'bytes': 8044982048, 'sha256': '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a', 'license': 'Apache-2.0', 'source': 'https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/5b3abca5c0dc1ecd4eaf3340868d09063fc1c965/split_files/text_encoders/qwen_3_4b.safetensors'}, {'pack': 'create_edit', 'folder': 'vae', 'name': 'flux2-vae.safetensors', 'bytes': 336211292, 'sha256': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3', 'license': 'Apache-2.0', 'source': 'https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/5b3abca5c0dc1ecd4eaf3340868d09063fc1c965/split_files/vae/flux2-vae.safetensors'}, {'pack': 'finish', 'folder': 'upscale_models', 'name': 'RealESRGAN_x4plus.pth', 'bytes': 67040989, 'sha256': '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1', 'license': 'BSD-3-Clause', 'source': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'}, {'pack': 'finish', 'folder': 'diffusion_models', 'name': 'seedvr2_3b_int8_convrot.safetensors', 'bytes': 3458259704, 'sha256': 'c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157', 'license': 'Apache-2.0', 'source': 'https://huggingface.co/Comfy-Org/SeedVR2/resolve/df48879708206a403d2a61acd55578c2e80fd233/diffusion_models/seedvr2_3b_int8_convrot.safetensors'}, {'pack': 'finish', 'folder': 'vae', 'name': 'seedvr2_ema_vae_fp16.safetensors', 'bytes': 501324814, 'sha256': '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1', 'license': 'Apache-2.0', 'source': 'https://huggingface.co/Comfy-Org/SeedVR2/resolve/df48879708206a403d2a61acd55578c2e80fd233/vae/seedvr2_ema_vae_fp16.safetensors'}, {'pack': 'finish', 'folder': 'mediapipe', 'name': 'face_landmarker.task', 'bytes': 3758596, 'sha256': '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff', 'license': 'Apache-2.0', 'source': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'}, {'pack': 'finish', 'folder': 'loras', 'name': 'bfs_head_v1_flux-klein_4b.safetensors', 'bytes': 369644384, 'sha256': '77aabcd6144612dd4873080b99283dd9a27977712548bee161894f62c91ff09c', 'license': 'MIT', 'source': 'https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/2209ae537ac5e883d6ec7db5b7bd6dccf1a0eba7/bfs_head_v1_flux-klein_4b.safetensors'}]}
LICENSE_FILES = {}
if not LICENSE_FILES and '__file__' in globals():
 _license_root=Path(__file__).with_name('licenses')
 if _license_root.is_dir():LICENSE_FILES={p.name:p.read_text() for p in _license_root.iterdir() if p.is_file()}

def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def selected_assets(pack):
 if pack not in ('all','create_edit','finish'):raise ValueError('Use all, create_edit or finish')
 return [(a['folder'],a['name'],a['source'],a['sha256'],a['bytes']) for a in CATALOG['assets'] if pack=='all' or a['pack']==pack]

def validate_manifest(manifest):
 if manifest.get('profile')!='commercial-v7' or not manifest.get('complete'):raise RuntimeError('Not a complete commercial-v7 pack')
 expected=selected_assets(manifest.get('pack'))
 records=manifest.get('files',[])
 wanted={(n,z,d) for _,n,_,d,z in expected}
 actual={(a.get('path'),a.get('bytes'),a.get('sha256')) for a in records}
 if wanted!=actual or len(records)!=len(wanted):raise RuntimeError('Manifest is not the approved commercial model allowlist')
 return True

def reject_extra_models(root,pack):
 allowed={a[1] for a in selected_assets(pack)}
 for p in Path(root).rglob('*'):
  if p.suffix.lower() in ('.onnx','.zip','.pth','.pt','.safetensors','.task','.bin') and (p.parent!=Path(root) or p.name not in allowed):raise RuntimeError('Unapproved extra model in pack: '+p.name)

def transfer(url,target,digest,size):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if target.stat().st_size==size and file_sha(target)==digest:
   print('Verified; reused:',target.name,flush=True);return
  raise RuntimeError('Existing file failed verification; preserve or move it before retrying: '+str(target))
 partial=target.with_name(target.name+'.partial')
 for attempt in range(3):
  offset=partial.stat().st_size if partial.exists() else 0
  if offset>=size:
   if offset==size and file_sha(partial)==digest:partial.replace(target);return
   partial.unlink();offset=0
  if shutil.disk_usage(target.parent).free<size-offset+512*1024**2:raise RuntimeError('Insufficient free disk for '+target.name)
  req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
  try:
   with urllib.request.urlopen(req,timeout=120) as response:
    resume=offset>0 and response.status==206
    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Unexpected range response')
    written=offset if resume else 0;reported=written
    with open(partial,'ab' if resume else 'wb') as f:
     while True:
      data=response.read(8*1024*1024)
      if not data:break
      f.write(data);written+=len(data)
      if written-reported>=128*1024**2:
       print(f'{target.name}: {written/size:.0%}',flush=True);reported=written
   if partial.stat().st_size!=size or file_sha(partial)!=digest:raise RuntimeError('Incomplete download or checksum mismatch')
   partial.replace(target);return
  except (OSError,urllib.error.URLError,RuntimeError) as e:
   if attempt==2:raise RuntimeError('Download failed for '+target.name+'; rerun to retry its partial file.') from e
   print('Retrying',target.name,type(e).__name__,flush=True);time.sleep(2)

def build_pack(pack,output_root=Path('/kaggle/working/image-studio-model-packs'),input_root=Path('/kaggle/input')):
 assets=selected_assets(pack)
 root=Path(output_root)/pack;root.mkdir(parents=True,exist_ok=True)
 lock=open(root/'.builder.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();raise RuntimeError('This pack is already being built.')
 try:
  manifest=root/'manifest.json'
  # An interrupted rebuild must never leave a completed manifest behind.
  if manifest.exists():manifest.replace(root/'manifest.previous.json')
  print('Building',pack,'on CPU; model bytes:',sum(a[4] for a in assets),flush=True)
  reject_extra_models(root,pack)
  required=sum(size for _,name,_,_,size in assets if not (root/name).exists())
  if shutil.disk_usage(root).free<required+512*1024**2:raise RuntimeError('Insufficient disk for the complete '+pack+' pack plus 512 MiB reserve. Existing files are retained; no downloads started.')
  records=[]
  for folder,name,url,digest,size in assets:
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
   # Saved packs must contain real files: input symlinks would not be a durable output.
   if not target.exists() and Path(input_root).exists():
    for candidate in Path(input_root).rglob(name):
     if candidate.is_file() and candidate.stat().st_size==size and file_sha(candidate)==digest:
      if shutil.disk_usage(target.parent).free<size+512*1024**2:raise RuntimeError('Insufficient disk to copy saved model')
      tmp=target.with_name(name+'.partial');shutil.copyfile(candidate,tmp);tmp.replace(target);print('Copied saved input:',name,flush=True);break
   transfer(url,target,digest,size)
   if target.is_symlink() or target.stat().st_size!=size or file_sha(target)!=digest:raise RuntimeError('Pack verification failed: '+name)
   records.append({'path':target.relative_to(root).as_posix(),'bytes':size,'sha256':digest,'source':url})
  data={'schema':2,'profile':'commercial-v7','pack':pack,'complete':True,'model_bytes':sum(a[4] for a in assets),'files':records}
  validate_manifest(data)
  data['license_catalog']=CATALOG
  for name,content in LICENSE_FILES.items():(root/name).write_text(content)
  tmp=manifest.with_suffix('.json.partial');tmp.write_text(json.dumps(data,indent=2));tmp.replace(manifest)
  print('\nPACK COMPLETE:',root,flush=True)
  print('Save Version with outputs. In Image Studio Commercial v7: Add Input → Notebook Output Files → select this saved version.',flush=True)
  print('No GPU or image generation was used. Keep model packs private and follow model licenses.',flush=True)
  return data
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def dataset_identity(owner,manifest):
 import re
 if not isinstance(owner,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',owner):raise ValueError('Set DATASET_OWNER to your Kaggle username or use configured Kaggle credentials.')
 fingerprint=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()[:12]
 # Local pack names may have underscores; Kaggle dataset URL slugs may not.
 pack_slug=re.sub(r'[^a-z0-9]+','-',manifest['pack'].lower()).strip('-')
 slug='image-studio-commercial-'+pack_slug+'-'+fingerprint
 if not 6<=len(slug)<=50 or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid Kaggle Dataset slug; no files were uploaded.')
 return owner+'/'+slug,fingerprint

def kaggle_error_text(error):
 """Keep the actionable reason without exposing credentials or response headers."""
 import re
 message=str(error)
 response=getattr(error,'response',None)
 if response is not None:
  try:
   data=response.json()
   if isinstance(data,dict):message=str(data.get('message') or data.get('error') or data.get('detail') or message)
  except (ValueError,TypeError):pass
 for key in ('KAGGLE_API_TOKEN','KAGGLE_KEY'):
  value=os.environ.get(key)
  if value:message=message.replace(value,'[REDACTED]')
 message=re.sub(r'(?i)Bearer\s+[^\s,;]+','Bearer [REDACTED]',message)
 message=re.sub(r'(?i)((?:api[_-]?key|access_token|token|authorization)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',message)
 return message[:1800]

def authenticated_api():
 # Kaggle API uses private notebook credentials. Never print or store their values.
 if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
  try:
   from kaggle_secrets import UserSecretsClient
   os.environ['KAGGLE_API_TOKEN']=UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
  except Exception:pass
 if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
  try:
   from kaggle_secrets import UserSecretsClient
   secrets=UserSecretsClient()
   os.environ['KAGGLE_USERNAME']=secrets.get_secret('KAGGLE_USERNAME')
   os.environ['KAGGLE_KEY']=secrets.get_secret('KAGGLE_KEY')
  except Exception:pass
 from kaggle.api.kaggle_api_extended import KaggleApi
 api=KaggleApi();api.authenticate()
 return api

def save_private_dataset(pack_root,owner='',api=None):
 root=Path(pack_root);manifest=json.loads((root/'manifest.json').read_text())
 validate_manifest(manifest)
 reject_extra_models(root,manifest['pack'])
 for name in LICENSE_FILES:
  if not (root/name).is_file() or (root/name).read_text()!=LICENSE_FILES[name]:raise RuntimeError('Missing/changed license notice: '+name)
 for record in manifest['files']:
  p=root/record['path']
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:raise RuntimeError('Model verification failed before Dataset upload: '+p.name)
 if api is None:api=authenticated_api()
 owner=owner or api.config_values.get('username','')
 handle,fingerprint=dataset_identity(owner,manifest)
 url='https://www.kaggle.com/datasets/'+handle
 receipt=root/'dataset-upload.json'
 if receipt.exists():
  previous=json.loads(receipt.read_text())
  if previous.get('handle')==handle and previous.get('submitted') and previous.get('manifest_fingerprint')==fingerprint:
   print('Previously submitted Dataset:',url,flush=True);return url
 # Match the previously working builder: call create_new directly.
 # Do not gate creation on our own dataset_status/datasets.get request.
 # Kaggle can deny lookup on a not-yet-created/private Dataset; that does not
 # establish that creation is forbidden. create_new never versions/overwrites.
 metadata={'title':'Image Studio '+manifest['pack']+' '+fingerprint,'id':handle,'licenses':[{'name':'other'}],'description':'Private verified model pack for Image Studio. Original model licenses apply; no relicensing. See manifest.json for model sources and SHA256 values.'}
 (root/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
 # Stage hardlinks to the exact pack files, avoiding an extra multi-GB copy.
 # Do not upload builder locks, old manifests, receipts or failure diagnostics.
 import tempfile
 with tempfile.TemporaryDirectory(prefix='.studio-upload-',dir=root.parent) as tmp:
  stage=Path(tmp)
  for name in [r['path'] for r in manifest['files']]+list(LICENSE_FILES)+['manifest.json','dataset-metadata.json']:
   os.link(root/name,stage/name)
  result=api.dataset_create_new(str(stage),public=False,quiet=False,convert_to_csv=False,dir_mode='skip')
 field=lambda key:result.get(key) if isinstance(result,dict) else getattr(result,key,None)
 error=field('error') or field('error_message')
 status=field('status')
 if error or (status and str(status).lower() in ('error','failed','failure')):
  raise RuntimeError('Kaggle rejected Dataset creation: '+kaggle_error_text(error or status)+'. Verified models are retained; no model downloads are needed.')
 if result is None:raise RuntimeError('Kaggle returned no creation response. Verified models are retained. Check the Dataset page before retrying.')
 receipt.write_text(json.dumps({'handle':handle,'url':url,'submitted':True,'requested_private':True,'manifest_fingerprint':fingerprint},indent=2))
 print('PRIVATE DATASET SUBMITTED:',url,flush=True)
 print('Wait for Kaggle processing, then Image Studio Commercial v7 → Add Input → Your Datasets. No PC download needed.',flush=True)
 return url


def publish_pack(pack_root,owner='',api=None):
 try:return save_private_dataset(pack_root,owner,api)
 except (Exception,SystemExit) as error:
  detail=kaggle_error_text(error)
  root=Path(pack_root)
  (root/'dataset-error.json').write_text(json.dumps({'stage':'dataset-publication','reason':detail,'models_retained':True},indent=2))
  raise RuntimeError('Dataset publication failed: '+detail+' Save Version with outputs before closing this session. Rerunning reuses verified weights. Diagnostic: '+str(root/'dataset-error.json')) from None

def create_complete_dataset(output_root=Path('/kaggle/working/image-studio-model-packs'),input_root=Path('/kaggle/input'),owner='',api=None):
 """Use one native create_new request; feed its uploader one verified file at a time."""
 import inspect,tempfile
 root=Path(output_root)/'all';root.mkdir(parents=True,exist_ok=True)
 lock=open(root/'.builder.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();raise RuntimeError('The ALL MODELS Dataset creator is already running.')
 original_upload=None
 try:
  assets=selected_assets('all')
  records=[dict(path=n,bytes=z,sha256=d,source=u) for _,n,u,d,z in assets]
  manifest=dict(schema=2,profile='commercial-v7',pack='all',complete=True,model_bytes=sum(a[4] for a in assets),files=records,license_catalog=CATALOG)
  validate_manifest(manifest)
  if api is None:api=authenticated_api()
  handle,fingerprint=dataset_identity(owner or api.config_values.get('username',''),manifest)
  url='https://www.kaggle.com/datasets/'+handle;receipt=root/'dataset-upload.json'
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if old.get('handle')==handle and old.get('submitted') and old.get('manifest_fingerprint')==fingerprint:
    print('Previously submitted ALL MODELS Dataset:',url,flush=True);return url
  # Locate verified sources before reserving disk. Existing source files are never
  # copied, moved or removed. Only this creator's missing-file cache is released.
  print('Finding verified saved models; existing sources will not be copied or deleted.',flush=True)
  sources={}
  for _,name,_,digest,size in assets:
   for search in [Path(input_root),Path(output_root)]:
    if not search.exists():continue
    found=next((q for q in search.rglob(name) if q.is_file() and q.stat().st_size==size and file_sha(q)==digest),None)
    if found:
     sources[name]=found.resolve();print('Reusing existing model:',name,flush=True);break
  cache=Path(output_root)/'.all-models-download-cache'
  if cache.is_symlink():raise RuntimeError('Temporary download cache must not be a symlink.')
  cache.mkdir(exist_ok=True)
  missing=[a for a in assets if a[1] not in sources]
  peak=max((a[4] for a in missing),default=0)
  free=shutil.disk_usage(cache).free
  print('ONE ALL MODELS Dataset:',len(assets),'models;',sum(a[4] for a in assets),'bytes total',flush=True)
  print('Verified existing sources:',len(sources),'; largest missing download:',peak,'bytes; free disk:',free,flush=True)
  if free<peak+(512*1024**2 if peak else 16*1024**2):raise RuntimeError('Need room for only the largest missing model ('+str(peak)+' bytes) plus 512 MiB reserve; free '+str(free)+'. Existing model files were not changed.')
  original_upload=api.upload_files;signature=inspect.signature(original_upload)
  if 'folder' not in signature.parameters:raise RuntimeError('Kaggle uploader has an unsupported folder argument; no downloads started.')
  metadata=dict(title='Image Studio all '+fingerprint,id=handle,licenses=[{'name':'other'}],description='Complete private Image Studio model pack. Original component licenses apply; see manifest.json and included license notices.')
  state=root/'upload-progress.json'
  upload_finished=False
  def stream_upload(*args,**kwargs):
   nonlocal upload_finished
   # dataset_create_new owns its normal request and upload context. Delegate all
   # transfers to its unmodified uploader, changing only each input folder.
   def send(folder):
    bound=signature.bind(*args,**kwargs);bound.arguments['folder']=str(folder)
    return original_upload(*bound.args,**bound.kwargs)
   uploaded=[]
   request=signature.bind(*args,**kwargs).arguments['request']
   for _,name,source_url,digest,size in sorted(assets,key=lambda a:a[1] not in sources):
    source=sources.get(name,cache/name)
    owned=source.parent.resolve()==cache.resolve()
    if owned:transfer(source_url,source,digest,size)
    if not source.is_file() or source.stat().st_size!=size or file_sha(source)!=digest:raise RuntimeError('Model failed upload verification: '+name)
    with tempfile.TemporaryDirectory(prefix='.single-model-',dir=root) as temp:
     # Kaggle's native uploader uses isfile/open, which follow this absolute link.
     # Uploaded Dataset files contain actual bytes, not the local link.
     (Path(temp)/name).symlink_to(source.resolve())
     print('Uploading verified model:',name,flush=True)
     count=len(request.files);send(temp)
     if len(request.files)!=count+1:raise RuntimeError('Kaggle uploader did not confirm the file token: '+name)
    uploaded.append(name);state.write_text(json.dumps({'handle':handle,'uploaded_models':uploaded,'dataset_created':False},indent=2))
    if owned:
     # Only newly downloaded cache files, after a successful native file upload.
     source.unlink();print('Released temporary download:',name,flush=True)
   with tempfile.TemporaryDirectory(prefix='.pack-notices-',dir=root) as temp:
    folder=Path(temp);(folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name,content in LICENSE_FILES.items():(folder/name).write_text(content)
    send(folder)
   upload_finished=True
  api.upload_files=stream_upload
  with tempfile.TemporaryDirectory(prefix='.dataset-request-',dir=root) as temp:
   (Path(temp)/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
   result=api.dataset_create_new(temp,public=False,quiet=False,convert_to_csv=False,dir_mode='skip')
  field=lambda k:result.get(k) if isinstance(result,dict) else getattr(result,k,None)
  error=field('error') or field('error_message');status=field('status')
  if result is None or error or str(status).lower() in ('error','failed','failure'):raise RuntimeError('Kaggle Dataset creation was not confirmed: '+kaggle_error_text(error or status or 'empty response'))
  if not upload_finished:raise RuntimeError('Kaggle did not run the complete model upload; no success receipt was written.')
  receipt.write_text(json.dumps(dict(handle=handle,url=url,submitted=True,requested_private=True,manifest_fingerprint=fingerprint,upload_mode='one-model-at-a-time'),indent=2))
  (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
  print('PRIVATE ALL MODELS DATASET SUBMITTED:',url,flush=True)
  print('Wait for processing, then attach this ONE Dataset to Studio v9. No model preparation runs in the studio.',flush=True)
  return url
 except (Exception,SystemExit) as error:
  detail=kaggle_error_text(error)
  (root/'dataset-error.json').write_text(json.dumps({'stage':'single-dataset-creation','reason':detail,'existing_sources_untouched':True,'temporary_uploaded_downloads_may_be_released':True},indent=2))
  raise RuntimeError(detail+' Existing source models are untouched. Dataset creation has not been confirmed; see '+str(root/'dataset-error.json')) from None
 finally:
  if original_upload is not None:api.upload_files=original_upload
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
 if AUTO_CREATE_DATASET:create_complete_dataset(owner=DATASET_OWNER)
 else:build_pack(PACK)
