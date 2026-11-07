# CPU-only model pack builder. Run once and save the outputs on Kaggle.
AUTO_CREATE_DATASET = False  # Optional; default matches your uploaded H3 model-pack workflow.
DATASET_OWNER = ''  # Usually detected from authenticated Kaggle API credentials.
PACK = 'standard'  # 'standard' = SDXL + RealESRGAN + SeedVR2 (~11.1 GB); 'flux3' = optional FLUX3 (~17.75 GB)
from pathlib import Path
import os, json, hashlib, shutil, time, urllib.request, urllib.error, fcntl
ASSET_CATALOG_SOURCE = None

def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def selected_assets(pack):
 if pack=='standard':return model_assets(True,True,False)
 if pack=='flux3':return [a for a in model_assets(False,False,True) if 'black-forest-labs' in a[2]]
 raise ValueError("PACK must be 'standard' or 'flux3'")

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
  records=[]
  for folder,name,url,digest,size in assets:
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
   # Saved packs must contain real files: input symlinks would not be a durable output.
   if not target.exists() and Path(input_root).exists():
    for candidate in Path(input_root).rglob(name):
     if candidate.is_file() and candidate.stat().st_size==size and file_sha(candidate)==digest:
      if shutil.disk_usage(target.parent).free<size+512*1024**2:raise RuntimeError('Insufficient disk to copy saved model')
      tmp=target.with_name(name+'.partial');shutil.copyfile(candidate,tmp);tmp.replace(target);print('Copied saved input:',name,flush=True);break
   if not target.exists() and 'black-forest-labs' in url:
    if shutil.disk_usage(target.parent).free<size+512*1024**2:raise RuntimeError('Insufficient free disk for FLUX3 asset')
    try:
     from kaggle_secrets import UserSecretsClient
     token=UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:raise RuntimeError('Accept FLUX3 terms on Hugging Face and enable Kaggle Secret HF_TOKEN.') from None
    try:
     from huggingface_hub import hf_hub_download
     saved=hf_hub_download(repo_id='black-forest-labs/flux-3-image-klein-comfy',filename=folder+'/'+name,revision='9d36114cf7a2e03a45c28f5cc4987d20ebfdb6c1',local_dir=root/'models',token=token)
     Path(saved).replace(target)
    finally:del token
   transfer(url,target,digest,size)
   if target.is_symlink() or target.stat().st_size!=size or file_sha(target)!=digest:raise RuntimeError('Pack verification failed: '+name)
   records.append({'path':target.relative_to(root).as_posix(),'bytes':size,'sha256':digest,'source':url})
  data={'schema':1,'pack':pack,'complete':True,'model_bytes':sum(a[4] for a in assets),'files':records}
  tmp=manifest.with_suffix('.json.partial');tmp.write_text(json.dumps(data,indent=2));tmp.replace(manifest)
  print('\nPACK COMPLETE:',root,flush=True)
  print('Save Version with outputs. In Image Studio v3: Add Input → Notebook Output Files → select this saved version.',flush=True)
  print('No GPU or image generation was used. Keep model packs private and follow model licenses.',flush=True)
  return data
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def save_private_dataset(pack_root,owner='',api=None):
 root=Path(pack_root);manifest=json.loads((root/'manifest.json').read_text())
 if not manifest.get('complete'):raise RuntimeError('Refusing to upload an incomplete pack.')
 for record in manifest['files']:
  p=root/record['path']
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:raise RuntimeError('Model verification failed before Dataset upload: '+p.name)
 if api is None:
  # Kaggle API uses private notebook credentials. Never print or store their values.
  if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
   try:
    from kaggle_secrets import UserSecretsClient
    os.environ['KAGGLE_API_TOKEN']=UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
   except Exception:pass
  from kaggle.api.kaggle_api_extended import KaggleApi
  api=KaggleApi();api.authenticate()
 owner=owner or api.config_values.get('username','')
 import re
 if not re.fullmatch(r'[A-Za-z0-9_-]+',owner):raise RuntimeError('Set DATASET_OWNER to your Kaggle username or configure Kaggle authentication privately.')
 fingerprint=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()[:12]
 slug='image-studio-'+manifest['pack']+'-'+fingerprint
 handle=owner+'/'+slug;url='https://www.kaggle.com/datasets/'+handle
 receipt=root/'dataset-upload.json'
 if receipt.exists() and json.loads(receipt.read_text()).get('handle')==handle:
  print('Previously submitted Dataset:',url,flush=True);return url
 metadata={'title':'Image Studio '+manifest['pack']+' '+fingerprint,'id':handle,'licenses':[{'name':'other'}],'description':'Private verified model pack for Image Studio. Original model licenses apply; no relicensing. See manifest.json for model sources and SHA256 values.'}
 (root/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
 # Files are flat: never zip many GiB or omit nested weights by accident.
 # create_new(public=False) will not modify an existing dataset or its visibility.
 result=api.dataset_create_new(str(root),public=False,quiet=False,convert_to_csv=False,dir_mode='skip')
 error=result.get('error') if isinstance(result,dict) else getattr(result,'error',None)
 if error:raise RuntimeError('Dataset creation was not confirmed: '+str(error)+'. Model files remain saved; existing datasets were not changed.')
 receipt.write_text(json.dumps({'handle':handle,'url':url,'submitted':True,'requested_private':True,'manifest_fingerprint':fingerprint},indent=2))
 print('PRIVATE DATASET SUBMITTED:',url,flush=True)
 print('Wait for Kaggle processing, then Image Studio v3 → Add Input → Your Datasets. No PC download needed.',flush=True)
 return url

if __name__=='__main__':
 build_pack(PACK)
 if AUTO_CREATE_DATASET:
  try:save_private_dataset(Path('/kaggle/working/image-studio-model-packs')/PACK,DATASET_OWNER)
  except (Exception,SystemExit):
   print('Automatic Dataset upload did not complete. The verified pack is retained in notebook outputs. Check Kaggle API authentication privately (KAGGLE_API_TOKEN secret), set DATASET_OWNER if needed, then rerun. No credentials are printed.',flush=True)
   raise

