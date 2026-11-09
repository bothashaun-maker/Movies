# Single-cell startup. No image generation starts automatically.
from pathlib import Path
import sys, os, subprocess, json, time, urllib.request, hashlib, shutil, re, signal, fcntl
# Turn off SDXL to run only the upscalers without its 7.1 GB checkpoint.
LOAD_SDXL = True
ENABLE_SEEDVR2 = True
ENABLE_FACESWAP = True
AUTO_PREPARE_FACE_PACK = True
AUTO_SAVE_PREPARED_FACE_PACK = True
PACK_BUILDER_SOURCE = None
STARTUP_MODELS_SOURCE = None
FACE_NODE_SOURCE = None
JJP_NODE_SOURCE = None
JJP_PRESETS_SOURCE = None
# Optional gated model: accept its terms on Hugging Face and add Kaggle Secret HF_TOKEN.
ENABLE_FLUX3 = False
REQUIRE_ATTACHED_MODELS = True
# The builder embeds the shared asset catalog here.
ASSET_CATALOG_SOURCE = None
WORKFLOW_PAYLOAD = {}
COMFY_REV='1332fabfcc507aa7c548cd5152071ed494c207e8'
MODEL='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'
MODEL_SHA='c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca'
MODEL_SIZE=7105348188
CF_SHA='1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872'
# Renderer dependencies are unchanged; reuse a retained v4 installation.
BASE=Path('/kaggle/temp/image-studio-v4'); BASE.mkdir(parents=True,exist_ok=True)
OUT=Path('/kaggle/working/image-studio'); OUT.mkdir(parents=True,exist_ok=True)
lock=open(BASE/'startup.lock','w')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise RuntimeError('Image Studio is already running. Stop its startup cell before starting again.')
children=[]
# Fail before downloading if the previous studio still owns this port.
import socket
probe=socket.socket()
try: probe.bind(('127.0.0.1',8188))
except OSError: raise RuntimeError('Port 8188 is occupied. Stop the earlier studio cell before starting v6.')
finally: probe.close()

def run(args,**kw):subprocess.run(list(map(str,args)),check=True,**kw)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def download(url,path,expected,size=None):
 if path.exists() and sha(path)==expected:return
 if path.exists():raise RuntimeError(f'Checksum mismatch: {path}. Preserve or move this file before retrying.')
 if size and shutil.disk_usage(path.parent).free<size+2*1024**3:raise RuntimeError('Insufficient free disk for model plus 2 GiB reserve.')
 temp=path.with_suffix(path.suffix+'.partial')
 print('Downloading',path.name,flush=True)
 with urllib.request.urlopen(url,timeout=120) as response,open(temp,'wb') as target:shutil.copyfileobj(response,target,8*1024*1024)
 if sha(temp)!=expected:raise RuntimeError(f'Download checksum mismatch: {temp}')
 temp.replace(path)
def spawn(args,log):
 f=open(log,'w');p=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();children.append(p);return p
# Prepare only missing face models; keep generation packs attached and untouched.
try:
 resolved_models=prepare_startup_models(
  model_assets(LOAD_SDXL,ENABLE_SEEDVR2,ENABLE_FLUX3,ENABLE_FACESWAP),PACK_BUILDER_SOURCE,
  require_attached=REQUIRE_ATTACHED_MODELS,prepare_faces=AUTO_PREPARE_FACE_PACK,
  save_face_dataset=AUTO_SAVE_PREPARED_FACE_PACK)
except BaseException:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();raise
try:
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Select a Kaggle GPU accelerator and restart this cell.')
 print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,flush=True)
 free,total=torch.cuda.mem_get_info(0)
 if free<10*1024**3:raise RuntimeError('GPU 0 has less than 10 GiB free. Stop other GPU notebooks/processes first.')
 # Separate T4s do not form one shared VRAM pool. This studio uses GPU 0.
 comfy=BASE/'ComfyUI'
 if not comfy.exists():
  run(['git','clone','--no-checkout','https://github.com/comfyanonymous/ComfyUI.git',comfy])
  run(['git','-C',comfy,'checkout','--detach',COMFY_REV])
 if subprocess.check_output(['git','-C',str(comfy),'rev-parse','HEAD'],text=True).strip()!=COMFY_REV:raise RuntimeError('Unexpected ComfyUI revision.')
 venv=BASE/'venv';python=venv/'bin/python'
 if not python.exists():run([sys.executable,'-m','venv','--system-site-packages','--without-pip',venv])
 constraints=BASE/'constraints.txt'
 import importlib.metadata as meta
 protected=['torch','torchvision','torchaudio','numpy','scipy']
 constraints.write_text('\n'.join(f'{p}=={meta.version(p)}' for p in protected))
 requirements=BASE/'requirements.txt'
 requirements.write_text('\n'.join(x for x in (comfy/'requirements.txt').read_text().splitlines() if x.strip() not in protected)+'\ntransformers==5.22.2\nhuggingface-hub==2.4.0\n')
 fingerprint=hashlib.sha256((requirements.read_text()+constraints.read_text()+sys.version).encode()).hexdigest()
 marker=BASE/'setup.complete'
 if not marker.exists() or marker.read_text()!=fingerprint:
  run([sys.executable,'-m','pip','--python',python,'install','-r',requirements,'-c',constraints])
  run([python,'-c','import torch, torchvision, transformers, aiohttp, safetensors; assert torch.cuda.is_available()'])
  marker.write_text(fingerprint)
 if ENABLE_FACESWAP:
  face_marker=BASE/'face-setup.complete'
  if not face_marker.exists() or face_marker.read_text()!=fingerprint:
   env=os.environ.copy()
   if shutil.which('gcc'):env['CC']='gcc'
   if shutil.which('g++'):env['CXX']='g++'
   run([sys.executable,'-m','pip','--python',python,'install','insightface==0.7.3','onnxruntime==1.23.2','-c',constraints],env=env)
   run([python,'-c','import insightface,onnxruntime,cv2'])
   face_marker.write_text(fingerprint)
  custom=comfy/'custom_nodes/image_studio_faces';custom.mkdir(parents=True,exist_ok=True)
  (custom/'__init__.py').write_text(FACE_NODE_SOURCE)
 custom_jjp=comfy/'custom_nodes/jjp_image_production';custom_jjp.mkdir(parents=True,exist_ok=True)
 (custom_jjp/'__init__.py').write_text(JJP_NODE_SOURCE)
 (custom_jjp/'jjp_presets.json').write_text(JJP_PRESETS_SOURCE)
 run([sys.executable,'-m','pip','--python',python,'freeze'],stdout=open(OUT/'packages.txt','w'))
 ultimate=comfy/'custom_nodes/ComfyUI_UltimateSDUpscale'
 if LOAD_SDXL:
  if not ultimate.exists():
   run(['git','clone','--no-checkout','https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git',ultimate])
   run(['git','-C',ultimate,'checkout','--detach','a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df'])
  if subprocess.check_output(['git','-C',str(ultimate),'rev-parse','HEAD'],text=True).strip()!='a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df':raise RuntimeError('Unexpected Ultimate SD Upscale revision.')
  run(['git','-C',ultimate,'submodule','update','--init','--recursive'])
 assets=model_assets(LOAD_SDXL,ENABLE_SEEDVR2,ENABLE_FLUX3,ENABLE_FACESWAP)
 for folder,filename,url,digest,size in assets:
  dest=comfy/'models'/folder/filename;dest.parent.mkdir(parents=True,exist_ok=True)
  verified=False
  if not dest.exists():
   candidates=resolved_models.get(filename,[])
   for candidate in candidates:
    if candidate.is_file() and candidate.stat().st_size==size and sha(candidate)==digest:
     dest.symlink_to(candidate);verified=True;print('Reusing verified model:',candidate,flush=True);break
  if REQUIRE_ATTACHED_MODELS and not dest.exists():
   raise RuntimeError('Missing saved model: '+filename+'. Attach your saved Image Studio Model Packs output via Add Input. No model was downloaded.')
  if ENABLE_FLUX3 and 'black-forest-labs' in url and not dest.exists():
   if shutil.disk_usage(dest.parent).free<size+2*1024**3:raise RuntimeError('Insufficient disk space for selected FLUX3 assets.')
   from kaggle_secrets import UserSecretsClient
   try: token=UserSecretsClient().get_secret('HF_TOKEN')
   except Exception: raise RuntimeError('Enable Kaggle Secret HF_TOKEN with access to the accepted FLUX 3 model terms.') from None
   env=os.environ.copy();env['HF_TOKEN']=token
   try:
    code="import os;from huggingface_hub import hf_hub_download;hf_hub_download(repo_id='black-forest-labs/flux-3-image-klein-comfy',filename="+repr(folder+'/'+filename)+",revision='9d36114cf7a2e03a45c28f5cc4987d20ebfdb6c1',local_dir="+repr(str(comfy/'models'))+",token=os.environ['HF_TOKEN'])"
    run([python,'-c',code],env=env)
   finally:env.pop('HF_TOKEN',None);del token
  if not verified:download(url,dest,digest,size)
 if ENABLE_FACESWAP:
  import zipfile,zlib
  archive=comfy/'models/insightface/buffalo_l.zip'
  folder=comfy/'models/insightface/models/buffalo_l';folder.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(archive) as z:
   allowed={'1k3d68.onnx','2d106det.onnx','det_10g.onnx','genderage.onnx','w600k_r50.onnx'}
   for item in z.infolist():
    if item.is_dir():continue
    if item.filename not in allowed:raise RuntimeError('Unexpected face model archive member: '+item.filename)
    target=folder/item.filename
    valid=target.exists() and target.stat().st_size==item.file_size
    if valid:
     crc=0
     with open(target,'rb') as f:
      for block in iter(lambda:f.read(8*1024*1024),b''):crc=zlib.crc32(block,crc)
     valid=(crc & 0xffffffff)==item.CRC
    if not valid:
     tmp=target.with_suffix('.partial')
     with z.open(item) as src,open(tmp,'wb') as dst:shutil.copyfileobj(src,dst)
     tmp.replace(target)
   if not all((folder/name).is_file() for name in allowed):raise RuntimeError('Incomplete buffalo_l model archive.')
 # Only install workflows for enabled model families.
 def workflow_enabled(wf):
  types={n['type'] for n in wf['nodes']}
  return (LOAD_SDXL or 'CheckpointLoaderSimple' not in types) and (ENABLE_FACESWAP or 'StudioFaceSwap' not in types) and (ENABLE_FLUX3 or 'TextGenerate' not in types) and (ENABLE_SEEDVR2 or 'SeedVR2Conditioning' not in types)
 WORKFLOW_PAYLOAD={k:v for k,v in WORKFLOW_PAYLOAD.items() if workflow_enabled(v)}

 workflow_dir=comfy/'user/default/workflows';workflow_dir.mkdir(parents=True,exist_ok=True)
 for name,wf in WORKFLOW_PAYLOAD.items():(workflow_dir/f'Image Studio - {name}.json').write_text(json.dumps(wf,indent=2))
 # Stable output destination: PNGs retain the actual graph, prompt, seed and settings.
 (OUT/'provenance.json').write_text(json.dumps({'comfy_revision':COMFY_REV,'model':MODEL,'sha256':MODEL_SHA,'assets':[{'file':f,'sha256':d,'bytes':z} for _,f,_,d,z in assets],'torch':torch.__version__},indent=2))
 cf=BASE/'cloudflared'
 download('https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/cloudflared-linux-amd64',cf,CF_SHA)
 cf.chmod(0o700)
 server=spawn([python,comfy/'main.py','--listen','127.0.0.1','--port','8188','--lowvram','--force-fp16','--preview-method','none','--output-directory',OUT/'images','--disable-auto-launch'],OUT/'comfy.log')
 deadline=time.monotonic()+180
 while True:
  if server.poll() is not None:raise RuntimeError('ComfyUI stopped. See comfy.log in Kaggle outputs.')
  try:
   with urllib.request.urlopen('http://127.0.0.1:8188/object_info',timeout=3) as r:info=json.load(r)
   break
  except OSError:
   if time.monotonic()>deadline:raise RuntimeError('ComfyUI startup timed out. See comfy.log.')
   time.sleep(1)
 # Validate installed node classes before offering the browser link.
 for wf in WORKFLOW_PAYLOAD.values():
  for node in wf['nodes']:
   if node['type'] not in info:raise RuntimeError('Required node unavailable: '+node['type'])
 tunnel=spawn([cf,'tunnel','--url','http://127.0.0.1:8188','--no-autoupdate'],OUT/'tunnel.log')
 deadline=time.monotonic()+90
 url=None
 while time.monotonic()<deadline:
  text=(OUT/'tunnel.log').read_text()
  matches=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',text)
  if matches:url=matches[-1];break
  if tunnel.poll() is not None:raise RuntimeError('Cloudflare stopped; inspect tunnel.log.')
  time.sleep(1)
 if not url:raise RuntimeError('No Cloudflare URL received; inspect tunnel.log.')
 print('\nOPEN IMAGE STUDIO:',url,flush=True)
 print('Workflow menu → Image Studio: '+', '.join(WORKFLOW_PAYLOAD)+'. Select a workflow, then press Run.')
 print('No generation has started. Keep this cell running. Anyone with this session URL can access the studio.')
 print('Save Version with outputs before ending Kaggle. Images:',OUT/'images',flush=True)
 while server.poll() is None and tunnel.poll() is None:time.sleep(2)
 raise RuntimeError('Studio or tunnel stopped; see the saved logs.')
finally:
 for p in reversed(children):
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
