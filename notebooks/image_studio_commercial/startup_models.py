"""Find and verify saved model weights. Never download or publish during startup."""
from pathlib import Path
import hashlib

def prepare_models(assets,roots):
 def digest(path):
  h=hashlib.sha256()
  with open(path,'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  return h.hexdigest()
 resolved={}
 for asset in assets:
  for root in map(Path,roots):
   if not root.exists():continue
   found=next((p for p in root.rglob(asset['name']) if p.is_file() and p.stat().st_size==asset['bytes'] and digest(p)==asset['sha256']),None)
   if found:resolved[asset['name']]=found;break
 missing=[a['name'] for a in assets if a['name'] not in resolved]
 if missing:raise RuntimeError('Attach your saved ALL MODELS Dataset (or its saved notebook output). Missing: '+', '.join(missing)+'. No model downloads or Dataset uploads were started. Build the all pack once on CPU with Image_Studio_All_Models_Dataset.ipynb.')
 return resolved
