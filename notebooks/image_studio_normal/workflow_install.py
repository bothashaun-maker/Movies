"""Replace app-managed menus while preserving old canvases outside the menu."""
from pathlib import Path
import json,hashlib,shutil

def install_workflows(directory,payload,archive):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 expected={'Image Studio - '+name+'.json' for name in payload}
 for old in directory.glob('Image Studio - *.json'):
  # Preserve all prior managed canvases; replace only this application's files.
  blob=old.read_bytes();dest=Path(archive)/(old.stem+'-'+hashlib.sha256(blob).hexdigest()[:10]+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copyfile(old,dest)
  if old.name not in expected:old.unlink()
 for name,wf in payload.items():(directory/('Image Studio - '+name+'.json')).write_text(json.dumps(wf,indent=2))
