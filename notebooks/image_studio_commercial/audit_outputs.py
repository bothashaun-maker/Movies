"""Read-only PNG/JSON metadata triage. Never treats missing evidence as clearance."""
import argparse,json
from pathlib import Path
from PIL import Image

def audit(path):
 try:
  if path.suffix.lower()=='.png':
   with Image.open(path) as im:metadata=dict(im.info)
  else:metadata=json.loads(path.read_text())
  text=json.dumps(metadata).lower();hits=[]
  for token in ['inswapper','buffalo_l','studiofaceswap','studiofacedetect']:
   if token in text:hits.append(token)
  if hits:status='restricted-face-pipeline-review'
  elif any(t in text for t in ['juggernaut','flux-3','flux3','flux.2-klein-9b']):status='model-terms-review'
  elif 'commercial-v7' in text:status='commercial-profile-recorded-still-review-input-rights'
  else:status='unknown-or-incomplete-provenance'
  return dict(path=str(path),status=status,hits=hits)
 except Exception as e:return dict(path=str(path),status='unreadable',error=type(e).__name__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 paths=[x for x in a.root.rglob('*') if x.suffix.lower() in ('.png','.json') and x.resolve()!=a.report.resolve()]
 report={'notice':'Metadata triage, not a legal verdict. Preserve originals. A clean-looking graph may omit upstream processing.','files':[audit(x) for x in paths]}
 a.report.write_text(json.dumps(report,indent=2));print('Read',len(paths),'files. Report:',a.report)
