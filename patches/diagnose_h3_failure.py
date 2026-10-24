# Paste into the existing Kaggle notebook. Reads history/logs only; no rendering,
# installation, model downloads, server restart, or GPU computation.
import json
from pathlib import Path
failure=globals().get('entry')
if not isinstance(failure,dict) or not failure.get('status'):
    if 'api_request' not in globals() or 'job' not in globals():
        raise RuntimeError('Run in the same notebook kernel as the failed job, or retrieve comfyui.log from its working folder.')
    failure=api_request('/history/'+job).get(job)
if not failure:raise RuntimeError('Failed job history is unavailable; retrieve comfyui.log before ending the session.')
errors=[]
for kind,detail in failure.get('status',{}).get('messages',[]):
    if kind=='execution_error':
        errors.append({k:detail.get(k) for k in ('node_id','node_type','exception_type','exception_message','traceback')})
print(json.dumps(errors,indent=2))
root=Path(globals().get('ROOT','/kaggle/working/minimax_h3_dual_t4'))
(root/'last-failed-history.json').write_text(json.dumps(failure,indent=2))
print('Full history saved:',root/'last-failed-history.json')
if not errors:
    log=root/'comfyui.log'
    if log.is_file():print(log.read_text(errors='replace')[-12000:])
