"""Actual tiny Real-ESRGAN video inference + pause/resume/automatic queue export.
Requires a CPU/GPU ComfyUI server, verified RealESRGAN weights and browser_check output.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

URL=os.environ.get('COMFY_URL','http://127.0.0.1:8188')
ROOT=Path(os.environ.get('MOVIE_INTEGRATION_ROOT','/tmp/movie-integration'))
ROOT.mkdir(parents=True,exist_ok=True)
api=json.loads(Path(os.environ.get('MOVIE_CHECK_OUTPUT','/tmp/movie-api-graphs.json')).read_text())['Movie Studio - RealESRGAN']
upload=next(k for k,n in api.items() if n['class_type']=='MovieStudioUpload')
source=Path(api[upload]['inputs']['source_path'])
assert source.is_file(),'Run browser_check.py to upload the real fixture first.'
loader=next(k for k,n in api.items() if n['class_type']=='MovieStudioChunk')
save=next(k for k,n in api.items() if n['class_type']=='MovieStudioCommit')
api[loader]['inputs'].update(chunk_frames=2,context_frames=0)
api[save]['inputs']['export_format']='high_quality_mp4'
api[save]['inputs']['auto_continue']=False

def request(path,data=None):
    req=urllib.request.Request(URL+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)

def queue(prompt):
    r=request('/prompt',{'prompt':prompt})
    assert 'prompt_id' in r and not r.get('node_errors'),r
    return r['prompt_id']

def wait(pid):
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        h=request('/history/'+pid)
        if pid in h:
            result=h[pid]
            assert result['status']['status_str']=='success',result
            return result
        time.sleep(0.5)
    raise TimeoutError('Actual Real-ESRGAN tiny inference did not complete in 180 seconds')

first=wait(queue(api))
text=first['outputs'][save]['text'][0]
assert 'PAUSED' in text,text
job=Path(text.rsplit('. ',1)[1])
assert (job/'chunk-000000.json').exists() and not (job/'chunk-000001.json').exists()
original=(job/'chunk-000000.mkv').stat().st_mtime_ns
# auto_continue changes must reuse existing restored frames.
api[save]['inputs']['auto_continue']=True
second=wait(queue(api))
assert (job/'chunk-000000.mkv').stat().st_mtime_ns==original
target=job/'upscaled-h264-crf14-slow.mp4'
assert target.exists(),second
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(target)]))
video=next(x for x in info['streams'] if x['codec_type']=='video')
assert (video['width'],video['height'],int(video['nb_read_frames']))==(48,32,3),video
assert sum(x['codec_type']=='audio' for x in info['streams'])==1
print('Actual Real-ESRGAN inference, pause/resume and MP4+audio export passed:',target)
# A completed Run reuses frames and changing only export format re-exports a lossless master.
api[save]['inputs']['export_format']='lossless_mkv'
third=wait(queue(api))
assert (job/'upscaled-lossless-rgb.mkv').exists(),third
assert (job/'chunk-000000.mkv').stat().st_mtime_ns==original
print('Completed-job lossless re-export without upscaling again passed.')
# A new graph setting creates a new job and automatically queues three chunks.
full=copy.deepcopy(api)
full[loader]['inputs']['chunk_frames']=1
pid=queue(full)
wait(pid)
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    history=request('/history')
    candidates=[x for x in history.values() if any('COMPLETE:' in str(y.get('text','')) for y in x.get('outputs',{}).values())]
    complete=[x for x in candidates if x['prompt'][2][loader]['inputs']['chunk_frames']==1 and x['prompt'][2].get(upload,{}).get('inputs',{}).get('source_path')==str(source)]
    if complete:
        assert complete[-1]['status']['status_str']=='success',complete[-1]
        print('Automatic native ComfyUI continuation across three chunks passed.')
        break
    q=request('/queue')
    if not q['queue_running'] and not q['queue_pending']:
        raise RuntimeError(f'Continuation stopped without completion: {history}')
    time.sleep(0.5)
else:raise TimeoutError('Automatic continuation timed out')
# One source-scale edit must also update the native resize connection for 4×.
four=copy.deepcopy(api)
four[loader]['inputs'].update(scale=4,chunk_frames=3)
four[save]['inputs'].update(export_format='lossless_mkv',auto_continue=False)
four_result=wait(queue(four))
four_path=Path(four_result['outputs'][save]['text'][0].removeprefix('COMPLETE: '))
four_info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(four_path)]))
four_video=next(x for x in four_info['streams'] if x['codec_type']=='video')
assert (four_video['width'],four_video['height'],int(four_video['nb_read_frames']))==(96,64,3),four_video
assert sum(x['codec_type']=='audio' for x in four_info['streams'])==1
print('Automatic 2× / 4× target synchronization and 4× lossless export passed.')
