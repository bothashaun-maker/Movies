import json,time,urllib.request,os,shutil
from pathlib import Path
from PIL import Image
R=Path(__file__).parent;comfy=Path(os.environ.get('COMFY_ROOT',str(R.parent/'ComfyUI')))
base=os.environ.get('COMFY_URL','http://127.0.0.1:8189')
shutil.copyfile(R.parent/'astronaut.png',comfy/'input/studio-source.png');shutil.copyfile(R.parent/'grace_hopper.jpg',comfy/'input/studio-target.jpg')
prompt={'1':{'class_type':'LoadImage','inputs':{'image':'studio-source.png'}},'2':{'class_type':'LoadImage','inputs':{'image':'studio-target.jpg'}},'3':{'class_type':'StudioFaceDetect','inputs':{'image':['1',0]}},'4':{'class_type':'StudioFaceDetect','inputs':{'image':['2',0]}},'5':{'class_type':'StudioFaceSwap','inputs':{'source_faces':['3',0],'target_faces':['4',0],'source_index':0,'target_index':0}},'6':{'class_type':'SaveImage','inputs':{'images':['5',0],'filename_prefix':'studio-face-integration'}}}
req=urllib.request.Request(base+'/prompt',json.dumps({'prompt':prompt}).encode(),{'Content-Type':'application/json'})
try:result=json.load(urllib.request.urlopen(req))
except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode()) from e
pid=result['prompt_id'];deadline=time.monotonic()+120
while time.monotonic()<deadline:
 h=json.load(urllib.request.urlopen(base+'/history/'+pid))
 if pid in h:break
 time.sleep(0.5)
assert pid in h
assert h[pid]['status']['status_str']=='success',h[pid]['status']
i=h[pid]['outputs']['6']['images'][0];p=comfy/'output'/i['subfolder']/i['filename']
assert Image.open(p).size==Image.open(R.parent/'grace_hopper.jpg').size
print('Real ComfyUI API face swap and PNG export passed:',p)
