"""Run against a CPU ComfyUI server; never generates with SDXL/SeedVR2/FLUX3."""
import json,urllib.request,pathlib,os,time
R=pathlib.Path(__file__).parent
COMFY=pathlib.Path(os.environ.get('COMFY_ROOT',str(R.parent/'ComfyUI')))
BASE=os.environ.get('COMFY_URL','http://127.0.0.1:8189')
def get(path):return json.load(urllib.request.urlopen(BASE+path))
info=get('/object_info');workflows=json.loads((R/'workflows.json').read_text())
# Validate actual node inputs and connection types, including optional sockets.
for name,w in workflows.items():
 nodes={n['id']:n for n in w['nodes']}
 for n in nodes.values():
  spec=info[n['type']]['input'];schema={**spec.get('required',{}),**spec.get('optional',{})}
  for inp in n.get('inputs',[]):
   assert inp['name'] in schema,(name,n['type'],inp['name'])
   assert inp['link'] or inp['name'] in spec.get('optional',{}),(name,n['type'],'missing required link')
 for _,a,slot,b,inp,typ in w['links']:
  assert nodes[a]['outputs'][slot]['type']==typ
  assert nodes[b]['inputs'][inp]['type']==typ or nodes[b]['inputs'][inp]['type']=='*'
print('Live node schemas and graph connections validated:',len(workflows))
# Real model, tiny synthetic input: test 4x dimensions and PNG output through the server.
from PIL import Image
input_dir=COMFY/'input';models=COMFY/'models/upscale_models'
models.mkdir(parents=True,exist_ok=True);dest=models/'RealESRGAN_x4plus.pth'
if not dest.exists():dest.symlink_to(pathlib.Path(os.environ.get('REAL_ESRGAN_MODEL',str(R.parent/'RealESRGAN_x4plus.pth'))).resolve())
im=Image.new('RGB',(24,24));im.putdata([(x*10,y*10,(x+y)*5) for y in range(24) for x in range(24)]);im.save(input_dir/'studio-smoke.png')
for scale in [2,4]:
 prompt={'1':{'class_type':'LoadImage','inputs':{'image':'studio-smoke.png'}},'2':{'class_type':'UpscaleModelLoader','inputs':{'model_name':'RealESRGAN_x4plus.pth'}},'3':{'class_type':'ImageUpscaleWithModel','inputs':{'image':['1',0],'upscale_model':['2',0]}},'5':{'class_type':'SaveImage','inputs':{'images':['3',0],'filename_prefix':'studio-smoke-4x'}}}
 if scale==2:
  prompt['4']={'class_type':'ImageScaleBy','inputs':{'image':['3',0],'upscale_method':'lanczos','scale_by':0.5}}
  prompt['5']['inputs']={'images':['4',0],'filename_prefix':'studio-smoke-2x'}
 req=urllib.request.Request(BASE+'/prompt',json.dumps({'prompt':prompt}).encode(),{'Content-Type':'application/json'})
 result=json.load(urllib.request.urlopen(req));pid=result['prompt_id'];deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  history=get('/history/'+pid)
  if pid in history:break
  time.sleep(0.5)
 assert pid in history,history
 h=history[pid];assert h['status']['status_str']=='success',h['status']
 image=h['outputs']['5']['images'][0]
 path=COMFY/'output'/image['subfolder']/image['filename']
 out=Image.open(path);assert out.size==(24*scale,24*scale),out.size
 print('RealESRGAN CPU smoke:',scale,'x',out.size,path)
