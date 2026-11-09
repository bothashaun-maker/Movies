"""Local CPU ComfyUI checks: deterministic delivery, cast persistence and duet wiring."""
import json,time,urllib.request,urllib.error,pathlib,os
import numpy as np
from PIL import Image
R=pathlib.Path(__file__).parent;COMFY=pathlib.Path(os.environ.get('COMFY_ROOT',str(R.parent/'ComfyUI')));BASE=os.environ.get('COMFY_URL','http://127.0.0.1:8189')
def get(path):return json.load(urllib.request.urlopen(BASE+path))
def run(prompt):
 req=urllib.request.Request(BASE+'/prompt',json.dumps({'prompt':prompt}).encode(),{'Content-Type':'application/json'})
 try:r=json.load(urllib.request.urlopen(req))
 except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode()) from e
 pid=r['prompt_id'];deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  history=get('/history/'+pid)
  if pid in history:break
  time.sleep(.25)
 assert pid in history;h=history[pid];assert h['status']['status_str']=='success',h['status'];return h
fixture=Image.fromarray(np.tile(np.linspace(0,255,384,dtype='uint8')[None,:,None],(256,1,3)))
fixture.save(COMFY/'input/jjp-fixture.png')
brief={'episode':'Verdict of the Lamb','format':'Square cover','scene':'A test fixture, not final artwork.','cast':'None','character_prefix':'','title':'TETELESTAI','scripture_reference':'John 19:30','use_episode_reference':True}
prompt={'1':{'class_type':'LoadImage','inputs':{'image':'jjp-fixture.png'}},'2':{'class_type':'JJPEpisodeBrief','inputs':brief},'3':{'class_type':'JJPArtworkExport','inputs':{'image':['1',0],'brief':['2',4],'layout':'From brief','lettering':True,'long_edge':512}}}
h=run(prompt);image=h['outputs']['3']['images'][0];root=COMFY/'output'/image['subfolder'];data=json.loads((root/'episode.json').read_text())
assert Image.open(root/'delivery.png').size==(512,512)
assert np.array_equal(np.asarray(Image.open(root/'master.png')),np.asarray(fixture))
assert data['scripture_reference']=='John 19:30';assert data['title']=='TETELESTAI'
print('ComfyUI brief → branded export: master pixels, delivery dimensions and episode JSON passed')
# Register public-domain test portraits only, never user/private cast photos.
for filename,role,label in [('astronaut.png','Lead One','CPU fixture NASA'),('grace_hopper.jpg','Lead Two','CPU fixture US Navy')]:
 Image.open(R.parent/filename).convert('RGB').save(COMFY/'input'/('jjp-'+filename+'.png'))
 run({'1':{'class_type':'LoadImage','inputs':{'image':'jjp-'+filename+'.png'}},'2':{'class_type':'JJPSaveApprovedCast','inputs':{'image':['1',0],'role':role,'label':label,'prompt_prefix':label,'approved':True}}})
keys=get('/object_info/JJPLoadApprovedCast')['JJPLoadApprovedCast']['input']['required']['reference'][0]
lead1=next(x for x in keys if 'CPU fixture NASA' in x);lead2=next(x for x in keys if 'CPU fixture US Navy' in x)
a=Image.open(R.parent/'grace_hopper.jpg').convert('RGB');b=Image.open(R.parent/'astronaut.png').convert('RGB');target=Image.new('RGB',(a.width+b.width,max(a.height,b.height)),'black');target.paste(a,(0,0));target.paste(b,(a.width,0));target.save(COMFY/'input/jjp-duet-target.png')
workflow=json.loads((R/'workflows.json').read_text())['JJP - Duet Face Swap from Cast']
# Construct the same visible duet node chain as the generated UI workflow.
prompt={'1':{'class_type':'JJPLoadApprovedCast','inputs':{'reference':lead1}},'2':{'class_type':'JJPLoadApprovedCast','inputs':{'reference':lead2}},'3':{'class_type':'LoadImage','inputs':{'image':'jjp-duet-target.png'}},'4':{'class_type':'StudioFaceDetect','inputs':{'image':['1',0]}},'5':{'class_type':'StudioFaceDetect','inputs':{'image':['2',0]}},'6':{'class_type':'StudioFaceDetect','inputs':{'image':['3',0]}},'7':{'class_type':'StudioFaceSwap','inputs':{'source_faces':['4',0],'target_faces':['6',0],'source_index':0,'target_index':0}},'8':{'class_type':'StudioFaceDetect','inputs':{'image':['7',0]}},'9':{'class_type':'StudioFaceSwap','inputs':{'source_faces':['5',0],'target_faces':['8',0],'source_index':0,'target_index':1}},'10':{'class_type':'JJPCombineCast','inputs':{'lead_one':['1',2],'lead_two':['2',2]}},'11':{'class_type':'JJPEpisodeBrief','inputs':{**brief,'cast':'Duet','character_prefix':['10',0]}},'12':{'class_type':'JJPArtworkExport','inputs':{'image':['9',0],'brief':['11',4],'cast_record':['10',1],'layout':'Original size','lettering':False,'long_edge':1024}}}
h=run(prompt);image=h['outputs']['12']['images'][0];root=COMFY/'output'/image['subfolder'];data=json.loads((root/'episode.json').read_text());assert Image.open(root/'delivery.png').size==target.size
assert len(data['cast_reference']['references'])==2
assert data['cast']=='Duet'
print('ComfyUI approved cast save/load → two real CPU swaps → delivery/provenance passed')
