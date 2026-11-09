"""JJP brief, approved cast and deterministic artwork export stages."""
from pathlib import Path
import json,hashlib,re,uuid,os,datetime
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps,PngImagePlugin
import folder_paths
PRESETS=json.loads((Path(__file__).parent/'jjp_presets.json').read_text())
CAST_ROLES=['Lead One','Lead Two','Accuser (non-human)','Seeker (faceless)']

def slug(value):return re.sub(r'[^a-zA-Z0-9_-]+','-',value).strip('-')[:70] or 'untitled'
def output_root():return Path(folder_paths.get_output_directory())/'JJP'
def cast_roots():
 roots=[output_root()/'cast']
 inp=Path(os.environ.get('JJP_INPUT_ROOT','/kaggle/input'))
 if inp.exists():roots.append(inp)
 return roots

def cast_records():
 records={}
 for root in cast_roots():
  if not root.exists():continue
  for path in root.rglob('*.cast.json'):
   try:
    record=json.loads(path.read_text());image=path.parent/record['image']
    if not record.get('approved') or Path(record['image']).name!=record['image'] or not image.is_file():continue
    key=record['id']+' | '+record['role']+' | '+record['label']
    records[key]=(record,image)
   except (OSError,ValueError,KeyError):continue
 return records

def single_image(image):
 if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]!=3:raise ValueError('Use one RGB still at a time.')
 return Image.fromarray((image[0].detach().cpu().numpy().clip(0,1)*255).round().astype('uint8'))
def tensor(image):return torch.from_numpy(np.array(image.convert('RGB'),copy=True).astype('float32')/255).unsqueeze(0)
def read_brief(value):
 data=json.loads(value)
 if data.get('schema')!='jjp-image-brief-v1':raise ValueError('Connect a JJP Episode Brief to this stage.')
 return data

class JJPEpisodeBrief:
 @classmethod
 def INPUT_TYPES(cls):
  return {'required':{'episode':(list(PRESETS['episodes']),),'format':(list(PRESETS['formats']),),'scene':('STRING',{'multiline':True,'default':''}),'cast':(['Episode default','None','Lead One','Lead Two','Duet','Accuser','Seeker'],),'character_prefix':('STRING',{'multiline':True,'default':''}),'title':('STRING',{'default':''}),'scripture_reference':('STRING',{'default':''}),'use_episode_reference':('BOOLEAN',{'default':True})},'optional':{'custom_world':('STRING',{'multiline':True,'default':''}),'custom_palette':('STRING',{'default':''}),'custom_wing':('STRING',{'default':''})}}
 RETURN_TYPES=('STRING','STRING','INT','INT','STRING');RETURN_NAMES=('positive','negative','width','height','brief')
 FUNCTION='brief';CATEGORY='JJP/Production'
 def brief(self,episode,format,scene,cast,character_prefix,title,scripture_reference,use_episode_reference,custom_world='',custom_palette='',custom_wing=''):
  p=dict(PRESETS['episodes'][episode])
  if custom_world.strip():p['world']=custom_world.strip()
  if custom_palette.strip():p['palette']=custom_palette.strip()
  if custom_wing.strip():p['wing']=custom_wing.strip()
  role=p['cast'] if cast=='Episode default' else cast
  reference=scripture_reference.strip() or (p['reference'] if use_episode_reference else '')
  prompt=' '.join(filter(None,[PRESETS['style'],p['world'],'Palette: '+p['palette']+'.',PRESETS['cast_rule'],'Cast for this shot: '+role+'.',character_prefix.strip(),scene.strip()]))
  w,h=PRESETS['formats'][format]
  data={'schema':'jjp-image-brief-v1','brand':PRESETS['brand'],'episode':episode,'wing':p['wing'],'palette':p['palette'],'cast':role,'format':format,'title':title.strip() or (episode if episode!='Custom' else ''),'scripture_reference':reference,'reference_status':'User must verify the reference against the intended passage; no scripture quotation was generated.','positive':prompt,'negative':PRESETS['negative'],'character_prefix':character_prefix,'generation_size':[w,h],'source_document':'JJP Launch System, Edition I','rights_notes':['Generation and likeness fidelity require human review.','Commercial-v7 uses Apache-2.0/BSD-3-Clause assets; preserve provenance.json and license notices. Generative identity edits require likeness review.'],'review':['Compare human likeness with approved reference; reject drift.','Confirm non-human/faceless antagonist design.','Verify scripture reference and any supplied quotation.','Check title spelling, composition and readable type.','Review likeness dignity and synthetic-content disclosure when publishing.']}
  return prompt,PRESETS['negative'],w,h,json.dumps(data,ensure_ascii=False)

class JJPSaveApprovedCast:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'role':(CAST_ROLES,),'label':('STRING',{'default':'Approved reference'}),'prompt_prefix':('STRING',{'multiline':True,'default':''}),'approved':('BOOLEAN',{'default':False})}}
 RETURN_TYPES=('IMAGE','STRING');RETURN_NAMES=('reference','record');FUNCTION='save';CATEGORY='JJP/Cast';OUTPUT_NODE=True
 def save(self,image,role,label,prompt_prefix,approved):
  if not approved:raise ValueError('Inspect the reference and set approved=true only when you want to add it to your cast library.')
  im=single_image(image);ident=uuid.uuid4().hex[:12];root=output_root()/'cast'/slug(role);root.mkdir(parents=True,exist_ok=True)
  path=root/(slug(label)+'-'+ident+'.png');im.save(path)
  record={'schema':'jjp-cast-v1','id':ident,'role':role,'label':label.strip() or 'Approved reference','prompt_prefix':prompt_prefix,'approved':True,'image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  path.with_suffix('.cast.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
  return {'ui':{'text':[f'Saved approved cast reference: {record["role"]} / {record["label"]}. Save private outputs to preserve your library.']},'result':(image,json.dumps(record,ensure_ascii=False))}

class JJPLoadApprovedCast:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'reference':(list(cast_records()) or ['Add approved cast reference first'],)}}
 RETURN_TYPES=('IMAGE','STRING','STRING');RETURN_NAMES=('reference_image','character_prefix','cast_record');FUNCTION='load';CATEGORY='JJP/Cast'
 @classmethod
 def IS_CHANGED(cls,reference):
  found=cast_records().get(reference)
  return hashlib.sha256(found[1].read_bytes()).hexdigest() if found else float('nan')
 def load(self,reference):
  found=cast_records().get(reference)
  if not found:raise ValueError('No matching approved reference. Use JJP - Approve Cast, then refresh the page/node options, or attach your saved private cast output.')
  record,path=found
  if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Approved cast file changed; review and register it again.')
  with Image.open(path) as im:reference_tensor=tensor(im)
  return reference_tensor,record['prompt_prefix'],json.dumps(record,ensure_ascii=False)

class JJPCombineCast:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'lead_one':('STRING',{'forceInput':True}),'lead_two':('STRING',{'forceInput':True})}}
 RETURN_TYPES=('STRING','STRING');RETURN_NAMES=('character_prefix','cast_record');FUNCTION='combine';CATEGORY='JJP/Cast'
 def combine(self,lead_one,lead_two):
  first,second=json.loads(lead_one),json.loads(lead_two)
  if first.get('role')!='Lead One' or second.get('role')!='Lead Two':raise ValueError('Select an approved Lead One reference and an approved Lead Two reference for this duet.')
  prefix='Lead One: '+first.get('prompt_prefix','')+'; Lead Two: '+second.get('prompt_prefix','')
  return prefix,json.dumps({'schema':'jjp-cast-group-v1','references':[first,second]},ensure_ascii=False)

def font(size):
 for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf']:
  if Path(path).is_file():return ImageFont.truetype(path,size)
 return ImageFont.load_default(size=size)
def draw_fit(draw,text,box,fill,start_size):
 if not text:return
 x,y,w,h=box
 for size in range(start_size,11,-1):
  f=font(size);words=text.split();lines=[];line=''
  for word in words:
   trial=(line+' '+word).strip()
   if draw.textlength(trial,font=f)>w and line:lines.append(line);line=word
   else:line=trial
  if line:lines.append(line)
  line_height=size+6
  if len(lines)*line_height<=h and all(draw.textlength(t,font=f)<=w for t in lines):break
 else:raise ValueError('Text is too long for the selected export. Shorten the title or reference.')
 for i,line in enumerate(lines):draw.text((x,y+i*line_height),line,font=f,fill=fill)

class JJPArtworkExport:
 @classmethod
 def INPUT_TYPES(cls):
  return {'required':{'image':('IMAGE',),'brief':('STRING',{'forceInput':True}),'layout':(['From brief','Original size','Square cover','YouTube 16:9','Vertical 9:16'],),'lettering':('BOOLEAN',{'default':True}),'long_edge':('INT',{'default':2048,'min':256,'max':8192,'step':64})},'optional':{'cast_record':('STRING',{'forceInput':True})},'hidden':{'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
 RETURN_TYPES=('IMAGE','STRING');RETURN_NAMES=('delivery','saved_files');FUNCTION='export';CATEGORY='JJP/Delivery';OUTPUT_NODE=True
 def export(self,image,brief,layout,lettering,long_edge,cast_record=None,prompt=None,extra_pnginfo=None):
  data=read_brief(brief);master=single_image(image);ident=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
  root=output_root()/'episodes'/slug(data['episode'])/ident;root.mkdir(parents=True,exist_ok=False)
  meta=PngImagePlugin.PngInfo();meta.add_text('jjp_brief',brief)
  if prompt is not None:meta.add_text('prompt',json.dumps(prompt))
  for key,val in (extra_pnginfo or {}).items():meta.add_text(key,json.dumps(val))
  master.save(root/'master.png',pnginfo=meta)
  # Padding preserves the whole source frame. Interpolation is explicitly reported.
  if layout=='From brief':layout={'Square cover':'Square cover','YouTube thumbnail':'YouTube 16:9','Vertical still':'Vertical 9:16'}.get(data['format'],'Original size')
  if layout=='Original size':canvas=master.copy()
  else:
   ratio={'Square cover':(1,1),'YouTube 16:9':(16,9),'Vertical 9:16':(9,16)}[layout]
   w=long_edge if ratio[0]>=ratio[1] else round(long_edge*ratio[0]/ratio[1]);h=long_edge if ratio[1]>=ratio[0] else round(long_edge*ratio[1]/ratio[0])
   canvas=Image.new('RGB',(w,h),'#101014');art=ImageOps.contain(master,(w,h),Image.Resampling.LANCZOS);canvas.paste(art,((w-art.width)//2,(h-art.height)//2))
  if lettering:
   w,h=canvas.size;draw=ImageDraw.Draw(canvas);band=max(110,round(h*0.23));draw.rectangle((0,h-band,w,h),fill='#101014')
   margin=max(12,round(w*0.04));base=max(14,round(min(w,h)*0.023))
   draw_fit(draw,PRESETS['brand'],(margin,h-band+10,w-2*margin,base+12),'#d9b56b',base)
   draw_fit(draw,data['title'],(margin,h-band+base+25,w-2*margin,int(band*0.43)),'#f6f2e9',max(20,round(min(w,h)*0.05)))
   draw_fit(draw,data['scripture_reference'],(margin,h-max(35,base+15),w-2*margin,base+12),'#d9b56b',base)
  canvas.save(root/'delivery.png',pnginfo=meta)
  if cast_record:data['cast_reference']=json.loads(cast_record)
  data['model_profile']='commercial-v7'
  provenance=Path(folder_paths.get_output_directory()).parent/'provenance.json'
  if provenance.exists():data['model_provenance']=json.loads(provenance.read_text())
  data.update({'run_id':ident,'files':['master.png','delivery.png','episode.json'],'delivery_layout':layout,'delivery_size':list(canvas.size),'master_size':list(master.size),'delivery_resize':'Lanczos interpolation/padding only; use the neural upscaling graph before export for learned detail.','lettering':lettering,'approval_status':'Needs human review; export does not approve likeness, scripture, model rights or publication.'})
  (root/'episode.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
  summary='Saved master.png, delivery.png and episode.json at '+str(root)
  return {'ui':{'images':[{'filename':'delivery.png','subfolder':str(root.relative_to(Path(folder_paths.get_output_directory()))),'type':'output'}],'text':[summary]},'result':(tensor(canvas),summary)}

NODE_CLASS_MAPPINGS={c.__name__:c for c in [JJPEpisodeBrief,JJPSaveApprovedCast,JJPLoadApprovedCast,JJPCombineCast,JJPArtworkExport]}
NODE_DISPLAY_NAME_MAPPINGS={'JJPEpisodeBrief':'JJP Episode / Shot Brief','JJPSaveApprovedCast':'JJP Approve Cast Reference','JJPLoadApprovedCast':'JJP Load Approved Cast','JJPCombineCast':'JJP Two-Character Cast','JJPArtworkExport':'JJP Artwork + Episode Export'}
