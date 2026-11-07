import json, pathlib, hashlib, argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path);args=parser.parse_args()
ROOT=pathlib.Path(__file__).parent
MODEL='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'
def workflow(mode):
 nodes=[]; links=[]
 def node(t,xy,widgets,ins,outs,title=None):
  i=len(nodes)+1
  n=dict(id=i,type=t,pos=xy,size=[320,250],flags={},order=i-1,mode=0,inputs=[dict(name=a,type=b,link=None) for a,b in ins],outputs=[dict(name=a,type=b,links=[]) for a,b in outs],properties={'Node name for S&R':t},widgets_values=widgets)
  if title:n['title']=title
  nodes.append(n);return i
 def link(a,slot,b,inp):
  idx=len(links)+1; typ=nodes[a-1]['outputs'][slot]['type'];links.append([idx,a,slot,b,inp,typ]);nodes[a-1]['outputs'][slot]['links'].append(idx);nodes[b-1]['inputs'][inp]['link']=idx
 if mode in ['Upscale 2x','Upscale 4x','Restore SeedVR2']:
  im=node('LoadImage',[0,100],['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Upload an image to enlarge')
  if mode=='Restore SeedVR2':
   resize=node('ImageScaleBy',[360,100],['lanczos',2.0],[('image','IMAGE')],[('IMAGE','IMAGE')],'2x target; start with source at 1024px or smaller')
   link(im,0,resize,0)
   prep=node('SeedVR2Preprocess',[720,100],[],[('resized_images','IMAGE')],[('images','IMAGE')]);link(resize,0,prep,0)
   vae=node('VAELoader',[360,440],['seedvr2_ema_vae_fp16.safetensors'],[],[('VAE','VAE')])
   model=node('UNETLoader',[0,440],['seedvr2_3b_int8_convrot.safetensors','default'],[],[('MODEL','MODEL')])
   enc=node('VAEEncodeTiled',[1080,100],[512,128,64,8],[('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')]);link(prep,0,enc,0);link(vae,0,enc,1)
   cond=node('SeedVR2Conditioning',[1080,440],[],[('model','MODEL'),('vae_conditioning','LATENT')],[('positive','CONDITIONING'),('negative','CONDITIONING')]);link(model,0,cond,0);link(enc,0,cond,1)
   sample=node('KSampler',[1440,100],[42,'fixed',1,1.0,'euler','simple',1.0],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],[('LATENT','LATENT')])
   for a,slot,inp in [(model,0,0),(cond,0,1),(cond,1,2),(enc,0,3)]:link(a,slot,sample,inp)
   dec=node('VAEDecodeTiled',[1800,100],[512,128,64,8],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')]);link(sample,0,dec,0);link(vae,0,dec,1)
   last=node('SeedVR2PostProcessing',[2160,100],['none'],[('images','IMAGE'),('original_resized_images','IMAGE')],[('images','IMAGE')]);link(dec,0,last,0);link(resize,0,last,1)
  else:
   loader=node('UpscaleModelLoader',[0,440],['RealESRGAN_x4plus.pth'],[],[('UPSCALE_MODEL','UPSCALE_MODEL')])
   up=node('ImageUpscaleWithModel',[360,100],[],[('upscale_model','UPSCALE_MODEL'),('image','IMAGE')],[('IMAGE','IMAGE')]);link(loader,0,up,0);link(im,0,up,1)
   last=up
   if mode=='Upscale 2x':
    last=node('ImageScaleBy',[720,100],['lanczos',0.5],[('image','IMAGE')],[('IMAGE','IMAGE')],'4x model then downsample to 2x');link(up,0,last,0)
  save=node('SaveImage',[2520 if mode=='Restore SeedVR2' else 1080,100],['ImageStudio/'+mode],[('images','IMAGE')],[]);link(last,0,save,0)
  return dict(last_node_id=len(nodes),last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={'ds':{'scale':0.65,'offset':[180,180]}},version=0.4)
 c=node('CheckpointLoaderSimple' ,[0,0],[MODEL],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')])
 p=node('CLIPTextEncode',[380,0],['Cinematic photograph of a lone explorer overlooking a vast alien coastline, dramatic clouds, natural light, detailed textures, thoughtful composition'],[('clip','CLIP')],[('CONDITIONING','CONDITIONING')],'Describe your image')
 n=node('CLIPTextEncode',[380,300],['blurry, low quality, distorted anatomy, watermark, oversaturated'],[('clip','CLIP')],[('CONDITIONING','CONDITIONING')],'Negative prompt')
 link(c,1,p,0);link(c,1,n,0)
 if mode=='Upscale Creative':
  im=node('LoadImage',[0,400],['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Upload source; 2x creative enhancement')
  loader=node('UpscaleModelLoader',[380,650],['RealESRGAN_x4plus.pth'],[],[('UPSCALE_MODEL','UPSCALE_MODEL')])
  up=node('UltimateSDUpscale',[800,0],[2.0,42,'fixed',20,5.0,'dpmpp_2m','karras',0.2,'Linear',768,768,8,64,'Half Tile',0.15,64,8,32,True,True,1],[('image','IMAGE'),('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('vae','VAE'),('upscale_model','UPSCALE_MODEL')],[('IMAGE','IMAGE')],'Creative 2x — tiles can change faces and lettering')
  for a,slot,inp in [(im,0,0),(c,0,1),(p,0,2),(n,0,3),(c,2,4),(loader,0,5)]:link(a,slot,up,inp)
  save=node('SaveImage',[1200,0],['ImageStudio/Creative'],[('images','IMAGE')],[]);link(up,0,save,0)
  return dict(last_node_id=len(nodes),last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={'ds':{'scale':0.65,'offset':[180,180]}},version=0.4)
 if mode in ['Restyle','Inpaint']:
  im=node('LoadImage',[0,400],['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Upload image; right-click to open MaskEditor' if mode=='Inpaint' else 'Upload a source image')
  if mode=='Inpaint':
   latent=node('VAEEncodeForInpaint',[380,650],[6],[('pixels','IMAGE'),('vae','VAE'),('mask','MASK')],[('LATENT','LATENT')]);link(im,1,latent,2)
  else:latent=node('VAEEncode',[380,650],[],[('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')])
  link(im,0,latent,0);link(c,2,latent,1)
 else:latent=node('EmptyLatentImage',[380,650],[1024,1024,1],[],[('LATENT','LATENT')])
 s=node('KSampler',[770,0],[42,'randomize',30,5.5,'dpmpp_2m','karras',0.6 if mode=='Restyle' else 1.0],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],[('LATENT','LATENT')])
 for a,slot,inp in [(c,0,0),(p,0,1),(n,0,2),(latent,0,3)]:link(a,slot,s,inp)
 if mode=='Detail':
  up=node('LatentUpscaleBy',[1130,0],['bislerp',1.5],[('samples','LATENT')],[('LATENT','LATENT')]);link(s,0,up,0)
  second=node('KSampler',[1490,0],[42,'fixed',20,5.0,'dpmpp_2m','karras',0.3],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],[('LATENT','LATENT')],'Detail pass — 1536px; costs more VRAM')
  for a,slot,inp in [(c,0,0),(p,0,1),(n,0,2),(up,0,3)]:link(a,slot,second,inp)
  s=second
 dec=node('VAEDecodeTiled',[1900 if mode=='Detail' else 1130,350],[512,64,64,8],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')]);link(s,0,dec,0);link(c,2,dec,1)
 save=node('SaveImage',[2260 if mode=='Detail' else 1490,350],['ImageStudio/'+mode],[('images','IMAGE')],[]);link(dec,0,save,0)
 return dict(last_node_id=len(nodes),last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={'ds':{'scale':0.65,'offset':[180,180]}},version=0.4)
workflows={m:workflow(m) for m in ['Create','Restyle','Inpaint','Detail','Upscale 2x','Upscale 4x','Upscale Creative','Restore SeedVR2']}
# Expand the official template into ordinary visible nodes; retain its prompt upsampler.
flux=json.loads((ROOT/'flux3_template.json').read_text())
keep={1,2,3,4,5,6,7,8,11,12,13,14,17}
fnodes=[n for n in flux['nodes'] if n['id'] in keep]
flinks=[]
for edge in flux['links']:
 a,b=edge['origin_id'],edge['target_id']
 if a in keep and b in keep:flinks.append([edge['id'],a,edge['origin_slot'],b,edge['target_slot'],edge['type']])
# Replace the optional switch with a mandatory enhancement path.
flinks.append([90,11,0,13,0,'STRING'])
for n in fnodes:
 n['inputs']=[i for i in n.get('inputs',[]) if i.get('link') in {e[0] for e in flinks} or not i.get('widget')]
 for o in n.get('outputs',[]):o['links']=[]
 n.pop('widgets_values_named',None)
 if n['id']==17:n['widgets_values']=['A cinematic photograph of a lighthouse above a stormy coastline at sunset.']
 if n['id']==7:n['widgets_values']=[42,'randomize',16,1.0,'euler','simple',1.0]
 if n['id']==8:n['type']='VAEDecodeTiled';n['properties']['Node name for S&R']='VAEDecodeTiled';n['widgets_values']=[512,64,64,8]
 for i in n['inputs']:i['link']=None
fnodes.append(dict(id=30,type='SaveImage',pos=[1400,30],size=[320,250],flags={},order=30,mode=0,inputs=[dict(name='images',type='IMAGE',link=None)],outputs=[],properties={'Node name for S&R':'SaveImage'},widgets_values=['ImageStudio/FLUX3']))
flinks.append([91,8,0,30,0,'IMAGE'])
fm={n['id']:n for n in fnodes}
for lid,a,slot,b,inp,typ in flinks:
 # A removed external widget input can shift the remaining index: match by original name.
 original=next((n for n in flux['nodes'] if n['id']==b),None)
 input_name=original['inputs'][inp]['name'] if original else 'images'
 target=next(j for j,i in enumerate(fm[b]['inputs']) if i['name']==input_name)
 flinks[flinks.index([lid,a,slot,b,inp,typ])]=[lid,a,slot,b,target,typ]
 fm[b]['inputs'][target]['link']=lid;fm[a]['outputs'][slot]['links'].append(lid)
workflows['Create FLUX3']=dict(last_node_id=30,last_link_id=91,nodes=fnodes,links=flinks,groups=[],config={},extra={'ds':{'scale':0.5,'offset':[750,1000]}},version=0.4)
(ROOT/'workflows.json').write_text(json.dumps(workflows,indent=2))
source=(ROOT/'launcher.py').read_text().replace('WORKFLOW_PAYLOAD = {}','WORKFLOW_PAYLOAD = '+repr(workflows))
source=source.replace('ASSET_CATALOG_SOURCE = None',(ROOT/'assets.py').read_text())
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','id':'image-studio-guide','metadata':{},'source':(ROOT/'intro.md').read_text().splitlines(True)},{'cell_type':'code','id':'image-studio-start','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
(args.output or ROOT/'Image_Studio_Kaggle_v3.ipynb').write_text(json.dumps(nb,indent=1))
print('Built notebook',len(source),'characters')
