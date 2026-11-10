"""Ordinary prompt/upload/run/save graphs using the existing reviewed model set."""
import json
from pathlib import Path
R=Path(__file__).parent
class Graph:
 def __init__(self):self.nodes=[];self.links=[]
 def node(self,t,widgets=[],ins=[],outs=[],title=None):
  i=len(self.nodes)+1;self.nodes.append(dict(id=i,type=t,pos=[((i-1)//3)*370,((i-1)%3)*320],size=[330,270],flags={},order=i-1,mode=0,inputs=[dict(name=n,type=k,link=None) for n,k in ins],outputs=[dict(name=n,type=k,links=[]) for n,k in outs],properties={'Node name for S&R':t},widgets_values=widgets,title=title or t));return i
 def link(self,a,s,b,i):
  k=len(self.links)+1;t=self.nodes[a-1]['outputs'][s]['type'];self.links.append([k,a,s,b,i,t]);self.nodes[a-1]['outputs'][s]['links'].append(k);self.nodes[b-1]['inputs'][i]['link']=k
 def finish(self):return dict(last_node_id=len(self.nodes),last_link_id=len(self.links),nodes=self.nodes,links=self.links,groups=[],config={},extra={'ds':{'scale':0.75,'offset':[150,150]},'model_profile':'commercial-v7'},version=0.4)
PROMPTS={
 'Create':'Cinematic photograph of a rain-soaked city street at blue hour, a lone traveller beneath warm shop lights, grounded realistic materials, subtle film grain, natural skin texture, thoughtful framing, dramatic but believable lighting. Describe your own scene here.',
 'Image Edit':'Edit image 1: describe the change you want here. Preserve the original composition and everything unrelated to the requested change.',
 'Inpaint':'Edit only the selected region of image 1: describe the replacement here. Match the original lighting and perspective.',
 'Face Swap':'Use image 1 as the target photograph. Replace the selected person’s face identity with the person in image 2. Preserve the target pose, expression, clothing, framing and background.',
 'Head Swap':'head_swap: Use Picture 1 as the base photograph. Replace its head with the head from Picture 2, preserving the reference identity, eyes, nose and hair. Match the target pose, gaze and expression. Keep body, clothing, framing, lighting and background intact.'}
def generation(mode='Create',masked=False):
 g=Graph();n=g.node;l=g.link
 model=n('UNETLoader',['flux-2-klein-4b.safetensors','default'],[],[('MODEL','MODEL')]);clip=n('CLIPLoader',['qwen_3_4b.safetensors','flux2','default'],[],[('CLIP','CLIP')]);vae=n('VAELoader',['flux2-vae.safetensors'],[],[('VAE','VAE')])
 text=n('CLIPTextEncode',[PROMPTS[mode]],[('clip','CLIP')],[('CONDITIONING','CONDITIONING')],'PROMPT — describe your image or edit');l(clip,0,text,0)
 zero=n('ConditioningZeroOut',[],[('conditioning','CONDITIONING')],[('CONDITIONING','CONDITIONING')]);l(text,0,zero,0);pos,neg=text,zero
 refs=[];target=None;scaled_target=None
 if mode!='Create':
  target=n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'TARGET — image to edit'+(' / right-click → MaskEditor' if masked else ''));refs.append(target)
 if mode in ('Face Swap','Head Swap'):refs.append(n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'SOURCE — face / identity reference'))
 for ref in refs:
  scale=n('ImageScaleToTotalPixels',['lanczos',1.0,16],[('image','IMAGE')],[('IMAGE','IMAGE')],'Reference size — about 1 megapixel');l(ref,0,scale,0)
  if ref==target:scaled_target=scale
  enc=n('VAEEncode',[],[('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')]);l(scale,0,enc,0);l(vae,0,enc,1)
  p=n('ReferenceLatent',[],[('conditioning','CONDITIONING'),('latent','LATENT')],[('CONDITIONING','CONDITIONING')]);l(pos,0,p,0);l(enc,0,p,1);pos=p
  z=n('ReferenceLatent',[],[('conditioning','CONDITIONING'),('latent','LATENT')],[('CONDITIONING','CONDITIONING')]);l(neg,0,z,0);l(enc,0,z,1);neg=z
 if mode=='Create':
  w=n('PrimitiveInt',[1216,'fixed'],[],[('INT','INT')],'WIDTH — multiples of 16');h=n('PrimitiveInt',[832,'fixed'],[],[('INT','INT')],'HEIGHT — multiples of 16');dims=[(w,0),(h,0)]
 else:
  size=n('GetImageSize',[],[('image','IMAGE')],[('width','INT'),('height','INT'),('batch_size','INT')],'Use target image aspect ratio');l(scaled_target,0,size,0);dims=[(size,0),(size,1)]
 latent=n('EmptyFlux2LatentImage',[1],[('width','INT'),('height','INT')],[('LATENT','LATENT')]);scheduler=n('Flux2Scheduler',[4],[('width','INT'),('height','INT')],[('SIGMAS','SIGMAS')])
 for i,(a,s) in enumerate(dims):l(a,s,latent,i);l(a,s,scheduler,i)
 if mode=='Head Swap':
  adapter=n('LoraLoaderModelOnly',['bfs_head_v1_flux-klein_4b.safetensors',1.0],[('model','MODEL')],[('MODEL','MODEL')],'BFS head swap — strength 1.0');l(model,0,adapter,0);model=adapter
 noise=n('RandomNoise',[42,'randomize'],[],[('NOISE','NOISE')],'SEED');sampler=n('KSamplerSelect',['euler'],[],[('SAMPLER','SAMPLER')]);guide=n('CFGGuider',[1.0],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING')],[('GUIDER','GUIDER')]);l(model,0,guide,0);l(pos,0,guide,1);l(neg,0,guide,2)
 sample=n('SamplerCustomAdvanced',[],[('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')],[('output','LATENT'),('denoised_output','LATENT')])
 for a,s,i in [(noise,0,0),(guide,0,1),(sampler,0,2),(scheduler,0,3),(latent,0,4)]:l(a,s,sample,i)
 dec=n('VAEDecodeTiled',[512,64,64,8],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')]);l(sample,0,dec,0);l(vae,0,dec,1);last=dec
 if masked:
  last=n('CommercialMaskedEdit',[12],[('original','IMAGE'),('edited','IMAGE'),('mask','MASK')],[('IMAGE','IMAGE')],'Paste only inside painted mask');l(target,0,last,0);l(dec,0,last,1);l(target,1,last,2)
 save=n('SaveImage',['ImageStudio/'+mode.replace(' ','')],[('images','IMAGE')],[],'SAVE IMAGE — original image, no lettering');l(last,0,save,0);return g.finish()
def mesh():
 g=Graph();n=g.node;l=g.link
 src=n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'SOURCE — face to transfer');tgt=n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'TARGET — image to change')
 swap=n('CommercialMeshSwap',[0,0,'Feather'],[('source','IMAGE'),('target','IMAGE')],[('image','IMAGE'),('report','STRING')],'Face composite — similar poses only');l(src,0,swap,0);l(tgt,0,swap,1)
 save=n('SaveImage',['ImageStudio/FaceComposite'],[('images','IMAGE')],[]);l(swap,0,save,0);report=n('PreviewAny',[],[('source','*')],[('STRING','STRING')]);l(swap,1,report,0);return g.finish()
def build():
 result={'Create - Cinematic':generation(),'Image Edit':generation('Image Edit'),'Inpaint - Masked Edit':generation('Inpaint',True),'Face Swap - Reference Edit':generation('Face Swap'),'Face Swap - Masked':generation('Face Swap',True),'Head Swap - BFS':generation('Head Swap'),'Face Composite - Simple':mesh()}
 # Reuse the original working upscaling graphs byte-for-byte.
 old=json.loads((R.parent/'image_studio/workflows.json').read_text())
 for name in ['Upscale 2x','Upscale 4x','Restore SeedVR2']:result[name]=old[name]
 (R/'workflows.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print('Built',len(build()),'normal workflows')
