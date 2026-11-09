import json,copy
from pathlib import Path
R=Path(__file__).parent
class Graph:
 def __init__(self):self.nodes=[];self.links=[]
 def node(self,t,widgets=[],ins=[],outs=[],title=None):
  i=len(self.nodes)+1
  self.nodes.append(dict(id=i,type=t,pos=[((i-1)//3)*370,((i-1)%3)*330],size=[330,280],flags={},order=i-1,mode=0,inputs=[dict(name=n,type=k,link=None) for n,k in ins],outputs=[dict(name=n,type=k,links=[]) for n,k in outs],properties={'Node name for S&R':t},widgets_values=widgets,title=title or t));return i
 def link(self,a,slot,b,inp):
  i=max([e[0] for e in self.links]+[0])+1;t=self.nodes[a-1]['outputs'][slot]['type'];self.links.append([i,a,slot,b,inp,t]);self.nodes[a-1]['outputs'][slot]['links'].append(i);self.nodes[b-1]['inputs'][inp]['link']=i
 def finish(self):return dict(last_node_id=len(self.nodes),last_link_id=max([e[0] for e in self.links]+[0]),nodes=self.nodes,links=self.links,groups=[],config={},extra={'ds':{'scale':0.5,'offset':[150,150]},'model_profile':'commercial-v7'},version=0.4)
def generation(mode='Create',episode=None,approved=False,duet=False):
 duet=duet or mode=='Duet Identity Edit'
 g=Graph();n=g.node;l=g.link
 brief=n('JJPEpisodeBrief',[episode or 'Custom','Square cover','A cinematic music video still.','Episode default','','','',True,'','',''],[],[('positive','STRING'),('negative','STRING'),('width','INT'),('height','INT'),('brief','STRING')])
 model=n('UNETLoader',['flux-2-klein-4b.safetensors','default'],[],[('MODEL','MODEL')]);clip=n('CLIPLoader',['qwen_3_4b.safetensors','flux2','default'],[],[('CLIP','CLIP')]);vae=n('VAELoader',['flux2-vae.safetensors'],[],[('VAE','VAE')])
 text=n('CLIPTextEncode',[],[('clip','CLIP'),('text','STRING')],[('CONDITIONING','CONDITIONING')]);l(clip,0,text,0);l(brief,0,text,1)
 zero=n('ConditioningZeroOut',[],[('conditioning','CONDITIONING')],[('CONDITIONING','CONDITIONING')]);l(text,0,zero,0);pos,neg=text,zero
 target=None;cast=None
 refs=[]
 if mode!='Create':
  target=n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Image 1 — clean target / scene');refs.append(target)
 if mode in ('Identity Edit','Duet Identity Edit') or approved:
  cast=n('JJPLoadApprovedCast',[],[],[('reference_image','IMAGE'),('character_prefix','STRING'),('cast_record','STRING')]) if approved else n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Image 2 — identity reference')
  refs.append(cast)
  if approved:
   g.nodes[brief-1]['widgets_values'].pop(4)
   g.nodes[brief-1]['inputs'].append(dict(name='character_prefix',type='STRING',link=None));l(cast,1,brief,0)
  if duet:
   second=n('JJPLoadApprovedCast',[],[],[('reference_image','IMAGE'),('character_prefix','STRING'),('cast_record','STRING')]) if approved else n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Image 3 — second identity')
   refs.append(second)
   if approved:
    combine=n('JJPCombineCast',[],[('lead_one','STRING'),('lead_two','STRING')],[('character_prefix','STRING'),('cast_record','STRING')]);l(cast,2,combine,0);l(second,2,combine,1)
    old=g.nodes[brief-1]['inputs'][0]['link'];g.links=[e for e in g.links if e[0]!=old];g.nodes[cast-1]['outputs'][1]['links'].remove(old)
    l(combine,0,brief,0);cast=combine
 for ref in refs:
  scale=n('ImageScaleToTotalPixels',['lanczos',1.0,1], [('image','IMAGE')],[('IMAGE','IMAGE')]);l(ref,0,scale,0)
  enc=n('VAEEncode',[],[('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')]);l(scale,0,enc,0);l(vae,0,enc,1)
  p=n('ReferenceLatent',[],[('conditioning','CONDITIONING'),('latent','LATENT')],[('CONDITIONING','CONDITIONING')]);l(pos,0,p,0);l(enc,0,p,1);pos=p
  z=n('ReferenceLatent',[],[('conditioning','CONDITIONING'),('latent','LATENT')],[('CONDITIONING','CONDITIONING')]);l(neg,0,z,0);l(enc,0,z,1);neg=z
 latent=n('EmptyFlux2LatentImage',[1],[('width','INT'),('height','INT')],[('LATENT','LATENT')]);l(brief,2,latent,0);l(brief,3,latent,1)
 schedule=n('Flux2Scheduler',[4],[('width','INT'),('height','INT')],[('SIGMAS','SIGMAS')]);l(brief,2,schedule,0);l(brief,3,schedule,1)
 noise=n('RandomNoise',[42,'randomize'],[],[('NOISE','NOISE')]);sampler=n('KSamplerSelect',['euler'],[],[('SAMPLER','SAMPLER')])
 guide=n('CFGGuider',[1.0],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING')],[('GUIDER','GUIDER')]);l(model,0,guide,0);l(pos,0,guide,1);l(neg,0,guide,2)
 sample=n('SamplerCustomAdvanced',[],[('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')],[('output','LATENT'),('denoised_output','LATENT')])
 for a,s,i in [(noise,0,0),(guide,0,1),(sampler,0,2),(schedule,0,3),(latent,0,4)]:l(a,s,sample,i)
 dec=n('VAEDecodeTiled',[512,64,64,8],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')]);l(sample,0,dec,0);l(vae,0,dec,1);last=dec
 if mode=='Masked Edit':
  paste=n('CommercialMaskedEdit',[12],[('original','IMAGE'),('edited','IMAGE'),('mask','MASK')],[('IMAGE','IMAGE')]);l(target,0,paste,0);l(dec,0,paste,1);l(target,1,paste,2);last=paste
 export=n('JJPArtworkExport',['From brief',True,2048],[('image','IMAGE'),('brief','STRING')]+([('cast_record','STRING')] if approved else []),[('delivery','IMAGE'),('saved_files','STRING')]);l(last,0,export,0);l(brief,4,export,1)
 if approved:l(cast,1 if duet else 2,export,2)
 # Explicit reference instructions in the brief scene (native multi-reference conditioning).
 if mode!='Create':g.nodes[brief-1]['widgets_values'][2]='Use image 1 as the scene and composition. '+('Replace only the selected person’s identity with the person from image 2. Preserve pose, clothing, lighting and the rest of the scene.' if mode=='Identity Edit' else 'Use image 2 for Lead One and image 3 for Lead Two; keep their identities distinct.' if duet else 'Edit image 1 as follows: describe the desired change here.')
 return g.finish()
def bfs(approved=False):
 wf=generation('Identity Edit',approved=approved)
 model=next(n for n in wf['nodes'] if n['type']=='UNETLoader');guide=next(n for n in wf['nodes'] if n['type']=='CFGGuider');brief=next(n for n in wf['nodes'] if n['type']=='JJPEpisodeBrief')
 brief['widgets_values'][2]='head_swap: Use Picture 1 as the base scene. Replace its head with the head from Picture 2, preserving the reference hair, eyes, nose and identity. Match the target head rotation, gaze and expression. Keep body, clothing, lighting, framing and background intact.'
 ident=wf['last_node_id']+1;edge=next(e for e in wf['links'] if e[1]==model['id'] and e[3]==guide['id']);lid=wf['last_link_id']+1
 wf['nodes'].append(dict(id=ident,type='LoraLoaderModelOnly',title='BFS dedicated head swap — 4B only, strength 1',pos=[370,1060],size=[330,240],flags={},order=ident-1,mode=0,inputs=[dict(name='model',type='MODEL',link=edge[0])],outputs=[dict(name='MODEL',type='MODEL',links=[lid])],properties={'Node name for S&R':'LoraLoaderModelOnly'},widgets_values=['bfs_head_v1_flux-klein_4b.safetensors',1.0]))
 edge[3]=ident;edge[4]=0;wf['links'].append([lid,ident,0,guide['id'],0,'MODEL']);guide['inputs'][0]['link']=lid;wf['last_node_id']=ident;wf['last_link_id']=lid
 return wf
def mesh(approved=False):
 g=Graph();n=g.node;l=g.link
 src=n('JJPLoadApprovedCast',[],[],[('reference_image','IMAGE'),('character_prefix','STRING'),('cast_record','STRING')]) if approved else n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Source face — similar pose')
 tgt=n('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Clean pre-swap target')
 swap=n('CommercialMeshSwap',[0,0,'Feather'],[('source','IMAGE'),('target','IMAGE')],[('image','IMAGE'),('report','STRING')]);l(src,0,swap,0);l(tgt,0,swap,1)
 save=n('SaveImage',['ImageStudioCommercial/MeshComposite'],[('images','IMAGE')],[]);l(swap,0,save,0)
 report=n('PreviewAny',[],[('source','*')],[('STRING','STRING')]);l(swap,1,report,0);return g.finish()
def build():
 result={m:generation(m) for m in ['Create','Reference Edit','Masked Edit','Identity Edit','Duet Identity Edit']}
 result['BFS Head Swap 4B']=bfs();result['Approved Cast BFS Head Swap 4B']=bfs(True)
 result['Mesh Face Composite']=mesh();result['Approved Cast Mesh Composite']=mesh(True)
 result['Approved Cast Identity Edit']=generation('Identity Edit',approved=True)
 result['Approved Duet Identity Edit']=generation('Duet Identity Edit',approved=True,duet=True)
 # Retain native upscaling graphs without importing/executing the obsolete builder.
 old=json.loads((R.parent/'image_studio/workflows.json').read_text())
 for name in ['Upscale 2x','Upscale 4x','Restore SeedVR2']:
  result[name]=old[name];result[name]['extra']['model_profile']='commercial-v7'
 g=Graph();im=g.node('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')]);save=g.node('JJPSaveApprovedCast',['Lead One','Approved reference','',False],[('image','IMAGE')],[('reference','IMAGE'),('record','STRING')]);g.link(im,0,save,0);result['Approve Cast']=g.finish()
 g=Graph();im=g.node('LoadImage',['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')]);brief=g.node('JJPEpisodeBrief',['Custom','Square cover','','None','','','',True,'','',''],[],[('positive','STRING'),('negative','STRING'),('width','INT'),('height','INT'),('brief','STRING')]);ex=g.node('JJPArtworkExport',['From brief',True,2048],[('image','IMAGE'),('brief','STRING')],[('delivery','IMAGE'),('saved_files','STRING')]);g.link(im,0,ex,0);g.link(brief,4,ex,1);result['Delivery Export']=g.finish()
 presets=json.loads((R/'jjp_presets.json').read_text())
 for ep in presets['episodes']:
  if ep!='Custom':result['JJP Episode - '+ep]=generation('Create',ep)
 (R/'workflows.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print('Built',len(build()),'workflows')
