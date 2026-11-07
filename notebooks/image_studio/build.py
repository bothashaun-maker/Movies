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
 c=node('CheckpointLoaderSimple',[0,0],[MODEL],[],[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')])
 p=node('CLIPTextEncode',[380,0],['Cinematic photograph of a lone explorer overlooking a vast alien coastline, dramatic clouds, natural light, detailed textures, thoughtful composition'],[('clip','CLIP')],[('CONDITIONING','CONDITIONING')],'Describe your image')
 n=node('CLIPTextEncode',[380,300],['blurry, low quality, distorted anatomy, watermark, oversaturated'],[('clip','CLIP')],[('CONDITIONING','CONDITIONING')],'Negative prompt')
 link(c,1,p,0);link(c,1,n,0)
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
 return dict(last_node_id=len(nodes),last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={'ds':{'scale':0.65,'offset':[50,100]}},version=0.4)
workflows={m:workflow(m) for m in ['Create','Restyle','Inpaint','Detail']}
(ROOT/'workflows.json').write_text(json.dumps(workflows,indent=2))
source=(ROOT/'launcher.py').read_text().replace('WORKFLOW_PAYLOAD = {}','WORKFLOW_PAYLOAD = '+repr(workflows))
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','metadata':{},'source':(ROOT/'intro.md').read_text().splitlines(True)},{'cell_type':'code','id':'image-studio-start','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}]}
(args.output or ROOT/'Image_Studio_Kaggle_v1.ipynb').write_text(json.dumps(nb,indent=1))
print('Built notebook',len(source),'characters')
