"""Visible native ComfyUI nodes, with an isolated output stage for T4 memory."""
import math

def browser_workflow(prompt,seed=42,megapixels=0.4,duration=5.0,family='ref2va',bridge=False,alpha=.1):
    width=round(math.sqrt(megapixels*1e6*2/3)/32)*32
    height=round(math.sqrt(megapixels*1e6*3/2)/32)*32
    length=round(duration*24);length+=(5-length%17)%17
    nodes=[];links=[]
    def node(i,kind,pos,inputs,outputs,widgets,size=(310,150),title=None):
        n={'id':i,'type':kind,'pos':list(pos),'size':list(size),'flags':{},'order':len(nodes),'mode':0,
           'inputs':[{'name':name,'type':typ,'link':None} for name,typ in inputs],
           'outputs':[{'name':name,'type':typ,'links':[]} for name,typ in outputs],
           'properties':{'Node name for S&R':kind},'widgets_values':widgets}
        if title:n['title']=title
        nodes.append(n);return n
    node(1,'UNETLoaderH3TensorParallel',(40,70),[],[('MODEL','MODEL')],[f'minimax_h3_{family}_pruned_int8_convrot.safetensors'],title='H3 diffusion · tensor parallel T4 ×2')
    node(2,'CLIPLoaderMultiGPU',(40,290),[],[('CLIP','CLIP')],['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','minimax','cuda:1'])
    node(3,'VAELoaderMultiGPU',(40,520),[],[('VAE','VAE')],['minimax_h3_video_vae_fp16.safetensors','cuda:0'])
    node(4,'VAELoaderMultiGPU',(40,720),[],[('VAE','VAE')],['minimax_h3_audio_vae_fp32.safetensors','cuda:1'])
    lora='minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors' if family=='ref2va' else 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors'
    node(5,'LoraLoaderModelOnly',(410,70),[('model','MODEL')],[('MODEL','MODEL')],[lora,1.0],title='Turbo LoRA · four steps')
    node(6,'MiniMaxH3SigmaShift',(770,70),[('model','MODEL')],[('MODEL','MODEL')],[6.0,3.0])
    widgets=[prompt,width,height,length]
    if bridge:widgets=[prompt,'MiniMaxH3_SemanticBridge_v1.safetensors',alpha,'per_token',width,height,length]
    node(7,'MiniMaxH3DistilledImageToVideo' if bridge else 'MiniMaxH3ImageToVideo',(410,310),
         [('clip','CLIP'),('vae','VAE'),('first_frame','IMAGE')],[('conditioning','CONDITIONING'),('latent','LATENT')],widgets,(510,380),title='Prompt / canvas / frames · optional first frame')
    node(8,'RandomNoise',(1000,320),[],[('NOISE','NOISE')],[seed,'fixed'])
    node(9,'BasicGuider',(1150,70),[('model','MODEL'),('conditioning','CONDITIONING')],[('GUIDER','GUIDER')],[])
    node(10,'KSamplerSelect',(1000,540),[],[('SAMPLER','SAMPLER')],['er_sde'])
    node(11,'BasicScheduler',(1000,740),[('model','MODEL')],[('SIGMAS','SIGMAS')],['sgm_uniform',4,1.0])
    node(12,'SamplerCustomAdvanced',(1400,330),[('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')],[('output','LATENT'),('denoised_output','LATENT')],[],(320,250))
    node(13,'VAEDecode',(1790,300),[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')],[],title='Video decode · native spatial/temporal tiles')
    node(14,'VAEDecodeAudio',(1790,550),[('samples','LATENT'),('vae','VAE')],[('AUDIO','AUDIO')],[])
    node(15,'CreateVideo',(2160,320),[('images','IMAGE'),('audio','AUDIO')],[('VIDEO','VIDEO')],[24.0,8])
    node(16,'H3IsolatedSaveVideo',(2530,320),[('video','VIDEO')],[],['kaggle_h3/clip'],(390,250),title='Save MP4 · isolated sampling / decode')
    node(18,'LoadImage',(410,800),[],[('IMAGE','IMAGE'),('MASK','MASK')],['','image'],(400,300),title='Optional image · upload, then connect to first_frame')
    lookup={n['id']:n for n in nodes}
    def link(a,out,b,inp):
        src=lookup[a];dst=lookup[b];slot=next(i for i,x in enumerate(dst['inputs']) if x['name']==inp)
        ident=len(links)+1;typ=src['outputs'][out]['type']
        links.append([ident,a,out,b,slot,typ]);src['outputs'][out]['links'].append(ident);dst['inputs'][slot]['link']=ident
    for a,o,b,k in [(1,0,5,'model'),(5,0,6,'model'),(2,0,7,'clip'),(3,0,7,'vae'),(6,0,9,'model'),(7,0,9,'conditioning'),(6,0,11,'model'),(8,0,12,'noise'),(9,0,12,'guider'),(10,0,12,'sampler'),(11,0,12,'sigmas'),(7,1,12,'latent_image'),(12,0,13,'samples'),(3,0,13,'vae'),(12,0,14,'samples'),(4,0,14,'vae'),(13,0,15,'images'),(14,0,15,'audio'),(15,0,16,'video')]:link(a,o,b,k)
    return {'last_node_id':18,'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],
            'config':{},'extra':{'ds':{'scale':0.65,'offset':[100,190]}},'version':0.4}
