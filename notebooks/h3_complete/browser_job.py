"""Split the actual edited browser graph at its joint latent boundary."""
import copy

def ancestors(graph,output):
    selected={}
    def visit(key):
        if key in selected:return
        node=graph[key];selected[key]=copy.deepcopy(node)
        for value in node['inputs'].values():
            if isinstance(value,list) and len(value)==2 and isinstance(value[0],str):visit(value[0])
    visit(output)
    return selected

def partition_browser_graph(graph,output_id,checkpoint_id):
    graph=copy.deepcopy(graph)
    output=graph[output_id]
    if output['class_type']!='H3IsolatedSaveVideo':raise ValueError('Use the isolated Save MP4 output node.')
    create_id=output['inputs']['video'][0];create=graph[create_id]
    if create['class_type']!='CreateVideo':raise ValueError('Connect CreateVideo to Save MP4.')
    video_id=create['inputs']['images'][0];audio_id=create['inputs']['audio'][0]
    if graph[video_id]['class_type']!='VAEDecode' or graph[audio_id]['class_type']!='VAEDecodeAudio':
        raise ValueError('Use native VAEDecode and VAEDecodeAudio for this H3 workflow.')
    samples=graph[video_id]['inputs']['samples']
    if samples!=graph[audio_id]['inputs']['samples']:raise ValueError('Both decoders must use the same joint sampler output.')
    if graph[samples[0]]['class_type']!='SamplerCustomAdvanced':raise ValueError('Expected SamplerCustomAdvanced output.')
    sample=ancestors(graph,samples[0]);checkpoint='h3_checkpoint'
    if checkpoint in graph:raise ValueError('Reserved checkpoint node ID already exists.')
    sample[checkpoint]={'class_type':'H3CheckpointBeforeDecode','inputs':{'samples':samples,'checkpoint_id':checkpoint_id}}
    for key in (video_id,audio_id):graph[key]['inputs']['samples']=[checkpoint,0]
    graph[checkpoint]={'class_type':'H3LoadDecodeCheckpoint','inputs':{'checkpoint_id':checkpoint_id}}
    graph[output_id]={'class_type':'SaveVideo','inputs':{'video':output['inputs']['video'],
        'filename_prefix':output['inputs']['filename_prefix'],'format':'mp4','format.codec':'h264'}}
    return sample,ancestors(graph,output_id)

def run_browser_job(job):
    graph=job['graph'];output_id=job['output_id']
    reachable=ancestors(graph,output_id)
    conditioning=[n for n in reachable.values() if n['class_type'] in ('MiniMaxH3ImageToVideo','MiniMaxH3DistilledImageToVideo')]
    if len(conditioning)!=1:raise ValueError('Use one H3 conditioning node per output.')
    c=conditioning[0]['inputs'];w,h,frames=c['width'],c['height'],c['length']
    if not (w%32==h%32==0 and 32<=w<=1024 and 32<=h<=1024 and w*h<=430000 and 5<=frames<=124 and frames%17==5):
        raise ValueError('T4 preset: dimensions divisible by 32, at most 0.43 MP and 124 frames on the H3 frame grid.')
    image_hashes={}
    for key,node in reachable.items():
        if node['class_type']=='LoadImage':
            name=node['inputs']['image'];path=(COMFY/'input'/name).resolve()
            if not path.is_relative_to((COMFY/'input').resolve()):raise ValueError('Reference image must be in ComfyUI input.')
            image_hashes[key]=sha256_file(path)
    identity={'graph':reachable,'images':image_hashes,'comfy':COMFY_COMMIT,'mgpu':MGPU_COMMIT,'weights':[(a['file'],a['sha256']) for a in selected_model_assets()],'version':3}
    run_id='h3_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    sample,decode=partition_browser_graph(reachable,output_id,run_id)
    directory=ROOT/'runs'/run_id;directory.mkdir(parents=True,exist_ok=True)
    for name,value in [('settings',identity),('sample_graph',sample),('decode_graph',decode)]:
        (directory/(name+'.json')).write_text(json.dumps(value,indent=2))
    # The controller executes the edited native graph, not a preset hidden behind it.
    globals()['prepare_job']=lambda:(run_id,directory,sample,decode)
    globals()['canvas_size']=lambda *args:(w,h)
    globals()['frame_count']=lambda *args:frames
    return render_clip()
