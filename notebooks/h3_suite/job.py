"""Partition edited Pixa/Director graphs and retain externally supplied audio."""
def suite_prune_director(graph):
    import json
    graph=copy.deepcopy(graph)
    for n in graph.values():
        if n['class_type']=='MiniMaxH3DirectorCS':
            inp=n['inputs'];timeline=json.loads(inp.get('timeline_data') or '{}')
            ref=str(timeline.get('reference_mode','OFF')).upper() != 'OFF'
            preferred='model_ref2va' if ref else 'model';other='model' if ref else 'model_ref2va'
            if preferred in inp:inp.pop(other,None)
    return graph

def suite_partition(graph,output_id,key):
    graph=suite_prune_director(graph)
    out=graph[output_id]['inputs']
    video=out['images'];audio=out['audio']
    decoder=graph[video[0]]
    if decoder['class_type']!='VAEDecode':raise ValueError('Connect the native video decoder to the isolated save node.')
    latent=decoder['inputs']['samples']
    if graph[latent[0]]['class_type'] not in ('KSampler','SamplerCustomAdvanced'):raise ValueError('Expected native sampler before VAE decode.')
    checkpoint='h3_suite_checkpoint'
    if checkpoint in graph:raise ValueError('Reserved suite node ID.')
    cp={'class_type':'H3SuiteCheckpoint','inputs':{'samples':latent,'checkpoint_id':key}}
    external=graph[audio[0]]['class_type']!='VAEDecodeAudio'
    if external:cp['inputs']['external_audio']=audio
    else:
        if graph[audio[0]]['inputs']['samples']!=latent:raise ValueError('Video/audio must use the same sampled latent.')
        graph[audio[0]]['inputs']['samples']=[checkpoint,0]
    graph[checkpoint]=cp
    sample=ancestors(graph,checkpoint)
    graph[checkpoint]={'class_type':'H3LoadDecodeCheckpoint','inputs':{'checkpoint_id':key}}
    decoder['inputs']['samples']=[checkpoint,0]
    if external:
        graph['h3_suite_audio']={'class_type':'H3SuiteLoadAudio','inputs':{'checkpoint_id':key}}
        audio=['h3_suite_audio',0]
    graph['h3_suite_video']={'class_type':'CreateVideo','inputs':{'images':video,'audio':audio,'fps':24.0,'bit_depth':8}}
    graph[output_id]={'class_type':'SaveVideo','inputs':{'video':['h3_suite_video',0],'filename_prefix':out['filename_prefix'],'format':'mp4','format.codec':'h264'}}
    decode=ancestors(graph,output_id)
    forbidden={'MiniMaxH3DirectorCS','KSampler','SamplerCustomAdvanced','PixaromaH3AudioSync','H3SuiteUNETLoader','H3SuiteCLIPLoader'}
    if any(n['class_type'] in forbidden for n in decode.values()):raise ValueError('Decode graph still depends on sampling models.')
    return sample,decode

def suite_geometry(key):
    import struct
    path=COMFY/'output/h3_latents'/(key+'.safetensors')
    if not path.is_file():return None
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if size>4*1024*1024:raise ValueError('Invalid latent checkpoint header.')
        header=json.loads(f.read(size))
    shape=header['video']['shape'];w=shape[-1]*16;h=shape[-2]*16;t=shape[2]
    if (t-2)%5:raise ValueError('Unexpected H3 latent time grid.')
    frames=(t-2)//5*17+5
    if w*h>430000 or frames>124:raise ValueError('T4 size limit exceeded.')
    globals()['canvas_size']=lambda *a:(w,h);globals()['frame_count']=lambda *a:frames
    return w,h,frames

def run_browser_job(job):
    graph=suite_prune_director(job['graph']);output_id=str(job['output_id'])
    graph=ancestors(graph,output_id)
    suite_preflight(graph)
    selected=suite_required_assets(graph)
    # Hash input media as well as prompt and timeline; changing a file must invalidate saved latents.
    graph_text=json.dumps(graph,ensure_ascii=False)
    hashes={str(p.relative_to(COMFY/'input')):sha256_file(p) for p in (COMFY/'input').rglob('*') if p.is_file() and (p.name in graph_text or json.dumps(p.name)[1:-1] in graph_text)}
    identity={'graph':graph,'inputs':hashes,'weights':[(a['file'],a['sha256']) for a in selected],'suite_version':1,'comfy':COMFY_COMMIT,'mgpu':MGPU_COMMIT}
    key='h3_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    directory=ROOT/'runs'/key;directory.mkdir(parents=True,exist_ok=True)
    sample,decode=suite_partition(graph,output_id,key)
    for name,data in [('settings',identity),('sample_graph',sample),('decode_graph',decode)]:
        (directory/(name+'.json')).write_text(json.dumps(data,indent=2))
    report=directory/'result.json'
    geometry=suite_geometry(key)
    if geometry and report.is_file():
        path=Path(json.loads(report.read_text())['file'])
        if path.is_file():verify_video(path,report);return path
    suite_prepare_assets(selected if not geometry else suite_required_assets(decode,require_diffusion=False))
    if not geometry:
        with comfy_process('sampling',directory,[sample]) as (proc,log):execute_graph(sample,proc,log,directory,'sampling')
        if not suite_geometry(key):raise RuntimeError('Sampling produced no saved latent.')
    validate_checkpoint(key)
    with comfy_process('decoding',directory,[decode]) as (proc,log):entry=execute_graph(decode,proc,log,directory,'decoding')
    paths=video_paths(entry)
    if len(paths)!=1:raise RuntimeError('Expected one MP4 output.')
    verify_video(paths[0],report)
    return paths[0]
