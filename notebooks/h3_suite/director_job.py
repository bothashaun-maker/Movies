"""Director-only three-process orchestration with one bounded phase cache."""
def director_execute(graph,proc,log,directory,phase):
    previous=TIMEOUT_MINUTES
    globals()['TIMEOUT_MINUTES']={'preparation':5,'sampling':10,'decoding':5}[phase]
    try:return execute_graph(graph,proc,log,directory,phase)
    finally:globals()['TIMEOUT_MINUTES']=previous

def director_partition(graph,output_id,key):
    sample,decode=suite_partition(graph,output_id,key)
    ids=[k for k,n in sample.items() if n['class_type']=='MiniMaxH3DirectorCS']
    if len(ids)!=1:raise ValueError('Use exactly one Director timeline node.')
    ref=ids[0];i=sample[ref]['inputs'];state=json.loads(i.get('timeline_data') or '{}')
    preferred='model_ref2va' if str(state.get('reference_mode','OFF')).upper()!='OFF' else 'model'
    model=i.get(preferred,i.get('model',i.get('model_ref2va')))
    if model is None:raise ValueError('Director has no active diffusion model.')
    prep=copy.deepcopy(sample)
    prep[ref]={'class_type':'H3DirectorPrepare','inputs':{k:v for k,v in i.items() if k not in ('model','model_ref2va')}}
    prep[ref]['inputs']['checkpoint_id']=key
    prep=ancestors(prep,ref)
    sample[ref]={'class_type':'H3DirectorResume','inputs':{'model':model,'shift_video':i.get('shift_video',12.),'shift_audio':i.get('shift_audio',3.),'checkpoint_id':key}}
    sample=ancestors(sample,'h3_suite_checkpoint')
    if any(n['class_type'] in ('H3SuiteUNETLoader','H3SuitePDDApply','H3SuiteLoraLoader') for n in prep.values()):raise ValueError('Preparation still loads diffusion.')
    if any(n['class_type'] in ('H3SuiteCLIPLoader','H3SuiteVAELoader','MiniMaxH3DirectorCS') for n in sample.values()):raise ValueError('Sampling still loads the encoder or reference VAE.')
    return prep,sample,decode

def run_browser_job(job):
    graph=ancestors(suite_prune_director(job['graph']),str(job['output_id']));suite_preflight(graph)
    selected=suite_required_assets(graph);graph_text=json.dumps(graph,ensure_ascii=False)
    hashes={str(p.relative_to(COMFY/'input')):sha256_file(p) for p in (COMFY/'input').rglob('*') if p.is_file() and (p.name in graph_text or json.dumps(p.name)[1:-1] in graph_text)}
    identity={'graph':graph,'inputs':hashes,'weights':[(a['file'],a['sha256']) for a in selected],'director_runtime':DIRECTOR_CODE_ID,'comfy':COMFY_COMMIT,'mgpu':MGPU_COMMIT}
    key='h3_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    directory=ROOT/'runs'/key;directory.mkdir(parents=True,exist_ok=True)
    prep,sample,decode=director_partition(graph,str(job['output_id']),key)
    for name,data in [('settings',identity),('preparation_graph',prep),('sample_graph',sample),('decode_graph',decode)]:
        (directory/(name+'.json')).write_text(json.dumps(data,indent=2))
    report=directory/'result.json';geometry=suite_geometry(key)
    prepared=COMFY/'output/h3_latents'/(key+'.director.safetensors')
    if geometry and report.is_file():
        path=Path(json.loads(report.read_text())['file'])
        if path.is_file():verify_video(path,report);return path
    if not geometry:
        if not prepared.is_file():
            director_prepare_assets(suite_required_assets(prep,require_diffusion=False))
            with comfy_process('preparation',directory,[prep]) as (proc,log):director_execute(prep,proc,log,directory,'preparation')
            if not prepared.is_file():raise RuntimeError('Director preparation checkpoint was not saved.')
        director_prepare_assets(suite_required_assets(sample))
        with comfy_process('sampling',directory,[sample]) as (proc,log):director_execute(sample,proc,log,directory,'sampling')
        if not suite_geometry(key):raise RuntimeError('Sampling produced no saved latent.')
    validate_checkpoint(key)
    director_prepare_assets(suite_required_assets(decode,require_diffusion=False))
    with comfy_process('decoding',directory,[decode]) as (proc,log):entry=director_execute(decode,proc,log,directory,'decoding')
    paths=video_paths(entry)
    if len(paths)!=1:raise RuntimeError('Expected one MP4 output.')
    verify_video(paths[0],report);return paths[0]
