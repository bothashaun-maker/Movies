"""Strict reachable-graph asset selection and preflight for the new collection."""
_suite_required_assets=suite_required_assets

def suite_required_assets(graph,require_diffusion=True):
    selected=_suite_required_assets(graph,require_diffusion)
    if any(n['class_type']=='H3NextMolmo' for n in graph.values()):selected += [a for a in SUITE_ASSETS if a.get('pack')=='molmo']
    if any(n['class_type']=='H3NextMotion' and n['inputs'].get('strength',0)!=0 for n in graph.values()):selected += [a for a in SUITE_ASSETS if a['file']=='dials/h3_motion.safetensors']
    return list({a['sha256']:a for a in selected}.values())

_original_preflight=suite_preflight
def suite_preflight(graph):
    _original_preflight(graph)
    for n in graph.values():
        if n['class_type']=='H3NextReference':
            i=n['inputs'];w=i['width'];h=i['height'];seconds=i['seconds']
            if not all(isinstance(x,(int,float)) for x in (w,h,seconds)):raise ValueError('Use the visible numeric size/duration controls.')
            if w%32 or h%32 or not 32<=w<=1024 or not 32<=h<=1024 or w*h>430000 or not 1<=seconds<=5:raise ValueError('T4 limits: at most 0.43 MP, dimensions divisible by 32, one to five seconds.')
            if not isinstance(i['prompt'],str) or not i['prompt'].strip() or len(i['prompt'])>5000:raise ValueError('Enter a prompt of 1–5,000 characters.')
        if n['class_type']=='H3NextMotion':
            value=n['inputs']['strength']
            if not isinstance(value,(int,float)) or not -2<=value<=2:raise ValueError('Motion control must be between -2 and 2.')

def next_partition(graph,output_id,key):
    sample,decode=suite_partition(graph,output_id,key)
    refs=[k for k,n in sample.items() if n['class_type']=='H3NextReference']
    if len(refs)!=1:raise ValueError('Use exactly one H3 Next conditioning node.')
    ref=refs[0]
    prepare=ancestors(sample,ref)
    prepare['h3_next_prepare']={'class_type':'H3NextSavePreparation','inputs':{'conditioning':[ref,0],'latent':[ref,1],'checkpoint_id':key}}
    sample[ref]={'class_type':'H3NextLoadPreparation','inputs':{'checkpoint_id':key}}
    sample=ancestors(sample,'h3_suite_checkpoint')
    assert not any(n['class_type'] in ('H3NextMolmo','H3NextReference','H3SuiteVAELoader') for n in sample.values())
    return prepare,sample,decode

def run_browser_job(job):
    graph=ancestors(job['graph'],str(job['output_id']));suite_preflight(graph)
    selected=suite_required_assets(graph);graph_text=json.dumps(graph,ensure_ascii=False)
    hashes={str(p.relative_to(COMFY/'input')):sha256_file(p) for p in (COMFY/'input').rglob('*') if p.is_file() and (p.name in graph_text or json.dumps(p.name)[1:-1] in graph_text)}
    identity={'graph':graph,'inputs':hashes,'weights':[(a['file'],a['sha256']) for a in selected],'runtime':'h3-next-v2-three-process','code':NEXT_CODE_IDENTITY,'comfy':COMFY_COMMIT,'mgpu':MGPU_COMMIT}
    key='h3_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    directory=ROOT/'runs'/key;directory.mkdir(parents=True,exist_ok=True)
    prepare,sample,decode=next_partition(graph,str(job['output_id']),key)
    for name,data in [('settings',identity),('prepare_graph',prepare),('sample_graph',sample),('decode_graph',decode)]:
        (directory/(name+'.json')).write_text(json.dumps(data,indent=2))
    report=directory/'result.json';geometry=suite_geometry(key)
    prepared=COMFY/'output/h3_latents'/(key+'.prepared.safetensors')
    if geometry and report.is_file():
        path=Path(json.loads(report.read_text())['file'])
        if path.is_file():verify_video(path,report);return path
    if geometry:needed=suite_required_assets(decode,require_diffusion=False)
    elif prepared.is_file():needed=suite_required_assets({**sample,**decode})
    else:needed=selected
    suite_prepare_assets(needed)
    if not geometry:
        if not prepared.is_file():
            with comfy_process('preparation',directory,[prepare]) as (proc,log):execute_graph(prepare,proc,log,directory,'preparation')
            if not prepared.is_file():raise RuntimeError('Preparation did not save conditioning.')
        with comfy_process('sampling',directory,[sample]) as (proc,log):execute_graph(sample,proc,log,directory,'sampling')
        if not suite_geometry(key):raise RuntimeError('Sampling produced no saved latent.')
    validate_checkpoint(key)
    with comfy_process('decoding',directory,[decode]) as (proc,log):entry=execute_graph(decode,proc,log,directory,'decoding')
    paths=video_paths(entry)
    if len(paths)!=1:raise RuntimeError('Expected one MP4 output.')
    verify_video(paths[0],report)
    return paths[0]
