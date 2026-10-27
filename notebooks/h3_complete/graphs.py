import math
def canvas_size(mp, aspect):
    w = max(32, round(math.sqrt(mp*1_000_000*aspect[0]/aspect[1])/32)*32)
    h = max(32, round(math.sqrt(mp*1_000_000*aspect[1]/aspect[0])/32)*32)
    return w, h

def frame_count(seconds):
    frames = max(5, round(seconds*24))
    return frames + (5-frames%17)%17

def sampling_graph(checkpoint_id):
    width, height = canvas_size(MEGAPIXELS, ASPECT)
    cond = dict(clip=['2',0], vae=['3',0], prompt=PROMPT,
                width=width, height=height, length=frame_count(DURATION))
    cond_type = 'MiniMaxH3ImageToVideo'
    if USE_BRIDGE:
        cond_type = 'MiniMaxH3DistilledImageToVideo'
        cond.update(distilled_adapter='MiniMaxH3_SemanticBridge_v1.safetensors',
                    alpha=BRIDGE_ALPHA, magnitude_match='per_token')
    graph = {}
    def add(n, kind, **inputs): graph[str(n)] = {'class_type':kind, 'inputs':inputs}
    add(1, 'UNETLoaderH3TensorParallel', unet_name=f'minimax_h3_{MODEL_FAMILY}_pruned_int8_convrot.safetensors')
    add(2, 'CLIPLoaderMultiGPU', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='cuda:1')
    add(3, 'VAELoaderMultiGPU', vae_name=f'minimax_h3_video_vae_{VIDEO_VAE}.safetensors',device='cuda:0')
    add(5, 'LoraLoaderModelOnly', model=['1',0],
        lora_name=('minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors' if MODEL_FAMILY=='ref2va' else 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors'), strength_model=1.0)
    add(6, 'MiniMaxH3SigmaShift', model=['5',0], shift_video=6.0, shift_audio=3.0)
    model=['6',0]
    if MODE == 'image':
        from PIL import Image, ImageOps
        with Image.open(IMAGE_PATH) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im = ImageOps.pad(im, (width,height), method=Image.Resampling.LANCZOS, color=(0,0,0))
            im.save(COMFY/'input/kaggle_first_frame.png')
        add(18, 'LoadImage', image='kaggle_first_frame.png')
        cond['first_frame'] = ['18',0]
    add(7, cond_type, **cond)
    add(8, 'RandomNoise', noise_seed=SEED)
    add(9, 'BasicGuider', model=model, conditioning=['7',0])
    add(10, 'KSamplerSelect', sampler_name='er_sde')
    add(11, 'BasicScheduler', model=model, scheduler='sgm_uniform', steps=STEPS, denoise=1.0)
    add(12, 'SamplerCustomAdvanced', noise=['8',0], guider=['9',0], sampler=['10',0], sigmas=['11',0], latent_image=['7',1])
    if SAMPLING_PRESET=='taomate3':
        graph['5']['inputs']['lora_name']='minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors'
        del graph['6']
        graph['9']['inputs']['model']=['5',0]
        graph['10']['inputs']['sampler_name']='res_multistep'
        graph['11']={'class_type':'ManualSigmas','inputs':{'sigmas':'1.0, 0.961165, 0.853333, 0.0'}}
    add(19,'H3CheckpointBeforeDecode',samples=['12',0],checkpoint_id=checkpoint_id)
    return graph

def decode_graph(checkpoint_id):
    return {
        '19': {'class_type':'H3LoadDecodeCheckpoint', 'inputs':{'checkpoint_id':checkpoint_id}},
        '3': {'class_type':'VAELoaderMultiGPU', 'inputs':{
            'vae_name':f'minimax_h3_video_vae_{VIDEO_VAE}.safetensors','device':'cuda:0'}},
        '4': {'class_type':'VAELoaderMultiGPU', 'inputs':{
            'vae_name':'minimax_h3_audio_vae_fp32.safetensors','device':'cuda:1'}},
        '13': {'class_type':'VAEDecode', 'inputs':{'samples':['19',0],'vae':['3',0]}},
        '14': {'class_type':'VAEDecodeAudio', 'inputs':{'samples':['19',0],'vae':['4',0]}},
        '15': {'class_type':'CreateVideo', 'inputs':{'images':['13',0],'audio':['14',0],'fps':24.0,'bit_depth':8}},
        '16': {'class_type':'SaveVideo', 'inputs':{'video':['15',0],
            'filename_prefix':f'kaggle_h3/{checkpoint_id}','format':'mp4','format.codec':'h264'}},
    }

def prepare_job():
    if MODE == 'image':
        (COMFY/'input').mkdir(parents=True, exist_ok=True)
    sample = sampling_graph('pending')
    identity = {
        'workflow_version':2, 'sample_graph':sample,
        'comfy':COMFY_COMMIT, 'multigpu':MGPU_COMMIT,
        'weights':[(a['file'], a['sha256']) for a in selected_model_assets()],
        'image_sha256':sha256_file(COMFY/'input/kaggle_first_frame.png') if MODE == 'image' else None,
        'runtime':{k:child_env[k] for k in ('MMH3_H3_FP16','MMH3_TP_CHUNKS','MMH3_TP_COMPILE')},
        'kitchen_attention':USE_KITCHEN_ATTENTION,
    }
    run_id = 'h3_'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    sample['19']['inputs']['checkpoint_id'] = run_id
    directory = ROOT/'runs'/run_id
    directory.mkdir(parents=True, exist_ok=True)
    decode = decode_graph(run_id)
    for name, data in [('settings',identity),('sample_graph',sample),('decode_graph',decode)]:
        (directory/(name+'.json')).write_text(json.dumps(data, indent=2))
    return run_id, directory, sample, decode
