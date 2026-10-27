"""Pinned suite assets and bounded on-demand cache management."""
ASSETS.extend([
 dict(repo='lightx2v/Minimax-h3-Turbo',revision='30f485ed874aa44c8183348ad361c5eabad318da',file='minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors',dest='loras/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors',size=1956192992,sha256='449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558'),
 dict(repo='aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI',revision='c169e9cd1dab4ac56b413b1343218ab4a3af2f19',file='minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors',dest='pdd_acc/minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors',size=1658719592,sha256='1dce71b90e5d993d3309474d47e414766e65dbd760f18978c399c82db5cda0ea'),
 dict(repo='aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI',revision='c169e9cd1dab4ac56b413b1343218ab4a3af2f19',file='minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors',dest='pdd_acc/minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors',size=1658719592,sha256='5531fa0da887c24ec0083b0050ea9e4ff03c479cfbe21ef586d1c5c79bdc78a1')])
SUITE_ASSETS=[a for a in ASSETS if not a.get('vsa') and 'Semantic-Bridge' not in a['repo']]

def suite_required_assets(graph,require_diffusion=True):
    text=json.dumps(graph,sort_keys=True)
    selected=[a for a in SUITE_ASSETS if Path(a.get('dest') or a['file']).name in text]
    if require_diffusion and not any((a.get('dest') or a['file']).startswith('diffusion_models/') for a in selected):raise ValueError('No known H3 diffusion model selected.')
    return selected

def suite_prepare_assets(selected):
    # Only our tracked downloads can be evicted. /kaggle/input and arbitrary user paths are never deleted.
    ledger_file=ROOT/'suite-download-cache.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.is_file() else {}
    globals()['selected_model_assets']=lambda:selected
    try:plan=model_storage_plan()
    except RuntimeError:
        # Adopt only exact pinned notebook cache paths, after verifying their contents.
        stores=([Path(MODEL_STORAGE)] if MODEL_STORAGE!='auto' else [Path('/kaggle/temp/minimax_h3_models'),Path('/tmp/minimax_h3_models'),ROOT/'model-store'])
        for store in stores:
            for asset in SUITE_ASSETS:
                cached=stored_asset(store,asset)
                if asset['sha256'] not in ledger and cached.is_file() and not cached.is_symlink() and file_has_size(cached,asset) and sha256_file(cached)==asset['sha256']:
                    ledger[asset['sha256']]={'path':str(cached),'store':str(store),'sha256':asset['sha256'],'size':asset['size'],'mtime_ns':cached.stat().st_mtime_ns}
        selected_names={a['sha256'] for a in selected}
        for item in list(ledger.values()):
            if item['sha256'] in selected_names:continue
            p=Path(item['path']);allowed=Path(item['store'])
            asset=next((a for a in SUITE_ASSETS if a['sha256']==item['sha256']),None)
            if asset is None or allowed not in stores:continue
            if p.resolve()!=stored_asset(allowed,asset).resolve() or not p.resolve().is_relative_to(allowed.resolve()) or str(p.resolve()).startswith('/kaggle/input/') or p.is_symlink():continue
            if p.is_file() and p.stat().st_size==item['size'] and p.stat().st_mtime_ns==item['mtime_ns']:
                print('Reclaiming notebook-downloaded cache:',p.name,flush=True);p.unlink()
        ledger_file.write_text(json.dumps(ledger,indent=2))
        plan=model_storage_plan()
    from huggingface_hub import hf_hub_download
    for a,source in plan['locations']:
        source.parent.mkdir(parents=True,exist_ok=True)
        dst=asset_target(a);dst.parent.mkdir(parents=True,exist_ok=True)
        if not file_has_size(source,a):
            print('Downloading selected workflow weight:',Path(a['file']).name,flush=True)
            stage=plan['store']/'.staging'/a['sha256'][:16]
            local=Path(hf_hub_download(repo_id=a['repo'],revision=a['revision'],filename=a['file'],local_dir=stage))
            if not file_has_size(local,a) or sha256_file(local)!=a['sha256']:raise RuntimeError('Weight checksum mismatch: '+a['file'])
            local.replace(source);shutil.rmtree(stage)
            ledger[a['sha256']]={'path':str(source),'store':str(plan['store']),'sha256':a['sha256'],'size':a['size'],'mtime_ns':source.stat().st_mtime_ns}
            ledger_file.write_text(json.dumps(ledger,indent=2))
        elif not (a['sha256'] in ledger and ledger[a['sha256']]['path']==str(source) and ledger[a['sha256']]['mtime_ns']==source.stat().st_mtime_ns):
            if sha256_file(source)!=a['sha256']:raise RuntimeError('Existing weight checksum mismatch: '+a['file'])
        if dst!=source:
            if dst.is_symlink():
                if dst.resolve()!=source.resolve():
                    if dst.exists():raise RuntimeError('Refusing to replace existing model link: '+str(dst))
                    dst.unlink()
            if not dst.exists() and not dst.is_symlink():dst.symlink_to(source.resolve())
    globals()['MODELS_READY']=True
