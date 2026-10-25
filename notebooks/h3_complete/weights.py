MODELS_READY=False
plan=model_storage_plan()
selected_assets=selected_model_assets()
pending=plan['pending']
# Keep downloads and staging on the selected volume; links expose them to ComfyUI.
for a,source in plan['locations']:
    dst=asset_target(a)
    dst.parent.mkdir(parents=True,exist_ok=True)
    source.parent.mkdir(parents=True,exist_ok=True)
    if dst==source:continue
    if dst.is_symlink():
        if dst.resolve()==source.resolve():continue
        if dst.exists():raise RuntimeError('Existing model link points elsewhere: '+str(dst))
        dst.unlink()  # Only replace a broken link, preserving regular files.
    if dst.exists():raise RuntimeError('Existing model file has an unexpected size: '+str(dst))
    dst.symlink_to(source.resolve())

download_env=child_env.copy()
if pending:
    try:
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret('HF_TOKEN')
        if token:download_env['HF_TOKEN']=token
    except Exception:pass  # Selected repositories are public.
verified_downloads=set()
for a,source in plan['locations']:
    if a not in pending:continue
    # Same filesystem as final weights: rename does not copy a 20 GiB checkpoint.
    stage=plan['store']/'.staging'/hashlib.sha256((a['repo']+a['file']).encode()).hexdigest()[:12]
    script=('from huggingface_hub import hf_hub_download; import sys; '
            'print(hf_hub_download(repo_id=sys.argv[1],revision=sys.argv[2],'
            'filename=sys.argv[3],local_dir=sys.argv[4]))')
    started=time.monotonic()
    print('Downloading:',Path(a['file']).name,flush=True)
    run([PY,'-c',script,a['repo'],a['revision'],a['file'],stage],env=download_env)
    print(f'Download finished in {(time.monotonic()-started)/60:.1f} min; verifying SHA-256...',flush=True)
    staged=stage/a['file']
    assert file_has_size(staged,a) and sha256_file(staged)==a['sha256'],'Downloaded artifact checksum mismatch.'
    staged.replace(source)
    verified_downloads.add(source.resolve())
    shutil.rmtree(stage)
for a in selected_assets:
    p=asset_target(a)
    assert file_has_size(p,a) and (p.resolve() in verified_downloads or sha256_file(p)==a['sha256']),f'Invalid model file: {p}'
    print('Verified:',p.name)
(ROOT/'model-manifest.json').write_text(json.dumps(selected_assets,indent=2))
print('Weight hashes verified; checking H3 quantization metadata...',flush=True)

# Inspect the real checkpoint's quantization metadata before GPU model loading.
metadata_probe="""import sys
from safetensors import safe_open
sys.path.insert(0,sys.argv[1])
from h3_quant_compat import validate_h3_quant_layers
with safe_open(sys.argv[2],framework='pt',device='cpu') as weights:
 layers=validate_h3_quant_layers(weights)
 print('H3 INT8 ConvRot metadata validated:',len(layers),'layers')
"""
unet=next(a for a in selected_model_assets() if a['file'].startswith('diffusion_models/'))
run([PY,'-c',metadata_probe,MGPU,asset_target(unet)])

if USE_BRIDGE:
    import zipfile
    archive = next(asset_target(a) for a in selected_assets if a['file'].endswith('.zip'))
    target = COMFY/'custom_nodes'
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            resolved = (target/member.filename).resolve()
            assert resolved.is_relative_to(target.resolve()), 'Unsafe ZIP path.'
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink refused.'
        folder = target/'MiniMax_H3_Semantic_Bridge'
        if folder.exists():
            for member in z.infolist():
                if not member.is_dir():
                    assert (target/member.filename).read_bytes() == z.read(member), 'Existing bridge differs; use a fresh ROOT.'
        else:
            z.extractall(target)
    assert (folder/'__init__.py').is_file(), 'Unexpected bridge ZIP layout.'

MODELS_READY=True
print('All weights and metadata verified; rendering is enabled.')
