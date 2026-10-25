stop_owned_servers()
MODELS_READY=False
# Check scratch capacity before spending time on CUDA/package installation.
package_probe=None
if PY.is_file():
    try:
        package_probe=subprocess.run([str(PY),'-I','-c',
            "from importlib.metadata import version; assert version('torch')=='2.10.0+cu130'; "
            "assert version('comfy-kitchen')=='0.2.35'; assert version('comfy-aimdo')=='0.5.5'"],
            capture_output=True,env=child_env,timeout=30)
    except (OSError,subprocess.TimeoutExpired):pass
package_reserve=0 if package_probe and package_probe.returncode==0 else 6*2**30
model_storage_plan(package_reserve=package_reserve)

def run(args,**kwargs):
    kwargs.setdefault('env',child_env)
    return subprocess.run([str(a) for a in args],check=True,**kwargs)

p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
    '--format=csv,noheader,nounits'],capture_output=True,text=True)
assert p.returncode==0,'Enable GPU T4 x2 in Kaggle.'
rows=[r for r in p.stdout.splitlines() if r.strip()]
assert len(rows)==2 and all('T4' in r for r in rows),'This notebook requires exactly two NVIDIA T4 GPUs.'
assert all(int(r.split(',')[-1].strip().split('.')[0])>=580 for r in rows),'CUDA 13 requires host driver 580+; select a compatible Kaggle session.'
print(p.stdout.strip())
ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))/2**20
assert ram>=24,f'At least 24 GiB host RAM required; detected {ram:.1f}.'
def ensure_environment():
    """Create a seeded Python 3.12 environment without host pip/ensurepip."""
    import io, zipfile
    expected_python='3.12.12'
    uv_version='0.9.26'
    uv_sha256='b7e89798bd3df7dcc4b2b4ac4e2fc11d6b3ff4fe7d764aa3012d664c635e2922'
    uv_url=('https://files.pythonhosted.org/packages/38/16/'
            'a07593a040fe6403c36f3b0a99b309f295cbfe19a1074dbadb671d5d4ef7/'
            'uv-0.9.26-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
    assert os.uname().machine=='x86_64','This pinned bootstrap requires Linux x86_64.'
    # Notebook kernels may themselves run inside a different virtual environment.
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','UV_SYSTEM_PYTHON'):
        child_env.pop(key,None)
    child_env.update(UV_PYTHON_INSTALL_DIR=str(ROOT/'python'),
                     UV_CACHE_DIR=str(ROOT/'uv-cache'),UV_LINK_MODE='copy',
                     UV_PYTHON_BIN_DIR=str(ROOT/'bootstrap/bin'),
                     UV_PYTHON_DOWNLOADS='automatic')
    health_code=("import sys,pip; from pathlib import Path; "
                 f"assert sys.version.split()[0]=={expected_python!r}; "
                 f"assert Path(sys.prefix).resolve()==Path({str(ENV)!r}).resolve(); "
                 "assert sys.prefix!=sys.base_prefix; print(sys.version.split()[0])")
    def healthy():
        if not PY.is_file():return False
        try:
            return subprocess.run([str(PY),'-I','-c',health_code],env=child_env,
                capture_output=True,text=True,timeout=30).returncode==0
        except (OSError,subprocess.TimeoutExpired):return False
    if healthy():
        print('Reusing verified Python',expected_python,'environment:',ENV)
        return
    assert shutil.disk_usage(ROOT).free>10*2**30,'Free at least 10 GiB for packages.'
    uv=ROOT/'bootstrap'/f'uv-{uv_version}'
    # Always check the wheel digest before executing its binary.
    with urllib.request.urlopen(uv_url,timeout=120) as response:wheel=response.read()
    assert hashlib.sha256(wheel).hexdigest()==uv_sha256,'uv download checksum mismatch.'
    uv.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        binary=archive.read(f'uv-{uv_version}.data/scripts/uv')
    uv.write_bytes(binary);uv.chmod(0o755)
    run([uv,'--version'])
    # Install Python first; a failed network request leaves the old environment intact.
    run([uv,'python','install',expected_python])
    if ENV.exists() or ENV.is_symlink():
        backup=ENV.with_name(ENV.name+'.incomplete-'+uuid.uuid4().hex[:8])
        ENV.rename(backup)
        print('Preserved incomplete or incompatible environment at',backup)
    run([uv,'venv','--python',expected_python,'--managed-python','--seed',ENV])
    assert healthy(),'Python/pip environment validation failed; inspect bootstrap output above.'
    print('Verified isolated Python',expected_python,'with pip; host ensurepip is not used.')

ensure_environment()

run([PY,'-m','pip','install','pip==25.3'])
run([PY,'-m','pip','install','--index-url','https://download.pytorch.org/whl/cu130',
    'torch==2.10.0','torchvision==0.25.0','torchaudio==2.10.0'])
probe="""import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert torch.version.cuda.startswith('13.')
assert 'sm_75' in torch.cuda.get_arch_list() or 'compute_75' in torch.cuda.get_arch_list()
for i in range(2):
 assert torch.cuda.get_device_capability(i)==(7,5)
 x=torch.ones((64,64),device=f'cuda:{i}',dtype=torch.float16)
 assert (x@x).float().mean().item()==64
 print(i,torch.cuda.get_device_name(i),'FP16 CUDA matmul passed')
# Explicit host-staged transfer: no NVLink/P2P assumption.
src=torch.arange(256,device='cuda:0',dtype=torch.float32)
host=torch.empty(256,pin_memory=True)
host.copy_(src);dst=host.to('cuda:1')
assert torch.equal(dst.cpu(),torch.arange(256,dtype=torch.float32))
print('Two-GPU CUDA and host-staged transfer probes passed')
"""
run([PY,'-c',probe])
