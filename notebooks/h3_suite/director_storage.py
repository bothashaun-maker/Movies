"""One 24 GiB cache reused after each Director GPU process has exited."""
def director_phase_budget(store,selected,ledger,limit=24*2**30):
    active={a['sha256'] for a in selected}
    for a in SUITE_ASSETS:
        if a['sha256'] in active:continue
        path=stored_asset(store,a);item=ledger.get(a['sha256'],{})
        if (item.get('path')==str(path) and item.get('store')==str(store)
            and path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(store.resolve())
            and path.stat().st_size==a['size'] and path.stat().st_mtime_ns==item.get('mtime_ns')):
            print('Releasing previous phase model cache:',path.name,flush=True)
            path.unlink();ledger.pop(a['sha256'],None)
    expected={str(stored_asset(store,a)):a for a in SUITE_ASSETS}
    occupied=sum(Path(p).stat().st_size for p in expected if Path(p).is_file())
    missing=sum(a['size'] for a in selected if not stored_asset(store,a).is_file())
    if occupied+missing>limit:
        raise RuntimeError('Director phase cache would exceed 24 GiB. No new model copy started; untracked files preserved. Use a fresh session.')

def director_prepare_assets(selected):
    globals()['SUITE_STAGE_ALL']=True
    # Reuse the verified sequential copier and the same cache paths; the phase
    # budget replaces the diffusion-only budget for Director alone.
    globals()['suite_checkpoint_budget']=director_phase_budget
    return suite_prepare_assets(selected)
