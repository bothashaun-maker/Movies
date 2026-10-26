"""Bounded text report for this notebook's render logs; no model execution."""
def render_diagnostics(root):
    root = Path(root).resolve()
    sections = ['H3 render diagnostics — recent controller and GPU-worker logs']
    paths = sorted((root/'browser-jobs').glob('*.log'), key=lambda p:p.stat().st_mtime, reverse=True)[:2]
    runs = sorted((root/'runs').glob('h3_*'), key=lambda p:max([p.stat().st_mtime]+[x.stat().st_mtime for x in p.glob('*.log')]), reverse=True)[:2]
    for directory in runs:
        paths.extend(directory/name for name in ('sampling.log','decoding.log','last-error.txt','result.json'))
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-24000))
            text=stream.read(24000).decode(errors='replace')
        sections.append('\n--- '+str(path.relative_to(root))+' (last 24 KB) ---\n'+text)
    if len(sections)==1:
        sections.append('No render logs yet in '+str(root))
    return '\n'.join(sections)
