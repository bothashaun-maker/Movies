"""Bound the optional single-GPU diagnostic beside resident two-GPU shards."""
from pathlib import Path

MARKER = '# H3_NEXT_BOUNDED_TP_CHECK_V1'
OLD = '''        ref = None
        if not self.checked:
            ref = original({**args, "img": args["img"].clone()})["img"].float()
'''
NEW = '''        ref = None
        # H3_NEXT_BOUNDED_TP_CHECK_V1
        # This diagnostic runs the unsplit block alongside both resident shards.
        # A full video sequence can OOM before the actual TP computation begins.
        # Keep small comparisons; never run this extra block above 4096 tokens.
        if not self.checked and args["img"].shape[0] > 4096:
            self.checked = True
            logger.info("[H3 Next] single-GPU comparison skipped for %d tokens (4096-token diagnostic cap); production TP remains enabled", args["img"].shape[0])
        if not self.checked:
            ref = original({**args, "img": args["img"].clone()})["img"].float()
'''

def patch_tp_check(root):
    path = Path(root)/'h3_tensor_parallel.py'
    source = path.read_text()
    if MARKER in source:
        if source.count(NEW) != 1:
            raise RuntimeError('Unexpected existing H3 Next TP comparison patch.')
        return
    if source.count(OLD) != 1:
        raise RuntimeError('Pinned TP comparison source changed; refusing an ambiguous patch.')
    source = source.replace(OLD, NEW)
    compile(source, str(path), 'exec')
    path.write_text(source)
