"""Retain root exceptions even when a long traceback exceeds the UI tail."""
def next_failure_summary(path):
    import re
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        text = stream.read().decode(errors='replace')
    lines = text.splitlines()
    details = []
    for i, line in enumerate(lines):
        if re.match(r'^(?:torch\.)?(?:OutOfMemoryError|RuntimeError|ValueError|TimeoutError):', line):
            details.extend(lines[i:i+7])
    return '\n'.join(details)[-6000:]+'\n--- traceback tail ---\n'+text[-6000:]
