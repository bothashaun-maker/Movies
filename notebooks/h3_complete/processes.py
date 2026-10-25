import signal

def is_owned_server(pid):
    """Only this notebook's exact Python, ComfyUI directory, and main.py command."""
    proc = Path('/proc')/str(pid)
    try:
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        return (argv[0].decode() == str(PY) and b'main.py' in argv
                and proc.joinpath('cwd').resolve() == COMFY.resolve()
                and proc.joinpath('stat').read_text().split(') ',1)[1][0] != 'Z')
    except (OSError, UnicodeError, IndexError):
        return False

def stop_owned_servers():
    # Also handles the previous notebook's server after a kernel restart.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or not is_owned_server(proc.name):
            continue
        pid = int(proc.name)
        print('Closing previous H3 process:', pid, flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic()+20
            while is_owned_server(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            if is_owned_server(pid):
                os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic()+10
            while is_owned_server(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            if is_owned_server(pid):
                raise RuntimeError(f'H3 process {pid} did not exit; refusing a second GPU process.')
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        except ProcessLookupError:
            pass
