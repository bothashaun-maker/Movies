with open("run_h3.py", "w", encoding="utf-8") as f:
    f.write(RUN_SCRIPT_CODE)
print("✅ run_h3.py written.")

import json
if os.path.exists("h3_settings.json"):  # Step 1's choices, for a Step 2 re-run after a kernel restart
    with open("h3_settings.json") as f:
        for key, value in json.load(f).items():
            os.environ.setdefault(key, value)

IGNORE = ("absl::InitializeLog", "Unable to register cu", "computation placer already registered", "MessageFactory",
          "GetPrototype", "Attempting to register factory", "UserWarning", "warnings.warn", "FutureWarning")
env = dict(os.environ, PYTHONWARNINGS="ignore", TQDM_DISABLE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

print("🚀 Starting the MiniMax H3 engine...\n")
import psutil

def stop_old_engine():
    stop_camera_engine(_CAMERA_WORK)


stop_old_engine()

# Step 1's downloads leave freed buffers in this kernel process: hand them back to the OS before the engine starts.
import ctypes
import gc
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
_kernel_gb = psutil.Process().memory_info().rss / 1024**3
_ram = psutil.virtual_memory()
print(f"🧠 RAM before the engine starts: {(_ram.total - _ram.available) / 1024**3:.1f} GB in use "
      f"(this notebook kernel: {_kernel_gb:.1f} GB)")
if _kernel_gb > 2.0:
    print("   ⚠️ The kernel itself holds a lot of RAM. If the engine runs out of memory: Run → Restart, then run Step 2 only.")


def _die_with_kernel():
    # Linux: the engine gets SIGKILL when this notebook kernel dies or restarts (no orphan holding the GPUs).
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


proc = subprocess.Popen([sys.executable, "-u", "run_h3.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                        preexec_fn=_die_with_kernel)
remember_camera_engine(_CAMERA_WORK,proc)
import threading
_peak = {"ram": 0.0, "engine": 0.0}


def _watch_ram():  # peak RAM while the engine runs, reported if it dies
    engine = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            vm = psutil.virtual_memory()
            _peak["ram"] = max(_peak["ram"], (vm.total - vm.available) / 1024**3)
            _peak["engine"] = max(_peak["engine"], engine.memory_info().rss / 1024**3)
        except Exception:
            pass
        time.sleep(0.5)


import time
threading.Thread(target=_watch_ram, daemon=True).start()
# Read raw bytes so carriage returns survive: the engine redraws its live progress line in place, and a text-mode
# pipe would turn every redraw into a new line.
decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
pending = ""
try:
    while True:
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            break
        pending += decoder.decode(chunk)
        parts = re.split(r"(?<=[\r\n])", pending)
        pending = parts.pop()  # text after the last line break / carriage return
        out = "".join(part for part in parts if not any(pattern in part for pattern in IGNORE))
        if pending and not any(pattern in pending for pattern in IGNORE):
            out, pending = out + pending, ""  # show the live progress line right away
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
    sys.stdout.write(pending)
    rc = proc.wait()
    if rc != 0:
        import signal
        how = f"killed by {signal.Signals(-rc).name}" if rc < 0 else f"exit code {rc}"
        hint = (" - the system killed it, which almost always means it ran out of RAM" if rc == -9 else
                " - a crash inside a native library (see the traceback above)" if rc in (-11, -6) else "")
        print(f"\n❌ The engine stopped ({how}){hint}.\n   Peak RAM while it ran: {_peak['ram']:.1f} GB in use of "
              f"{psutil.virtual_memory().total / 1024**3:.1f} GB (engine process {_peak['engine']:.1f} GB).")
except KeyboardInterrupt:
    proc.kill()  # frees the GPUs and RAM right away
    proc.wait()
    print("🛑 Engine stopped. Re-run this cell to start it again.")