"""Build the standalone Kaggle notebook from reviewable Python sources."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks/h3_complete'

def read(name):
    return (SOURCE/(name+'.py')).read_text()

def code(source, name, hidden=False):
    compile(source, name, 'exec')
    # Cell compilation cannot detect syntax errors inside Python -c payloads.
    # Check the embedded programs after assembling the final cell as well.
    programs = {'probe', 'kernel_probe', 'metadata_probe', 'script', 'code',
                'compat_source', 'checkpoint_source'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in programs and isinstance(node.value.value, str):
                    compile(node.value.value, f'{name}:{target.id}', 'exec')
    return nbformat.v4.new_code_cell(source, id=name, metadata={
        'jupyter':{'source_hidden':hidden}, 'tags':['implementation'] if hidden else []})

def md(source, name):
    return nbformat.v4.new_markdown_cell(source, id=name)

install = 'STACK_READY=False\n'+read('bootstrap')+'\n'+read('stack')
install += '\ncheckpoint_source = '+repr(read('checkpoint_nodes'))+'\n'
install += "checkpoint_plugin=COMFY/'custom_nodes/H3CheckpointRecovery'\ncheckpoint_plugin.mkdir(parents=True,exist_ok=True)\n(checkpoint_plugin/'__init__.py').write_text(checkpoint_source)\nSTACK_READY=True\nprint('Installation and both T4 kernel probes passed.')\n"
# Wrap statements, not text lines: indenting a multiline string changes the
# program passed to Python -c even though the enclosing notebook still compiles.
weight_tree = ast.parse("""def prepare_models():
    global MODELS_READY
    assert STACK_READY, 'The installation stage must complete successfully before rendering.'
    try:
        pass
    except BaseException:
        MODELS_READY=False
        raise

prepare_models()
""")
guard = next(node for node in weight_tree.body[0].body if isinstance(node, ast.Try))
guard.body = ast.parse(read('weights')).body
weights = ast.unparse(ast.fix_missing_locations(weight_tree))+'\n'
# There is exactly one executable notebook cell. A cell's Play button and Run All
# therefore both execute the entire workflow, without relying on frontend queues.
stages = {
    'catalog':read('processes')+'\n'+read('assets'),
    'install':install,
    'weights':weights,
    'workflow':read('graphs')+'\n'+read('runtime'),
    'render':"RESULT_PATH = render_clip()\nprint('Finished:', RESULT_PATH)",
    'result':"assert RESULT_PATH is not None and RESULT_PATH.is_file(), 'Render has not completed successfully.'\nfrom IPython.display import Video, FileLink, display\nexport = Path('/kaggle/working')/RESULT_PATH.name\nif export.resolve() != RESULT_PATH.resolve():\n    shutil.copy2(RESULT_PATH, export)\ndisplay(Video(str(export), embed=True))\ndisplay(FileLink(str(export)))\nprint('Download from Kaggle Output:', export.name)",
}
for name, source in stages.items():
    code(source, name)  # Also validate embedded Python before packaging the stages.
from pprint import pformat
entry = read('settings')
entry += "\n# Embedded runtime below. Edit the settings above; no other cells need to be started.\n"
entry += '_H3_PIPELINE_SOURCES = '+pformat(stages, sort_dicts=False, width=110)+'\n\n'
entry += read('pipeline')+"\nrun_pipeline(_H3_PIPELINE_SOURCES, globals())\n"
nb = nbformat.v4.new_notebook(cells=[
    md('''# MiniMax H3 — one-click Kaggle T4 ×2 notebook

Select **GPU T4 ×2** and **Internet ON**, edit the settings at the top of the single code cell, then press **that cell's ▶ button once**. **Run All** also works. Setup, weight verification, sampling, decoding and preview continue automatically in the same cell. There is no separate step 4 to start.

The default makes one 512 × 768 clip with native audio, four Turbo LoRA steps and 124 frames at 24 fps (about 5.17 seconds). Sampling uses both T4s. Its process exits after saving joint video/audio latents; a fresh process performs native tiled decoding and MP4 export. The finished video's frame count, dimensions and audio are checked.

Rerun this same cell after fixing an error. Existing weights are reused and verified. A valid saved sampling checkpoint skips sampling; an identical completed job reuses its verified MP4. Change the prompt or seed to make another clip. **Save Version** retains working outputs; scratch model weights are temporary.

Stage progress appears below the cell. A genuine failure stops execution and names the failed stage. Model loading and first kernel compilation can pause visible progress before sampling starts.''','intro'),
    code(entry,'run'),
    md('''## Runtime and storage

The embedded runtime uses isolated Python 3.12.12, PyTorch 2.10 / CUDA 13, pinned ComfyUI and the T4 tensor-parallel fork. It checks driver 580 or newer, FP16 operations, transfers between GPUs and native ConvRot INT8 kernels on both T4s before downloading missing weights. Older H3 processes are closed only when their Python executable and ComfyUI directory match this notebook.

The first run downloads roughly 41.38 GiB of pinned weights unless attached or cached. It searches `/kaggle/input`, measures scratch storage and stages downloads on the selected model volume. Symlinks avoid copying weights. Downloading and SHA-256 verification use network/CPU/disk, so low GPU utilization during these stages is expected. No fixed download-time promise is made.

Turbo LoRA is enabled. Semantic Bridge is optional for **FL2VA**, as its author excludes Ref2VA. FastVideo VSA sparse attention is excluded because its kernels are not a validated SM75 path in this stack. Comfy Kitchen attention and INT8 ConvRot are used instead.

Per-stage logs and graphs are in `minimax_h3_dual_t4/runs/`. The MP4 is also copied to the top of `/kaggle/working` and previewed automatically.''','details'),
    md('''## Validation and references

The source revision used for this notebook was checked against the pinned ComfyUI graph validator. CPU tests cover checkpoint persistence, graph execution at the checkpoint boundary, native temporal decode geometry, process cleanup and recovery, and real H.264 MP4 export with audio. **A full model render on Kaggle T4 ×2 has not been executed in the development environment.** The installation stage runs the GPU-specific kernel probes on your actual session before downloading missing weights.

- [ComfyUI v0.37.0](https://github.com/comfyanonymous/ComfyUI/tree/73c9bad4d21e7addbe1d13bc92eee0f1431b017d)
- [T4 tensor-parallel fork](https://github.com/MrKK4/ComfyUI-MultiGPU-Kaggle/tree/e0232dc1d1321e23ef308a0e6a30c39c8e062b46)
- [Pinned MiniMax H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6)
- [Semantic Bridge](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge)
- [Notebook source and checks](https://github.com/bothashaun-maker/Movies/tree/main/notebooks/h3_complete)

The notebook never asks for a Hugging Face token for public models. If an `HF_TOKEN` Kaggle Secret is already attached, it uses it without printing it.''','validation'),
], metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12.12'},
             'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'}})
nbformat.validate(nb)
output=ROOT/'minimax_h3_kaggle_complete.ipynb'
nbformat.write(nb, output)
print(output)
