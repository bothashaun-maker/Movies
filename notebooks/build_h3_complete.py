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
    assert STACK_READY, 'Run the installation cell successfully first.'
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
nb = nbformat.v4.new_notebook(cells=[
    md('''# MiniMax H3 — complete Kaggle T4 ×2 notebook

Select **GPU T4 ×2** and **Internet ON**, edit the settings below, then **Run All**. This is a complete notebook; no GitHub helper-script downloads or pasted repair cells are needed.

The default makes one 512 × 768 clip with native audio, four Turbo LoRA steps and 124 frames at 24 fps (about 5.17 seconds). Sampling uses both T4s. The notebook saves the joint video/audio latents, exits that process, and starts a fresh process for native tiled decoding and MP4 export. It verifies the finished video's frame count, dimensions and audio stream.

Rerunning the render cell reuses a valid checkpoint after a decode failure. Rerunning an identical completed job reuses its verified MP4. Change the prompt or seed to make another clip. Checkpoints and results survive within the session; **Save Version** preserves working outputs. Scratch model weights are temporary.

The first run downloads roughly 41.38 GiB of pinned weights unless attached or cached. Downloads and SHA-256 checks use network/CPU/disk; GPU utilization during those stages is expected to be low. No fixed download-time promise is made.''','intro'),
    md('## 1. Settings','settings-title'),
    code(read('settings'),'settings'),
    md('## 2. Runtime helpers and pinned model catalog\nThis cell is self-contained and contains the original supplied workflow for reference. Normally no edits are needed.','helpers-title'),
    code(read('processes')+'\n'+read('assets'),'catalog',True),
    md('''## 3. Install and test the T4 runtime

Checks both GPUs, driver and measured storage before large downloads. Uses isolated Python 3.12.12, PyTorch 2.10 / CUDA 13, pinned ComfyUI and the T4 tensor-parallel fork. It tests FP16 CUDA operations, transfer between GPUs, and native ConvRot INT8 kernels on each T4. CUDA 13 requires driver 580 or newer. Existing matching packages and source checkouts are reused.

This cell closes an older H3 server only when its Python executable and ComfyUI directory exactly match this notebook.''','install-title'),
    code(install,'install',True),
    md('''## 4. Prepare and verify weights

Automatically searches attached `/kaggle/input` datasets and existing model files. Measures `/kaggle/temp`, `/tmp` and working storage without counting one filesystem twice. Downloads stage on the selected model volume; symlinks expose the files to ComfyUI without copying them. Existing files are SHA-256 checked; new downloads are checked once. Interrupted Hugging Face downloads can resume.

Turbo LoRA is enabled. Semantic Bridge is optional for **FL2VA**, as its author excludes Ref2VA. The FastVideo VSA sparse-attention path is not enabled on T4: its required kernels are not a validated SM75 path in this stack. The notebook uses Comfy Kitchen attention and INT8 ConvRot instead.''','weights-title'),
    code(weights,'weights',True),
    md('## 5. Load the render workflow\nDefines the sampler, checkpoint, clean-process decoder, progress reporting and output verification.','workflow-title'),
    code(read('graphs')+'\n'+read('runtime'),'workflow',True),
    md('''## 6. Render one clip

To retry a failed decode, rerun **this cell**. A matching saved checkpoint skips sampling automatically. Both processes close on success, error or notebook interruption. Progress is printed every 30 seconds; full per-stage logs and graphs are in `minimax_h3_dual_t4/runs/`.

Model loading and the first kernel compilation can pause visible progress before GPU sampling starts.''','render-title'),
    code("RESULT_PATH = None\nRESULT_PATH = render_clip()\nprint('Finished:', RESULT_PATH)",'render'),
    md('## 7. Preview and download\nThe MP4 and matching latent checkpoint remain under `/kaggle/working`. Save the notebook version to retain those outputs.','result-title'),
    code("assert RESULT_PATH is not None and RESULT_PATH.is_file(), 'Render has not completed successfully.'\nfrom IPython.display import Video, FileLink, display\nexport = Path('/kaggle/working')/RESULT_PATH.name\nif export.resolve() != RESULT_PATH.resolve():\n    shutil.copy2(RESULT_PATH, export)\ndisplay(Video(str(export), embed=True))\ndisplay(FileLink(str(export)))\nprint('Download from Kaggle Output:', export.name)",'result'),
    md('''## Validation and references

The source revision used for this notebook was checked against the pinned ComfyUI graph validator. CPU tests cover checkpoint persistence, graph execution at the checkpoint boundary, native temporal decode geometry, process cleanup and recovery, and real H.264 MP4 export with audio. **A full model render on Kaggle T4 ×2 has not been executed in the development environment.** The installation cell runs the GPU-specific kernel probes on your actual session before downloading missing weights.

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
