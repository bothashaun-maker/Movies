"""ComfyUI entry point for the isolated Kaggle H3 renderer."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
import torch
import folder_paths
import comfy.model_management

_RENDER_LOCK=threading.Lock()

class H3KaggleRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'prompt':('STRING',{'multiline':True,'default':'A cinematic scene. Audio: ambient sound.'}),
            'seed':('INT',{'default':42,'min':0,'max':2**64-1,'control_after_generate':True}),
            'megapixels':('FLOAT',{'default':0.4,'min':0.2,'max':0.4,'step':0.1}),
            'duration':('FLOAT',{'default':5.0,'min':1.0,'max':5.0,'step':1.0}),
            'aspect':(['portrait','landscape','square'],),
        },'optional':{'image':('IMAGE',)}}
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION='render'
    CATEGORY='MiniMax H3/Kaggle'
    DESCRIPTION='Four-step H3 with native audio. Sampling and decoding run in separate GPU processes; the browser server remains running.'

    def render(self,prompt,seed,megapixels,duration,aspect,image=None):
        config=json.loads(Path(__file__).with_name('config.json').read_text())
        root=Path(config['root']);job_id=uuid.uuid4().hex
        jobs=root/'browser-jobs';jobs.mkdir(parents=True,exist_ok=True)
        result=jobs/(job_id+'-result.json');log_path=jobs/(job_id+'.log')
        job=dict(prompt=prompt,seed=seed,megapixels=megapixels,duration=duration,
                 aspect={'portrait':[2,3],'landscape':[3,2],'square':[1,1]}[aspect],
                 image_path='',result=str(result))
        if image is not None:
            from PIL import Image
            if image.shape[0]!=1:raise ValueError('Use one reference image per clip.')
            path=jobs/(job_id+'.png')
            Image.fromarray((image[0].detach().cpu().clamp(0,1)*255).to(torch.uint8).numpy()).save(path)
            job['image_path']=str(path)
        job_path=jobs/(job_id+'.json');job_path.write_text(json.dumps(job))
        with _RENDER_LOCK,log_path.open('w') as log:
            proc=subprocess.Popen([config['python'],config['worker'],str(job_path)],
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    time.sleep(1)
                if proc.returncode:
                    raise RuntimeError('H3 render failed; saved latents are retained. Log: '+str(log_path)+'\n'+log_path.read_text()[-8000:])
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        path=Path(json.loads(result.read_text())['file']).resolve()
        output=Path(folder_paths.get_output_directory()).resolve()
        if not path.is_file() or not path.is_relative_to(output):
            raise RuntimeError('Renderer returned an invalid output file.')
        return {'ui':{'images':[{'filename':path.name,'subfolder':str(path.parent.relative_to(output)),
                                'type':'output','format':'video/mp4'}]}}

NODE_CLASS_MAPPINGS={'H3KaggleRender':H3KaggleRender}
NODE_DISPLAY_NAME_MAPPINGS={'H3KaggleRender':'MiniMax H3 — Kaggle T4'}
