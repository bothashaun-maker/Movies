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

class H3IsolatedSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'video':('VIDEO',{'lazy':True}),
                            'filename_prefix':('STRING',{'default':'kaggle_h3/clip'})},
                'hidden':{'prompt':'PROMPT','unique_id':'UNIQUE_ID'}}
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION='render'
    CATEGORY='MiniMax H3/Kaggle'
    DESCRIPTION='Executes the connected native graph in isolated sampling and decode processes. Keeps the browser server free of model weights.'

    def check_lazy_status(self,video=None,filename_prefix=None,prompt=None,unique_id=None):
        return []  # Dependencies execute in GPU workers, never in this UI server.

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        return float('nan')  # Worker fingerprint controls checkpoint/result reuse.

    def render(self,filename_prefix,video=None,prompt=None,unique_id=None):
        config=json.loads(Path(__file__).with_name('config.json').read_text())
        root=Path(config['root']);job_id=uuid.uuid4().hex
        jobs=root/'browser-jobs';jobs.mkdir(parents=True,exist_ok=True)
        result=jobs/(job_id+'-result.json');log_path=jobs/(job_id+'.log')
        job=dict(graph=prompt,output_id=str(unique_id),result=str(result))
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

NODE_CLASS_MAPPINGS={'H3IsolatedSaveVideo':H3IsolatedSaveVideo}
NODE_DISPLAY_NAME_MAPPINGS={'H3IsolatedSaveVideo':'Save MP4 — isolated H3 render'}
