"""Visible InsightFace detection and swapping stages for Image Studio."""
from pathlib import Path
from functools import lru_cache
import os
import numpy as np
import torch
import folder_paths

ROOT=Path(folder_paths.models_dir)/'insightface'

def image_bgr(image):
 if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]!=3:
  raise ValueError('Use one RGB source photo and one RGB target photo, not image batches.')
 return np.ascontiguousarray((image[0].detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)[:,:,::-1])

@lru_cache(maxsize=1)
def analyser():
 from insightface.app import FaceAnalysis
 import onnxruntime as ort
 models=ROOT/'models/buffalo_l'
 for name in ['det_10g.onnx','w600k_r50.onnx']:
  if not (models/name).is_file():raise RuntimeError('Attach the face-swap model pack and rerun studio startup; missing '+name)
 options=ort.SessionOptions();options.intra_op_num_threads=min(4,os.cpu_count() or 1)
 app=FaceAnalysis(name='buffalo_l',root=str(ROOT),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'],sess_options=options)
 app.prepare(ctx_id=-1,det_size=(640,640))
 return app

@lru_cache(maxsize=1)
def swapper():
 from insightface.model_zoo import get_model
 import onnxruntime as ort
 path=ROOT/'inswapper_128.onnx'
 if not path.is_file():raise RuntimeError('Missing inswapper_128.onnx; attach the face-swap model pack.')
 options=ort.SessionOptions();options.intra_op_num_threads=min(4,os.cpu_count() or 1)
 return get_model(str(path),providers=['CPUExecutionProvider'],sess_options=options)

class StudioFaceDetect:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',)}}
 RETURN_TYPES=('STUDIO_FACES','STRING')
 RETURN_NAMES=('faces','detected_faces')
 FUNCTION='detect';CATEGORY='Image Studio/Face Swap'
 def detect(self,image):
  bgr=image_bgr(image)
  faces=sorted(analyser().get(bgr),key=lambda f:float((f.bbox[0]+f.bbox[2])/2))
  if not faces:raise ValueError('No face detected. Use a clear, larger, unobstructed face photo.')
  descriptions=[f'{i}: box '+','.join(str(int(v)) for v in f.bbox) for i,f in enumerate(faces)]
  data={'faces':faces,'image':bgr,'shape':bgr.shape}
  return data,'Faces numbered left to right, starting at 0. '+ '; '.join(descriptions)

class StudioFaceSwap:
 @classmethod
 def INPUT_TYPES(cls):
  return {'required':{'source_faces':('STUDIO_FACES',),'target_faces':('STUDIO_FACES',),'source_index':('INT',{'default':0,'min':0,'max':99}),'target_index':('INT',{'default':0,'min':0,'max':99})}}
 RETURN_TYPES=('IMAGE',);FUNCTION='swap';CATEGORY='Image Studio/Face Swap'
 def swap(self,source_faces,target_faces,source_index,target_index):
  if not 0<=source_index<len(source_faces['faces']):raise ValueError('Source face index is outside the detected face list.')
  if not 0<=target_index<len(target_faces['faces']):raise ValueError('Target face index is outside the detected face list.')
  result=swapper().get(target_faces['image'].copy(),target_faces['faces'][target_index],source_faces['faces'][source_index],paste_back=True)
  if result.shape!=target_faces['shape'] or not np.isfinite(result).all():raise RuntimeError('Invalid face-swap output.')
  rgb=np.ascontiguousarray(result[:,:,::-1]);return (torch.from_numpy(rgb.astype(np.float32)/255.0).unsqueeze(0),)

NODE_CLASS_MAPPINGS={'StudioFaceDetect':StudioFaceDetect,'StudioFaceSwap':StudioFaceSwap}
NODE_DISPLAY_NAME_MAPPINGS={'StudioFaceDetect':'Detect Faces — Image Studio','StudioFaceSwap':'Swap Selected Face — Image Studio'}
