"""CPU landmark mesh compositor. No identity encoder or learned face-swap weights."""
import os,json
from pathlib import Path
import numpy as np
import torch
import cv2
import folder_paths

def rgb(t):
 if t.ndim!=4 or t.shape[0]!=1:raise ValueError('Use one image per input.')
 return (t[0].cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
def detect(image):
 import mediapipe as mp
 task=Path(os.environ.get('STUDIO_LANDMARK_MODEL',str(Path(folder_paths.models_dir)/'mediapipe/face_landmarker.task')))
 if not task.is_file():raise ValueError('Attach the commercial finish pack: face_landmarker.task is missing.')
 options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),num_faces=10,output_face_blendshapes=False,output_facial_transformation_matrixes=False)
 with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
  result=model.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(image)))
 h,w=image.shape[:2]
 points=[np.array([(min(w-1,max(0,p.x*w)),min(h-1,max(0,p.y*h))) for p in face[:468]],np.float32) for face in result.face_landmarks]
 return sorted(points,key=lambda p:p[:,0].mean())
def composite(source,target,source_index=0,target_index=0,blend_mode="Feather"):
 sp,tp=detect(source),detect(target)
 if not 0<=source_index<len(sp) or not 0<=target_index<len(tp):raise ValueError(f'Faces found: source={len(sp)}, target={len(tp)}; indices start at 0, left to right.')
 a,b=sp[source_index],tp[target_index];h,w=target.shape[:2];warp=np.zeros_like(target)
 # Triangulation derives only from landmark positions; never loads an embedding model.
 hull=cv2.convexHull(b).reshape(-1,2);sub=cv2.Subdiv2D((0,0,w,h))
 for p in b:sub.insert(tuple(map(float,p)))
 for tri in sub.getTriangleList():
  pts=np.array(tri).reshape(3,2)
  if np.any(pts[:,0]<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]<0) or np.any(pts[:,1]>=h):continue
  ids=[int(np.argmin(np.linalg.norm(b-p,axis=1))) for p in pts]
  if len(set(ids))<3:continue
  src=a[ids];dst=b[ids];sx,sy,sw,sh=cv2.boundingRect(src);x,y,tw,th=cv2.boundingRect(dst)
  if min(sw,sh,tw,th)<1:continue
  mat=cv2.getAffineTransform((src-[sx,sy]).astype(np.float32),(dst-[x,y]).astype(np.float32))
  tile=cv2.warpAffine(source[sy:sy+sh,sx:sx+sw],mat,(tw,th),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
  mask=np.zeros((th,tw),np.uint8);cv2.fillConvexPoly(mask,np.int32(dst-[x,y]),255)
  view=warp[y:y+th,x:x+tw];view[mask>0]=tile[mask>0]
 mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.int32(hull),255)
 # Keep edge inside the face hull and use Poisson blending for lighting continuity.
 mask=cv2.erode(mask,np.ones((3,3),np.uint8))
 x,y,bw,bh=cv2.boundingRect(mask)
 if not np.any(mask):raise ValueError('Face is too small to composite.')
 if blend_mode=='Poisson':
  out=cv2.seamlessClone(warp,target,mask.copy(),(x+bw//2,y+bh//2),cv2.NORMAL_CLONE)
 else:
  # Preserve the transferred features; Poisson blending can weaken identity contrast.
  alpha=cv2.distanceTransform(mask,cv2.DIST_L2,3);alpha=np.clip(alpha/max(2,min(bw,bh)*0.04),0,1)[...,None]
  out=(warp*alpha+target*(1-alpha)).round().astype(np.uint8)
 out[mask==0]=target[mask==0]
 return out,dict(method='mediapipe-mesh-pixel-composite',source_faces=len(sp),target_faces=len(tp),source_index=source_index,target_index=target_index,blend_mode=blend_mode,limitations='Use similar pose/expression; this does not synthesize an identity or guarantee likeness.')
class CommercialMeshSwap:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'source':('IMAGE',),'target':('IMAGE',),'source_index':('INT',{'default':0,'min':0,'max':9}),'target_index':('INT',{'default':0,'min':0,'max':9}),'blend_mode':(['Feather','Poisson'],)}}
 RETURN_TYPES=('IMAGE','STRING');RETURN_NAMES=('image','report');FUNCTION='swap';CATEGORY='Image Studio Commercial/Face'
 def swap(self,source,target,source_index,target_index,blend_mode="Feather"):
  out,report=composite(rgb(source),rgb(target),source_index,target_index,blend_mode)
  return torch.from_numpy(out.astype(np.float32)/255).unsqueeze(0),json.dumps(report)
class CommercialMaskedEdit:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'original':('IMAGE',),'edited':('IMAGE',),'mask':('MASK',),'feather':('INT',{'default':12,'min':0,'max':128})}}
 RETURN_TYPES=('IMAGE',);FUNCTION='paste';CATEGORY='Image Studio Commercial/Edit'
 def paste(self,original,edited,mask,feather):
  a,b=rgb(original),rgb(edited);h,w=a.shape[:2];b=cv2.resize(b,(w,h),interpolation=cv2.INTER_LANCZOS4)
  m=mask[0].cpu().numpy();m=cv2.resize(m,(w,h));m=np.clip(m,0,1)
  if not m.any():raise ValueError('Paint a mask in LoadImage → MaskEditor first (white = replace).')
  if feather:m=cv2.GaussianBlur(m,(0,0),feather/3)
  out=a*(1-m[...,None])+b*m[...,None]
  return (torch.from_numpy(out.astype(np.float32)/255).unsqueeze(0),)
NODE_CLASS_MAPPINGS={c.__name__:c for c in [CommercialMeshSwap,CommercialMaskedEdit]}
