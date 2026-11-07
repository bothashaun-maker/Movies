"""Optional CPU integration test with real weights and public-domain reference photos."""
from pathlib import Path
import sys,os,hashlib,importlib.util
import numpy as np
from PIL import Image
import torch
R=Path(__file__).parent
sys.path.insert(0,os.environ.get('COMFY_ROOT',str(R.parent/'ComfyUI')))
spec=importlib.util.spec_from_file_location('studio_faces',R/'face_nodes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for name,expected in [('inswapper_128.onnx','e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af')]:
 p=m.ROOT/name;h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected
source=Image.open(R.parent/'astronaut.png').convert('RGB');target=Image.open(R.parent/'grace_hopper.jpg').convert('RGB')
def tensor(im):return torch.from_numpy(np.asarray(im).copy().astype(np.float32)/255).unsqueeze(0)
d=m.StudioFaceDetect();a,desc=d.detect(tensor(source));b,desc2=d.detect(tensor(target));print(desc);print(desc2)
out=m.StudioFaceSwap().swap(a,b,0,0)[0]
assert out.shape==tensor(target).shape and torch.isfinite(out).all()
assert not torch.equal(out,tensor(target))
Image.fromarray((out[0].numpy()*255).round().astype('uint8')).save(R.parent/'face-swap-cpu-smoke.png')
try:d.detect(torch.zeros((1,128,128,3)));raise AssertionError('Blank image accepted')
except ValueError as e:assert 'No face' in str(e)
try:m.StudioFaceSwap().swap(a,b,99,0);raise AssertionError('Invalid source index accepted')
except ValueError:pass
print('Real face detection/swap, dimensions, finite values, no-face and invalid-index checks passed on CPU.')
