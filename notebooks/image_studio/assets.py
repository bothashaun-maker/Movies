from pathlib import Path
import hashlib
MODEL='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'
MODEL_SHA='c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca'
MODEL_SIZE=7105348188
def model_assets(load_sdxl=True, enable_seedvr2=True, enable_flux3=False):
 assets=[('upscale_models','RealESRGAN_x4plus.pth','https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth','4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',67040989)]
 if load_sdxl:assets.append(('checkpoints',MODEL,'https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/'+MODEL,MODEL_SHA,MODEL_SIZE))
 if enable_seedvr2:
  prefix='https://huggingface.co/Comfy-Org/SeedVR2/resolve/df48879708206a403d2a61acd55578c2e80fd233/'
  assets.extend([
   ('diffusion_models','seedvr2_3b_int8_convrot.safetensors',prefix+'diffusion_models/seedvr2_3b_int8_convrot.safetensors','c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157',3458259704),
   ('vae','seedvr2_ema_vae_fp16.safetensors',prefix+'vae/seedvr2_ema_vae_fp16.safetensors','20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1',501324814)])
 if enable_flux3:
  prefix='https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/9d36114cf7a2e03a45c28f5cc4987d20ebfdb6c1/'
  assets.extend([
   ('diffusion_models','flux3-image-klein-int8-convrot.safetensors',prefix+'diffusion_models/flux3-image-klein-int8-convrot.safetensors','847d8933548afd93df4af2cac2498e70891bf28d9dae06d695f536992e168fe2',5827874779),
   ('text_encoders','qwen3vl_4b_int8_convrot.safetensors',prefix+'text_encoders/qwen3vl_4b_int8_convrot.safetensors','32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c',4858559240),
   ('vae','flux2-vae.safetensors',prefix+'vae/flux2-vae.safetensors','868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3',336211292),
   ('text_encoders','qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors',prefix+'text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors','e5d85a7421d197bf5a12474511936ecabce19a833252f16e9310a983194a4458',6724090780)])
 return assets
