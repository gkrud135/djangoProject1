import sys
sys.path.append('clipseg')
# sys.path.append('C:/Users/gkrud/PycharmProjects/djangoProject1/clipseg')

import torch
from torchvision import transforms
from PIL import Image
from clipseg.models.clipseg import CLIPDensePredT
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
import io
def background_inpainting(image_path):
    # 모델 설정
    model_path = 'weights/clipseg_weights/rd64-uni.pth'
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
    
    # Stable Diffusion 인페인팅 파이프라인 설정
    model_dir = "stabilityai/stable-diffusion-2-inpainting"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    # 이미지 로드 및 전처리
    if isinstance(image_path, io.BytesIO):
        image_path.seek(0)  # 스트림의 시작 부분으로 포인터 이동
        source_image = Image.open(image_path).convert('RGB')  # BytesIO를 PIL Image 객체로 변환 및 RGB로 변환
    elif isinstance(image_path, str):
        source_image = Image.open(image_path).convert('RGB')  # 파일 경로에서 이미지 로드 및 RGB로 변환
    elif isinstance(image_path, Image.Image):
        source_image = image_path.convert('RGB')  # PIL Image 객체를 RGB로 변환

    tensor_image = transforms.ToTensor()(source_image).unsqueeze(0)

    # ClipSeg를 사용하여 'sky' 세그멘테이션 후 마스크 생성
    clipseg_prompt = 'sky'
    with torch.no_grad():
        preds = model(tensor_image, [clipseg_prompt])[0]
    processed_mask = torch.special.ndtr(preds[0][0])
    stable_diffusion_mask = transforms.ToPILImage()(processed_mask)

    # 'dark and cloudy sky'로 인페인팅 수행
    inpainting_prompt = "dark and cloudy sky"
    generator = torch.Generator(device="cuda").manual_seed(77)
    result_image = pipe(prompt=inpainting_prompt, guidance_scale=11, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]

    return result_image

