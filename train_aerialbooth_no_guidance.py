import os
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler

def run_diffusion_experiment(init_image_path, output_dir, prompt, zero_output_path, view_mode, device='cuda'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else device)
    torch.hub.set_dir('/scratch0/')
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline='./models/aerialbooth_noisy',
        cache_dir='dir_name',
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)

    generator = torch.Generator(device).manual_seed(0)

    init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))
    image_hom = Image.open(zero_output_path).convert("RGB").resize((512, 512))
    
    # res = pipe.train(
    #     prompt,
    #     image=init_image,
    #     generator=generator, 
    #     text_embedding_optimization_steps=1000,
    #     model_fine_tuning_optimization_steps=500
    #     # image_hom=image_hom
    # )
    res = pipe.train(
        prompt,
        image=image_hom,
        generator=generator, 
        text_embedding_optimization_steps=1000,
        model_fine_tuning_optimization_steps=500,
        image_hom=init_image
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
        # eval_prompt = f'{view_mode} view, {prompt}'
        # print(f"prompt: {eval_prompt}")
        # for i in range(5):  
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0, eval_prompt=prompt, image_hom=image_hom)
    image = res.images[0]
    image_basename = os.path.basename(init_image_path).split('.')[0]
    image.save(f'{output_dir}/{view_mode}_{image_basename}.png')

