import os
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler

def run_diffusion_experiment(init_image_path, zero_output_path, output_dir, prompt, device='cuda'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else device)
    torch.hub.set_dir('/scratch0/')
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline='./models/aerialbooth_avg',
        cache_dir='dir_name',
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
    ).to(device)

    generator = torch.Generator(device).manual_seed(0)

    init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))
    # image_hom = Image.open(zero_output_path).convert("RGB").resize((512, 512))
    
    main_image = Image.open(zero_output_path).convert("RGB")

    num_columns = 2
    num_rows = 3
    sub_image_width = main_image.width // num_columns
    sub_image_height = main_image.height // num_rows

    sub_images = []
    
    for row in range(num_rows):
        for col in range(num_columns):
            left = col * sub_image_width
            upper = row * sub_image_height
            right = left + sub_image_width
            lower = upper + sub_image_height

            sub_image = main_image.crop((left, upper, right, lower))
            sub_images.append(sub_image)

    aerial_hom = sub_images[0]
    bottom_hom = sub_images[1]
    side_hom = sub_images[2]
    back_hom = sub_images[3]
    
    os.makedirs(output_dir, exist_ok=True)
    view_modes = ['bottom', 'side', 'back', 'aerial']
    
    for view_mode in view_modes:
        if view_mode == 'bottom':
            image_hom = bottom_hom
        elif view_mode == 'aerial':
            image_hom = aerial_hom
        elif view_mode == 'side':
            image_hom = side_hom
        else:
            image_hom = back_hom
        
        eval_prompt = f"{view_mode} view, {prompt}"

        res = pipe.train(
            prompt,
            image=init_image,
            generator=generator, 
            text_embedding_optimization_steps=1000,
            model_fine_tuning_optimization_steps=500, 
            image_hom=image_hom
        )

        image_hom = image_hom.resize(512,512)

        for i in range(5):  
            res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt=eval_prompt)
            image = res.images[0]
            image.save(f'{output_dir}/{view_mode}{i+1}.png')

