import os
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler

def run_diffusion_experiment(init_image_path, zero_output_path, output_dir, prompt, device='cuda', prev_image=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device('cpu' if not torch.cuda.is_available() else device)
    torch.hub.set_dir('/scratch0/')

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline='./models/aerialbooth_viewarg',
        cache_dir='dir_name',
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
    ).to(device)

    generator = torch.Generator(device).manual_seed(0)

    if prev_image is not None:
        init_image = prev_image
    else:
        init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

    res = pipe(
        prompt,
        init_image=init_image, 
        generator=generator,
        text_embedding_optimization_steps=1000,
        model_fine_tuning_optimization_steps=500,
    )

    generated_image_path = os.path.join(output_dir, "generated_image.png")
    res.images[0].save(generated_image_path)

    return generated_image_path


