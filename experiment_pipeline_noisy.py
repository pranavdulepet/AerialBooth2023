from train_aerialbooth_no_guidance import run_diffusion_experiment as exp
import os

def main():
    output_dir = "./exp_outs/03052024/"
    view_modes = ['aerial', 'back', 'bottom', 'side']
    prompts = [
        "A coastal lighthouse with a spiral staircase.",
        "A magical forest where animals talk.",
        "A fluffy baby penguin learning to waddle.",
        "A futuristic cityscape with towering buildings.",
        "A bustling city street during a summer festival.",
        "A modern museum with white walls and glass ceilings",
        "A group of goslings following their parent.",
        "A young wizard learning to control their powers.",
        "A scientist working in a laboratory"
    ]

    images = [
        "architectures34",
        "animation3",
        "birdanimal1",
        "architectures23",
        "city27",
        "architectures8",
        "birdanimal38",
        "animation4",
        "human8"
    ]

    for image, prompt in zip(images, prompts):
        init_image_path = f"/gammascratch/pdulepet/AerialBooth2023/dataset/synthetic_sdxl_images/{image}.png"
        for view_mode in view_modes:
            zero_output_path = f"./composite_outputs/{view_mode}_{image}.png"
            print(f"Processing {image} in {view_mode} mode with prompt: {prompt}")
            eval_prompt = f'{view_mode} view, {prompt}'
            
            exp(init_image_path=init_image_path, zero_output_path=zero_output_path, output_dir=output_dir, prompt=eval_prompt, view_mode=view_mode)

if __name__ == "__main__":
    main()

