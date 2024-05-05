# from train_aerialbooth_sum import run_diffusion_experiment as exp1
# from train_aerialbooth_avg import run_diffusion_experiment as exp2
# from train_aerialbooth_bias_input import run_diffusion_experiment as exp3
# from train_aerialbooth_bias_z123 import run_diffusion_experiment as exp4
# from train_aerialbooth_noisy_z123 import run_diffusion_experiment as exp5
# from train_aerialbooth_zero_yes_mutual import run_diffusion_experiment as exp
# from train_aerialbooth_exp1 import run_diffusion_experiment as exp

from train_aerialbooth_no_guidance import run_diffusion_experiment as exp
import os

def main():
    
    prompt = "a coastal lighthouse with a spiral staircase."

    init_image_path = "/gammascratch/mukunds/downloads/AerialBooth2023/dataset/synthetic_sdxl_images/architectures34.png"
    output_dir = "./exp_outs/03052024/"

    view_modes=['aerial', 'back', 'bottom', 'side']
    for view_mode in view_modes:
        zero_output_path = f"./zero_outputs/architectures34_{view_mode}.png"
        print(f"Output dir: {output_dir}")
        eval_prompt = f'{view_mode} view, {prompt}'

        exp(init_image_path=init_image_path, zero_output_path=zero_output_path, output_dir=output_dir, prompt=eval_prompt, view_mode=view_mode)

if __name__ == "__main__":
    main()
