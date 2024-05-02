# from train_aerialbooth_sum import run_diffusion_experiment as exp1
# from train_aerialbooth_avg import run_diffusion_experiment as exp2
# from train_aerialbooth_bias_input import run_diffusion_experiment as exp3
# from train_aerialbooth_bias_z123 import run_diffusion_experiment as exp4
# from train_aerialbooth_noisy_z123 import run_diffusion_experiment as exp5
# from train_aerialbooth_zero_yes_mutual import run_diffusion_experiment as exp
# from train_aerialbooth_exp1 import run_diffusion_experiment as exp

from train_aerialbooth_no_guidance_noisy import run_diffusion_experiment as exp
import os

def main():
    
    prompt = "a coastal lighthouse with a spiral staircase."

    init_image_path = "/gammascratch/pdulepet/AerialBooth2023/dataset/synthetic_sdxl_images/architectures34.png"
    output_dir = "./exp_outs/sum_grid_idk/"
    zero_output_path = f"./zero_outputs/output_architectures34.png"
    print(f"Output dir: {output_dir}")

    exp(init_image_path=init_image_path, zero_output_path=zero_output_path, output_dir=output_dir, prompt=prompt)

if __name__ == "__main__":
    main()
