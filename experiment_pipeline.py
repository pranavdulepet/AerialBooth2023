# from train_aerialbooth_sum import run_diffusion_experiment as exp1
# from train_aerialbooth_avg import run_diffusion_experiment as exp2
# from train_aerialbooth_bias_input import run_diffusion_experiment as exp3
# from train_aerialbooth_bias_z123 import run_diffusion_experiment as exp4
# from train_aerialbooth_noisy_z123 import run_diffusion_experiment as exp5

from train_aerialbooth_zero_yes_mutual import run_diffusion_experiment as exp
import os

def main():
    path = "./dataset/video_frames/panda"
    imgs = os.listdir(path)

    print(imgs)
    prompt = "A panda taking a selfie."

    for img in imgs:
        init_image_path = path + "/"  + img
        output_dir = f"./exp_outs/videos/panda/res_{img[:-4]}"
        zero_output_path = f"./zero_outputs/panda/{img}"
        print(f"Output dir: {output_dir}")

        exp(init_image_path, zero_output_path, output_dir, prompt)

if __name__ == "__main__":
    main()