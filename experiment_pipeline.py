from train_aerialbooth_no_guidance import run_diffusion_experiment as exp1
from train_aerialbooth_zero_no_mutual import run_diffusion_experiment as exp2
from train_aerialbooth_zero_yes_mutual import run_diffusion_experiment as exp3

def main():
    path = "./dataset/synthetic_sdxl_images/"
    img_lst = ["architectures8", "architectures23", "architectures34", "birdanimal38", "city27", "human8"]
    prompts = ["A modern museum with white walls and glass ceilings", "A futuristic cityscape with towering buildings.", "A coastal lighthouse with a spiral staircase.", "A group of goslings following their parent.", "A bustling city street during a summer festival.", "A scientist working in a laboratory."]

    for i in range(len(img_lst)):
       init_image_path = path + img_lst[i] + ".png" 
       output_dir = f"./exp_outputs/{img_lst[i]}/"
       prompt = prompt[i]
       zero_output_path = "./zero_outputs/output" + img_lst[i] + ".png" 
       exp1(init_image_path, output_dir, prompt)
       exp2(init_image_path, zero_output_path, output_dir, prompt)
       exp3(init_image_path, zero_output_path, output_dir, prompt)

if __name__ == "__main__":
    main()

