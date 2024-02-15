
from train_aerialbooth_zero_yes_mutual import run_diffusion_experiment as exp

import os1

def main():
    path = "./dataset/synthetic_sdxl_images/"
    img_lst = [
        "birdanimal1",
        "animation4",
        "animation3",
        "architectures8",
        "architectures23",
        "architectures34",
        "birdanimal38",
        "city27",
        "human8",
        "human81",
        "indoor2",
        "indoor44",
        "nature16",
        "nature18",
        "traffic24",
        "traffic10"
    ]
    prompts = [
        "A fluffy baby penguin learning to waddle.",
        "A young wizard learning to control their powers.",
        "A magical forest where animals talk.",
        "A modern museum with white walls and glass ceilings.",
        "A futuristic cityscape with towering buildings.",
        "A coastal lighthouse with a spiral staircase.",
        "A group of goslings following their parent.",
        "A bustling city street during a summer festival.",
        "A scientist working in a laboratory.",
        "A teacher inspiring students with a captivating lesson.",
        "A modern living room with a large TV, a glass coffee table, and leather chairs, with a view of a city skyline.",
        "An upscale clothing boutique with designer labels.",
        "A cave with sparkling stalagmites and stalactites.",
        "A beach with black sand and dramatic rock formations.",
        "An airport with planes taking off and landing.",
        "A motorcycle lane with motorcycles riding by."
    ]

    for i in range(len(img_lst)):
       init_image_path = path + img_lst[i] + ".png" 
       output_dir = f"./exp_outputs/{img_lst[i]}/"
       prompt = prompts[i]
       zero_output_path = "./zero_outputs/output" + img_lst[i] + ".png" 
       exp(init_image_path, zero_output_path, output_dir, prompt)

if __name__ == "__main__":
    main()

