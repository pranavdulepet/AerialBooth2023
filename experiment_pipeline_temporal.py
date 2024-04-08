from train_aerialbooth_temporal import run_diffusion_experiment as exp
import os

def main():
    path = "./dataset/video_frames/panda"
    imgs = sorted(os.listdir(path))  

    prompt = "A panda taking a selfie."  

    prev_image_path = None  
    for img in imgs:
        init_image_path = os.path.join(path, img)
        output_dir = f"./exp_outs/videos_temporal/panda/res_{img[:-4]}"
        zero_output_path = f"./zero_outputs_2/panda/{img}"

        generated_image_path = exp(init_image_path=init_image_path, 
                                   zero_output_path=zero_output_path, 
                                   output_dir=output_dir, 
                                   prompt=prompt, 
                                   prev_image=prev_image_path) 

        prev_image_path = generated_image_path 

if __name__ == "__main__":
    main()

