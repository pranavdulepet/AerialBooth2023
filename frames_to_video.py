import os
import cv2
import glob
from PIL import Image

print(f'frame outs: {os.listdir("exp_outs/videos_temporal/panda/res")}')

output_dir = './frames_to_vids/panda'
os.makedirs(output_dir, exist_ok=True)


# aerial_frames = []
# back_frames = []
# bottom_frames = []
# side_frames = []

video_size = (Image.open('exp_outs/videos_temporal/panda/res/generated_image_0.png')).size

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

aerial_video = cv2.VideoWriter(f'{output_dir}/aerial_panda.mp4', fourcc, 1, video_size)
# back_video = cv2.VideoWriter(f'{output_dir}/back_panda.mp4', fourcc, 1, video_size)
# bottom_video = cv2.VideoWriter(f'{output_dir}/bottom_panda.mp4', fourcc, 1, video_size)
# side_video = cv2.VideoWriter(f'{output_dir}/side_panda.mp4', fourcc, 1, video_size)

for i in range(8):
    img_path = f'exp_outs/videos_temporal/panda/res/generated_image_{i}.png'
    
    aerial_img = cv2.imread(img_path)
    aerial_img = cv2.cvtColor(aerial_img, cv2.COLOR_BGR2RGB)
    aerial_video.write(aerial_img)

    # back_img = cv2.imread(f'{curr_dir}/back.png')
    # back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
    # back_video.write(back_img)

    # bottom_img = cv2.imread(f'{curr_dir}/bottom.png')
    # bottom_img = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB)
    # bottom_video.write(bottom_img)

    # side_img = cv2.imread(f'{curr_dir}/side.png')
    # side_img = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
    # side_video.write(side_img)

aerial_video.release()
# back_video.release()
# bottom_video.release()
# side_video.release()

print(f"video saved in {output_dir}")