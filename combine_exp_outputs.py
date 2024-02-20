from PIL import Image, ImageDraw, ImageFont
import os

main_folder = 'exp_outputs'
sub_categories = ['sum_out', 'avg_out', 'input_bias_out', 'z123_bias_out']
sub_category_img = [
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
image_types = ['aerial', 'back', 'bottom', 'side']

output_directory = 'composite_outputs'
os.makedirs(output_directory, exist_ok=True)

try:
    font = ImageFont.truetype("arial.ttf", 20)  
except IOError:
    font = ImageFont.load_default()
    print("using default font")

def create_composite_image(image_type):
    for sub_img in sub_category_img:
        images = []  
        for sub_cat in sub_categories:
            for i in range(1, 6):
                image_name = f'{image_type}{i}.png'
                image_path = os.path.join(main_folder, sub_cat, sub_img, image_name)
                try:
                    img = Image.open(image_path)
                    images.append(img)
                except FileNotFoundError:
                    print(f"File not found: {image_path}")

        if images:
            img_width, img_height = images[0].size
            composite_image = Image.new('RGB', (img_width * len(sub_categories) + 100, img_height * 5 + 50), 'white')
            draw = ImageDraw.Draw(composite_image)

            for idx, label in enumerate(sub_categories):
                draw.text(((idx * img_width) + img_width/2 - 20, img_height * 5 + 10), label, fill="black", font=font)

            for i in range(5):
                draw.text((img_width * len(sub_categories) + 10, i * img_height + img_height/2 - 10), f'{image_type}{i+1}', fill="black", font=font)

            for i, img in enumerate(images):
                x_offset = (i % len(sub_categories)) * img_width
                y_offset = (i // len(sub_categories)) * img_height
                composite_image.paste(img, (x_offset, y_offset))

            composite_image.save(os.path.join(output_directory, f'{image_type}_{sub_img}.png'))
            print(f"Composite image created for {image_type} in {sub_img}")

for image_type in image_types:
    create_composite_image(image_type)

print("Composite images generation completed.")

