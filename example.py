import os
from PIL import Image
from rembg.bg import remove

background_image = None
background_image_path = "data/background/true-back-e5e6eb.png"
background_image = Image.open(background_image_path)

image_path = "data/input/gai_images/hard-sdxl-human/039.png"
image_name = image_path.split("/")[-1]
raw_image = Image.open(image_path).convert("RGB")
background_image = background_image.resize(raw_image.size)

output_image = remove(
    data = raw_image,
    alpha_matting = True,
    #alpha_matting_foreground_threshold: int = 240,
    #alpha_matting_background_threshold: int = 10,
    #alpha_matting_erode_size: int = 10,
    session = None, # session object for the 'u2net' model.
    only_mask = False, # flag indicating whether to return only the binary masks
    post_process_mask = False, # flag indicating whether to post-process the masks
    bgcolor = (229, 230, 235, 255), # RGBA value of background color
)
output_folder_path = "data/image_background/e5e6eb/test-rembg"
output_image_path = f"{output_folder_path}/{image_name}"
output_image.save(output_image_path)

