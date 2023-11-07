import os
from PIL import Image
from rembg.bg import remove

def segment_image(image_path:str,
                  output_folder_path:str,
                  foreground:int,
                  background:int,
                  erode_size:int) -> None:
    image_name = os.path.basename(image_path)

    image_name = image_path.split("/")[-1]
    raw_image = Image.open(image_path).convert("RGB")
    assert image_name.endswith(".png"), f"Image {image_path} must end with PNG"
    
    output_image = remove(
        data = raw_image,
        alpha_matting = True,
        alpha_matting_foreground_threshold = foreground,
        alpha_matting_background_threshold = background,
        alpha_matting_erode_size = erode_size,
        session = None, # session object for the 'u2net' model.
        only_mask = False, # flag indicating whether to return only the binary masks
        post_process_mask = False, # flag indicating whether to post-process the masks
        bgcolor = (229, 230, 235, 255), # RGBA value of background color
        )

    output_image_path = f"{output_folder_path}/{str(threshold)}_{image_name}"
    output_image.save(output_image_path)

    print(f"Image {image_name} segmented")

if __name__ == "__main__":
    images_path = "data/input/gai_images/hard-sdxl-human/"
    image_path = "data/input/gai_images/hard-sdxl-human/039.png"
    output_folder_path = "data/image_background/e5e6eb/test-rembg/erode_size"
    foregrounds = [60, 120, 180, 240]
    backgrounds = [110,130]
    erode_sizes = [2, 4, 6, 8, 10]
    for val in values:
        print(f"Testing {val}")
        segment_image(image_path, output_folder_path, threshold=val)

