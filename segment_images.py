import os
from PIL import Image
from rembg.bg import remove

def segment_image(image_path:str, output_folder_path:str) -> None:
    image_name = os.path.basename(image_path)

    image_name = image_path.split("/")[-1]
    raw_image = Image.open(image_path).convert("RGB")
    assert image_name.endswith(".png"), f"Image {image_path} must end with PNG"
    
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

    output_image_path = f"{output_folder_path}/{image_name}"
    output_image.save(output_image_path)

    print(f"Image {image_name} segmented")

if __name__ == "__main__":
    images_path = "data/input/gai_images/hard-sdxl-human/"
    output_folder_path = "data/image_background/e5e6eb/test-rembg"
    png_files = [file for file in os.listdir(images_path) if file.endswith('.png')]
    for i, image_name in enumerate(png_files):
        if image_name not in os.listdir(output_folder_path):
            print(f"{i}/{len(png_files)}: Segmenting image {images_path}")
            segment_image(os.path.join(images_path, image_name), output_folder_path)
        else:
            print(f"{i}/{len(png_files)}: Image {image_name} already exists in {output_folder_path}")

