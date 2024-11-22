import os
import glob
from PIL import Image
import numpy as np

def preprocess_data(prompt_data_path):
    """
    Crops video frames to apply the structural consistency metric
    """
    folder_path = os.path.join(prompt_data_path, "save/it*-test/rgb_images/*.png")

    folder_images_path = glob.glob(folder_path)
    print('Len: ', folder_path, len(folder_images_path))

    # delete all rgba images and then we will redump them
    del_rgba = glob.glob(os.path.join(prompt_data_path, "save/it*-test/rgb_images/*_rgba.png"))
    [os.system('rm -r -v ' + img) for img in del_rgba]
    
    video_max_side = -1
    for rgb_img_path in folder_images_path:
        if "_rgba" in rgb_img_path: 
            # data already exists
            continue

        try:
            # Get crop factor and update video max side
            max_side = get_crop_factor(rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
            video_max_side = max(video_max_side, max_side)
        except Exception as e: 
            print(f"Error in get_crop_factor for {rgb_img_path}: {e}")
            continue

    for rgb_img_path in folder_images_path:
        if "_rgba" in rgb_img_path: 
            # data already exists
            continue

        try:
            convert_to_rgba(video_max_side, rgb_img_path, rgb_img_path.replace('rgb_images', 'opacity'))
        except Exception as e: 
            print(f"Error in convert_to_rgba for {rgb_img_path}: {e}")
            continue

def get_crop_factor(rgb_image_path, opacity_mask_path):
    rgb_img = np.asarray(Image.open(rgb_image_path))
    opacity_mask_img = np.asarray(Image.open(opacity_mask_path))[...,0:1]
    
    opacity_mask_img = np.asarray((opacity_mask_img>50)*255, dtype=np.uint8)
    non_zero_indices = np.nonzero(opacity_mask_img)
    min_y, min_x = np.min(non_zero_indices[0], axis=0), np.min(non_zero_indices[1], axis=0)
    max_y, max_x = np.max(non_zero_indices[0], axis=0), np.max(non_zero_indices[1], axis=0)

    cropped_rgb = rgb_img[min_y:max_y, min_x:max_x]
    cropped_opacity_mask = opacity_mask_img[min_y:max_y, min_x:max_x]

    height, width = cropped_rgb.shape[:2]
    max_side = max(height, width)
    return max_side

def convert_to_rgba(max_side, rgb_image_path, opacity_mask_path):
    rgb_img = np.asarray(Image.open(rgb_image_path))
    opacity_mask_img = np.asarray(Image.open(opacity_mask_path))[...,0:1]
    opacity_mask_img = np.asarray((opacity_mask_img>50)*255, dtype=np.uint8)
    
    if (opacity_mask_img>200).sum()==0:
        padded_rgb = rgb_img
        padded_opacity_mask = opacity_mask_img
    else:

        non_zero_indices = np.nonzero(opacity_mask_img)
        min_y, min_x = np.min(non_zero_indices[0], axis=0), np.min(non_zero_indices[1], axis=0)
        max_y, max_x = np.max(non_zero_indices[0], axis=0), np.max(non_zero_indices[1], axis=0)

        # Crop the RGB image and opacity mask using the bounding box
        cropped_rgb = rgb_img[min_y:max_y, min_x:max_x]
        cropped_opacity_mask = opacity_mask_img[min_y:max_y, min_x:max_x]

        height, width = cropped_rgb.shape[:2]
        max_side = max_side + 40
        pad_height = max_side - height
        pad_width = max_side - width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_rgb = np.pad(cropped_rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=255)
        padded_opacity_mask = np.pad(cropped_opacity_mask[...,0], ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)[...,None]

    padded_rgba_image = np.dstack((padded_rgb, padded_opacity_mask))
    Image.fromarray(padded_rgba_image).save(rgb_image_path.replace('.png', '_cropped_rgba.png'))