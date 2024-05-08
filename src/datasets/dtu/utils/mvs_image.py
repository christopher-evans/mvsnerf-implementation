import numpy as np

from PIL import Image
from torchvision import transforms

"""
Define transforms for images: map to [0, 1] RGB values and normalize.

TODO: script to calculate mean and std for this function
"""
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def load_mvs_image_file(file_name, down_sample):
    image = Image.open(file_name)
    down_sample_resolution = np.round(np.array(image.size) * down_sample).astype('int')
    image = image.resize(down_sample_resolution, resample=Image.Resampling.BILINEAR)
    return transforms(image)


def load_mvs_image(
    data_dir,
    scan_id,
    viewpoint_id,
    lighting_id,
    down_sample
):
    file_name = f'{data_dir}/Rectified/{scan_id}_train/rect_{viewpoint_id + 1:03d}_{lighting_id}_r5000.png'
    return load_mvs_image_file(file_name, down_sample=down_sample)
