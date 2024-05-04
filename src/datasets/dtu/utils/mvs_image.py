import cv2
import numpy as np

from PIL import Image
from torchvision import transforms


# TODO define these somewhere else?
def get_transforms():
    """
    Define transforms for images: map to [0, 1] RGB values and normalize.

    TODO: script to calculate mean and std for this function
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


class MvsImage:
    def __init__(self, file_name):
        super(MvsImage).__init__()

        self.file_name = file_name
        self.transforms = get_transforms()

    # TODO: document the process here
    def read(
        self,
        down_sample
    ):
        image = Image.open(self.file_name)
        down_sample_resolution = np.round(np.array(image.size) * down_sample).astype('int')
        image = image.resize(down_sample_resolution, resample=Image.Resampling.BILINEAR)
        image = self.transforms(image)

        return image
