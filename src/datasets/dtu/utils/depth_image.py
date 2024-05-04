import os.path

import cv2
import numpy as np

from src.utils.pfm import read_pfm_file

class DepthImage:
    def __init__(self, file_name):
        super(DepthImage).__init__()

        self.file_name = file_name

    # TODO: document the process here
    def read(
        self,
        down_sample,
        first_scale=(0.5, 0.5),
        crop_x = (44, 556),
        crop_y = (80, 720),
        second_scale=(0.25, 0.25)
    ):
        # check for depth file
        if not os.path.exists(self.file_name):
            return np.zeros((1, 1))

        # parse PFM data
        depth_data, _ = read_pfm_file(self.file_name)
        depth_data = np.array(depth_data, dtype=np.float32) # 800×800

        # down sample
        fx, fy = first_scale
        depth_data = cv2.resize(depth_data, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)  # 600×800

        # crop and down sample
        depth_data = depth_data[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]  # 512×640
        depth_data = cv2.resize(depth_data, None, fx=down_sample, fy=down_sample, interpolation=cv2.INTER_NEAREST)

        # down sample
        # TODO apparently not used in source code
        # fx, fy = second_scale
        # depth = cv2.resize(depth_data, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        # mask = depth > 0

        return depth_data
