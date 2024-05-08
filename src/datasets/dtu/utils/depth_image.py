import os.path

import cv2
import numpy as np

from src.utils.pfm import read_pfm_file


def load_depth_image_file(
        file_name,
        down_sample,
        first_scale=(0.5, 0.5),
        crop_x=(44, 556),
        crop_y=(80, 720),
        # second_scale=(0.25, 0.25)
):
    if not os.path.exists(file_name):
        return np.zeros((1, 1))

    # parse PFM data
    depth_data, _ = read_pfm_file(file_name)
    depth_data = np.array(depth_data, dtype=np.float32)  # 800×800

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


def load_depth_image(
        data_dir,
        scan_id,
        viewpoint_id,
        down_sample
):
    file_name = f'{data_dir}/Depths/{scan_id}/depth_map_{viewpoint_id:04d}.pfm'
    return load_depth_image_file(file_name, down_sample=down_sample)
