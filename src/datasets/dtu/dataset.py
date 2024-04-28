import cv2
import numpy as np
import torch
import os

from torch.utils.data import Dataset
from src.utils.pfm import read_pfm_file
from PIL import Image
from torchvision import transforms


def view_ids_top_three():
    return list(range(3))


def view_ids_rand_top_five():
    return torch.randperm(5)[:3]


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


class DTUDataset(Dataset):

    def __init__(
        self,
        mvs_configurations,
        view_id_sampler,
        viewpoint_index_map,
        projection_matrices,
        intrinsic_parameters,
        world_to_cameras,
        cameras_to_worlds,
        data_dir: str = ".data/processed/dtu_example",
        max_length=-1,
        scale_factor=1.0 / 200,
        down_sample=1.0,
    ):
        super().__init__()

        self.mvs_configurations = mvs_configurations
        self.view_id_sampler = view_id_sampler
        self.viewpoint_index_map = viewpoint_index_map
        self.data_dir = data_dir
        self.max_length = max_length
        self.scale_factor = scale_factor
        self.down_sample = down_sample

        self.projection_matrices = projection_matrices
        self.intrinsic_parameters = intrinsic_parameters
        self.world_to_cameras = world_to_cameras
        self.cameras_to_worlds = cameras_to_worlds

        # image transforms to apply to data
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.mvs_configurations) if self.max_length <= 0 else self.max_length

    def read_depth(self, depth_map_filename):
        # TODO: documented dimensions here seem que
        depth_h = np.array(read_pfm_file(depth_map_filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.down_sample, fy=self.down_sample, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4, interpolation=cv2.INTER_NEAREST)
        mask = depth > 0

        return depth, mask, depth_h

    def __getitem__(self, index):
        sample = {}
        scan, lighting_id, reference_view, source_views = self.mvs_configurations[index]
        # TODO: switched the order of the reference view and source due to logic below for projection matrices
        view_ids = [reference_view] + [source_views[index] for index in self.view_id_sampler()]

        affine_mat, affine_mat_inv = [], []
        images, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for index, view_id in enumerate(view_ids):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            image_filename = f'{self.data_dir}/Rectified/{scan}_train/rect_{view_id + 1:03d}_{lighting_id}_r5000.png'
            depth_filename = f'{self.data_dir}/Depths/{scan}/depth_map_{view_id:04d}.pfm'

            image = Image.open(image_filename)
            image_wh = np.round(np.array(image.size) * self.down_sample).astype('int')
            image = image.resize(image_wh, resample=Image.Resampling.BILINEAR)
            image = self.transforms(image)
            images += [image]

            index_mat = self.viewpoint_index_map[view_id]
            proj_mat_ls, near_far = self.projection_matrices[index_mat]
            intrinsics.append(self.intrinsic_parameters[index_mat])
            w2cs.append(self.world_to_cameras[index_mat])
            c2ws.append(self.cameras_to_worlds[index_mat])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            if index == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            if os.path.exists(depth_filename):
                depth, mask, depth_h = self.read_depth(depth_filename)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

            near_fars.append(near_far)

        images = torch.stack(images).float()

        depths_h = np.stack(depths_h)
        proj_mats = np.stack(proj_mats)[:, :3]
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        view_ids_all = [reference_view] + list(source_views) if type(source_views[0]) is not list else [j for sub in source_views for j in sub]
        c2ws_all = np.array([self.cameras_to_worlds[self.viewpoint_index_map[i]] for i in view_ids_all])

        sample['images'] = images  # (V, H, W, 3)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['light_id'] = np.array(lighting_id)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = scan
        sample['c2ws_all'] = c2ws_all.astype(np.float32)

        return sample
