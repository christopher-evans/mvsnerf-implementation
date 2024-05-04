"""
Wrapper class for train, test or validation DTU dataset.
"""
import numpy as np
import torch

from torch.utils.data import Dataset
from datasets.dtu.utils.depth_image import DepthImage
from datasets.dtu.utils.mvs_image import MvsImage
from dataclasses import dataclass


@dataclass
class DataMatrices:
    """Matrices for camera intrinsic and extrinsic description."""
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    depth_bound: np.ndarray
    image_warp: np.ndarray
    intrinsic_params: np.ndarray
    affine_map: np.ndarray
    affine_map_inverse: np.ndarray


@dataclass
class DataPoint:
    """Matrices for camera intrinsic and extrinsic description."""
    scan_id: np.ndarray
    viewpoint_ids: np.ndarray
    lighting_id: np.ndarray
    mvs_images: torch.Tensor
    depth_maps: np.ndarray
    matrices: DataMatrices


class DTUDataset(Dataset):

    def __init__(
        self,
        mvs_configurations,
        camera_matrices,
        data_dir: str = ".data/processed/dtu_example",
        down_sample=1.0,
        scale_factor=1.0 / 200,
        max_length=-1
    ):
        super().__init__()

        self.mvs_configurations = mvs_configurations
        self.camera_matrices = camera_matrices
        self.data_dir = data_dir
        self.max_length = max_length
        self.down_sample = down_sample
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mvs_configurations) if self.max_length <= 0 else min(self.max_length,
                                                                             len(self.mvs_configurations))

    def read_mvs_image(self, image_file_name):
        return MvsImage(image_file_name).read(self.down_sample)

    def read_depth_image(self, depth_map_filename):
        return DepthImage(depth_map_filename).read(down_sample=self.down_sample) * self.scale_factor

    def mvs_image_file_name(self, scan_id, viewpoint_id, lighting_id):
        return f'{self.data_dir}/Rectified/{scan_id}_train/rect_{viewpoint_id + 1:03d}_{lighting_id}_r5000.png'

    def depth_image_file_name(self, scan_id, viewpoint_id):
        return f'{self.data_dir}/Depths/{scan_id}/depth_map_{viewpoint_id:04d}.pfm'

    def __getitem__(self, index):
        mvs_config = self.mvs_configurations[index]
        reference_view = mvs_config.reference_view
        source_views = mvs_config.source_views.fetch()
        lighting_condition_id = mvs_config.lighting_condition_id

        reference_view_matrices = self.camera_matrices[reference_view]
        reference_projection_matrix = reference_view_matrices.projection_matrix
        reference_projection_inverse = np.linalg.inv(reference_projection_matrix)

        # initialize with reference view data
        world_to_camera_matrices = [reference_view_matrices.world_to_camera]
        camera_to_world_matrices = [reference_view_matrices.camera_to_world]
        depth_bound_matrices = [reference_view_matrices.depth_bounds]
        image_warp_matrices = [np.eye(4)]
        intrinsic_param_matrices = [reference_view_matrices.intrinsic_params]
        affine_map_matrices = [reference_projection_matrix]
        affine_map_inverse_matrices = [reference_projection_inverse]

        # initialize images with reference view data
        image_file_name = self.mvs_image_file_name(mvs_config.scan_id, reference_view, mvs_config.lighting_condition_id)
        depth_file_name = self.depth_image_file_name(mvs_config.scan_id, reference_view)

        mvs_images = [self.read_mvs_image(image_file_name)]
        depth_maps = [self.read_depth_image(depth_file_name)]

        for viewpoint_id in source_views:
            image_file_name = self.mvs_image_file_name(mvs_config.scan_id, viewpoint_id,
                                                       mvs_config.lighting_condition_id)
            depth_file_name = self.depth_image_file_name(mvs_config.scan_id, viewpoint_id)

            mvs_images.append(self.read_mvs_image(image_file_name))
            depth_maps.append(self.read_depth_image(depth_file_name))

            camera_matrices = self.camera_matrices[viewpoint_id]
            projection_matrix = camera_matrices.projection_matrix

            world_to_camera_matrices.append(camera_matrices.world_to_camera)
            camera_to_world_matrices.append(camera_matrices.camera_to_world)
            depth_bound_matrices.append(camera_matrices.depth_bounds)
            intrinsic_param_matrices.append(camera_matrices.intrinsic_params)
            affine_map_matrices.append(camera_matrices.projection_matrix)
            affine_map_inverse_matrices.append(np.linalg.inv(projection_matrix))
            image_warp_matrices.append(projection_matrix @ reference_projection_inverse)

        mvs_images = torch.stack(mvs_images).float()
        depth_maps = np.stack(depth_maps)

        world_to_camera_matrices, camera_to_world_matrices = np.stack(world_to_camera_matrices), np.stack(camera_to_world_matrices)
        depth_bound_matrices = np.stack(depth_bound_matrices)
        intrinsic_param_matrices = np.stack(intrinsic_param_matrices)
        affine_map_matrices, affine_map_inverse_matrices = np.stack(affine_map_matrices), np.stack(affine_map_inverse_matrices)
        image_warp_matrices = np.stack(image_warp_matrices)[:, :3]

        return {
            'scan_id': mvs_config.scan_id,
            'viewpoint_ids': [reference_view] + source_views,
            'lighting_id': lighting_condition_id,
            'mvs_images': mvs_images,
            'depth_maps': depth_maps.astype(np.float32),
            'world_to_camera': world_to_camera_matrices.astype(np.float32),
            'camera_to_world': camera_to_world_matrices.astype(np.float32),
            'depth_bound': depth_bound_matrices.astype(np.float32),
            'image_warp': image_warp_matrices.astype(np.float32),
            'intrinsic_param_matrices': intrinsic_param_matrices.astype(np.float32),
            'affine_map_matrices': affine_map_matrices,
            'affine_map_inverse_matrices': affine_map_inverse_matrices,
        }
