import numpy as np
from dataclasses import dataclass


@dataclass
class CameraMatrices:
    """Matrices for camera intrinsic and extrinsic description."""
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    intrinsic_params: np.ndarray
    projection_matrix: np.ndarray
    depth_bounds: []

    def __getitem__(self, keys):
        return tuple(getattr(self, k) for k in keys)


def parse_file(file_name, scale_factor, down_sample):
    # read camera configuration file
    with open(file_name, encoding='utf-8') as camera_config:
        lines = [line.rstrip() for line in camera_config.readlines()]

    # extrinsic parameters: lines 1-4 define a 4×4 matrix
    extrinsic_params_flat = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsic_params = extrinsic_params_flat.reshape((4, 4))
    extrinsic_params[:3, 3] *= scale_factor

    # intrinsics parameters: lines 7-9 define a 3×3 matrix
    intrinsic_params_flat = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsic_params = intrinsic_params_flat.reshape((3, 3))
    # TODO: why scaling by 4 here?
    # see comment: https://github.com/jzhangbs/Vis-MVSNet/issues/15
    # appears to be difference between depth maps and image resolution
    intrinsic_params[:2] = intrinsic_params[:2] * down_sample * 4

    # depth_min & depth_interval: line 11
    # TODO: why the factor of 192?
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor

    return extrinsic_params, intrinsic_params, (depth_min, depth_max)


def create_matrices(extrinsic_params, intrinsic_params, depth_bounds):
    world_to_camera = extrinsic_params
    camera_to_world = np.linalg.inv(world_to_camera)

    projection_matrix = np.eye(4)
    intrinsic_params_copy = intrinsic_params.copy()
    intrinsic_params_copy[:2] = intrinsic_params_copy[:2] / 4
    projection_matrix[:3, :4] = intrinsic_params_copy @ world_to_camera[:3, :4]

    return CameraMatrices(
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        intrinsic_params=intrinsic_params,
        projection_matrix=projection_matrix,
        depth_bounds=depth_bounds
    )


def load_camera_matrices(data_dir, viewpoint_id, scale_factor, down_sample):
    camera_config_file = f'{data_dir}/Cameras/train/{viewpoint_id:08d}_cam.txt'
    return create_matrices(*parse_file(camera_config_file, scale_factor, down_sample))
