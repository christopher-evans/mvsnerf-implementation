import numpy as np
from dataclasses import dataclass


@dataclass
class CameraMatrices:
    """
    Matrices for plane sweep cost calculations and depth bounds.
    """
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    intrinsic_params: np.ndarray
    projection_matrix: np.ndarray
    depth_bounds: tuple

    def __getitem__(self, keys):
        """
        Accessor for object properties.

        :param keys: Keys to retrieve
        :return: Values for provided keys
        """
        return tuple(getattr(self, k) for k in keys)


def parse_file(file_name, depth_scale_factor, image_down_sample):
    """
    Parse values from a camera configuration file and map these to camera parameters
    and depth values for the viewpoint.

    :param file_name: File containing camera configuration
    :param depth_scale_factor: Scale factor for depth information for an image
    :param image_down_sample: Down sampling factor for image resolution
    :return: Intrinsic and extrinsic matrices, min and max depth values
    """
    # read camera configuration file
    with open(file_name, encoding='utf-8') as camera_config:
        lines = [line.rstrip() for line in camera_config.readlines()]

    # extrinsic parameters: lines 1-4 define a 4×4 matrix
    extrinsic_params_flat = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsic_params = extrinsic_params_flat.reshape((4, 4))
    extrinsic_params[:3, 3] *= depth_scale_factor

    # intrinsics parameters: lines 7-9 define a 3×3 matrix
    intrinsic_params_flat = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsic_params = intrinsic_params_flat.reshape((3, 3))
    # see comment: https://github.com/jzhangbs/Vis-MVSNet/issues/15
    # appears to be difference between depth maps and image resolution
    # Note also intrinsics should probably have 0.0 / 4.0 and 44.0 / 4.0 added
    # see https://github.com/idiap/GeoNeRF/blob/main/data/dtu.py#L161C9-L162C57
    intrinsic_params[0, 2] = intrinsic_params[0, 2]# + 80.0 / 4.0
    intrinsic_params[1, 2] = intrinsic_params[1, 2]# + 44.0 / 4.0
    intrinsic_params[:2] = intrinsic_params[:2] * image_down_sample * 4

    # depth_min & depth_interval: line 11
    # TODO: why the factor of 192?
    depth_min = float(lines[11].split()[0]) * depth_scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * depth_scale_factor

    return extrinsic_params, intrinsic_params, (depth_min, depth_max)


def create_matrices(extrinsic_params, intrinsic_params, depth_bounds):
    """
    Create matrices for plane sweep and depth bounds from extrinsic and extrinsic parameters.

    :param extrinsic_params: Camera extrinsic parameters
    :param intrinsic_params: Camera extrinsic parameters
    :param depth_bounds: Min and max depth values for viewpoint
    :return: Camera matrices
    """
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


def load_camera_matrices(data_dir, viewpoint_id, depth_scale_factor, image_down_sample):
    """
    Load all matrices needed for plane sweep from a configuration file.

    :param data_dir: Location of camera data
    :param viewpoint_id: Viewpoint ID
    :param depth_scale_factor: Scale factor for depth data
    :param image_down_sample: Down sampling of image resolution
    :return: Camera matrices
    """
    #WIP
    camera_config_file = f'{data_dir}/Cameras/train/{viewpoint_id:08d}_cam.txt'
    return create_matrices(*parse_file(camera_config_file, depth_scale_factor, image_down_sample))
