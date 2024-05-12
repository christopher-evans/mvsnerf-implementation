import torch


def ray_offsets_sampled(
    height,
    width,
    ray_count,
    batch_size=1,
    dtype=None,
    device=None
):
    """
    Generate pixel offsets for rays for a batch of images.
    Assumes image dimensions are the same across a batch.

    :param int height: Image height
    :param int width: Image width
    :param int ray_count: Number of rays to sample for each item in batch
    :param int batch_size: Batch size
    :param torch.dtype dtype: Data type for return tensor
    :param torch.device device: Device for return tensor

    :return tensor[batch_size, ray_count]: Sampled pixel offsets
    """
    return torch.randint(0, width, (batch_size, ray_count), dtype=dtype, device=device), \
        torch.randint(0, height, (batch_size, ray_count), dtype=dtype, device=device)


def create_ray_offsets_sampled(*args, **kwargs):
    return lambda : ray_offsets_sampled(*args, **kwargs)


def ray_offsets_deterministic(
    height,
    width,
    ray_count,
    batch_size=1,
    dtype=None,
    device=None
):
    """
    Compute all pixel offsets for a batch of images.
    Assumes image dimensions are the same across a batch.

    :param int height: Image height
    :param int width: Image width
    :param int ray_count: Not used
    :param int batch_size: Batch size
    :param torch.dtype dtype: Data type for return tensor
    :param torch.device device: Device for return tensor

    :return tensor[batch_size, height * width]: Pixel offsets for each (x, y) in each image in batch
    """
    y_offsets, x_offsets = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=dtype, device=device),
        torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    )
    x_offsets = x_offsets.reshape(height * width) \
        .unsqueeze(0) \
        .repeat((batch_size, 1))
    y_offsets = y_offsets.reshape(height * width) \
        .unsqueeze(0) \
        .repeat((batch_size, 1))

    return x_offsets, y_offsets


def create_ray_offsets_deterministic(*args, **kwargs):
    return lambda : ray_offsets_deterministic(*args, **kwargs)


def create_rays(ray_offset_function, intrinsics, cameras_to_world):
    """
    Create rays in world co-ordinates for a camera with given intrinsic and extrinsic parameters.

    :param Callable ray_offset_function: Function returning pixel offsets for rays
    :param intrinsics: Camera intrinsic parameters
    :param cameras_to_world: Mapping from camera frame to world frame

    :return tuple[tensor[batch_size, 3], tensor[batch_size, n_offsets, 3], tensor[batch_size, 2, n_offsets]]: Ray origins, directions and pixel co-ordinates
    """
    # move rays to same device as camera
    device = intrinsics.device
    dtype = cameras_to_world.dtype

    # x_offsets.shape [batch_size, n_offsets]
    # y_offsets.shape [batch_size, n_offsets]
    x_offsets, y_offsets = ray_offset_function()

    # add a dimension at end of intrinsics because PyTorch broadcast
    # semantics would prepend instead
    # this allows batch-wise operations on pixel offsets
    intrinsics = intrinsics.unsqueeze(-1)

    # scale offsets with intrinsic parameters and append ones for z values
    # camera_directions.shape [batch_size, n_offsets, 3]
    camera_directions = torch.stack(
        [(
             x_offsets - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0],
             (y_offsets - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1],
             torch.ones_like(x_offsets, dtype=dtype, device=device)
         ],
        -1
    )

    # rotations.shape: (batch_size, 3, 3)
    rotations = cameras_to_world[:, :3, :3]
    # translations.shape: (batch_size, 3, 1)
    translations = cameras_to_world[:, :3, -1]

    # ray_directions.shape (batch_size, n_offsets, 3)
    # pixel_coordinates.shape (batch_size, 2, n_offsets)
    ray_directions = camera_directions @ rotations.transpose(-2, -1)
    pixel_coordinates = torch.stack((y_offsets, x_offsets), dim=1)
    print(ray_directions.shape, pixel_coordinates.shape)

    # ray origins are translation of cameras to world
    rays_origin = translations.clone()

    return rays_origin, ray_directions, pixel_coordinates


def get_ndc_coordinates():
    pass


def march_rays(
    mvs_images,
    intrinsic_params,
    cameras_to_world,
    world_to_cameras,
    ray_offset_function
):
    batch_size, viewpoints, channels, height, width = mvs_images.shape


    for viewpoint in range(viewpoints):
        intrinsics = intrinsic_params[:, viewpoint]
        camera_to_world = cameras_to_world[:, viewpoint]
        world_to_camera = world_to_cameras[:, viewpoint]

        rays_origin, ray_directions, pixel_coordinates = ray_offset_function()
