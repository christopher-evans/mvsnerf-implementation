import torch

import torch.nn.functional as functional

from utils.ray_marching import get_nd_coordinates


def interpolate_volume_encoding(volume_encoding, all_point_samples_ndc):
    """

    :param volume_encoding: Neural volume encoding
    :type volume_encoding: tensor[batch_size, 8, depth_resolution, feature_height, feature_width]

    :param all_point_samples_ndc: All point samples in normalized device coordinates
    :type all_point_samples_ndc: tensor[batch_size, 1, ray_count, ray_sample_count, 3]

    :return:Neural encoding volume tri-linearly interpolated at each of the point samples
    :rtype: tensor[batch_size * ray_count * ray_sample_count, 8]
    """

    # input    (batch_size, 8, depth_resolution, feature_height, feature_width)
    # at grid  (batch_size, 1, ray_count, ray_sample_count, 3)
    # result:  (batch_size, 8, 1, ray_count, ray_sample_count)
    batch_size, _, ray_count, ray_sample_count, _ = all_point_samples_ndc.shape
    return functional.grid_sample(volume_encoding, all_point_samples_ndc, align_corners=True, mode='bilinear') \
        .permute(0, 2, 3, 4, 1) \
        .squeeze(dim=1) \
        .contiguous() \
        .view(batch_size * ray_count * ray_sample_count, 8)


def interpolate_pixel_colours(all_point_samples, mvs_images, world_to_cameras, intrinsic_params, depth_bounds):
    """

    :param all_point_samples: Ray marching sample points
    :type all_point_samples: tensor[batch_size, ray_count, ray_sample_count, 3]

    :param mvs_images: MVS source images
    :type mvs_images: tensor[batch_size, source_viewpoints, channels, height, width]

    :param world_to_cameras: World to camera mappings for source viewpoints
    :type world_to_cameras: tensor[batch_size, source_viewpoints, 3, 4]

    :param intrinsic_params: Intrinsic parameters for source viewpoints
    :type intrinsic_params: tensor[batch_size, source_viewpoints, 3, 3]

    :return: Colours of all source images at all sample points
    :rtype: tensor[batch_size * ray_count * ray_sample_count, channels * source_viewpoints]
    """
    batch_size, source_viewpoints, channels, height, width = mvs_images.shape
    _, ray_count, ray_sample_count, _ = all_point_samples.shape
    device, dtype = mvs_images.device, mvs_images.dtype
    image_scale = torch.tensor([width - 1, height - 1], device=device, dtype=dtype)
    colours = torch.empty((batch_size, channels * source_viewpoints, ray_count, ray_sample_count), device=device, dtype=dtype)

    for source_index in range(source_viewpoints):
        # TODO effect of padding here?
        # point_samples_pixel.shape: (batch_size, ray_count, ray_sample_count, 2)
        point_samples_pixel = get_nd_coordinates(
            all_point_samples,
            world_to_cameras[:, source_index],
            intrinsic_params[:, source_index],
            depth_bounds[:, source_index],
            image_scale,
            padding=0
        )[..., :2] * 2.0 - 1.0

        # mvs_images[:, source_index].shape: (batch_size, channels, width, height)
        # point_samples_pixel.shape: (batch_size, ray_count, ray_sample_count, 2)
        # colours[:].shape: (batch_size, channels, ray_count, ray_sample_count)
        colours[:, source_index * channels:(source_index + 1) * channels, :, :] = functional.grid_sample(
            mvs_images[:, source_index],
            point_samples_pixel[..., :2],
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )

    return colours.permute(0, 2, 3, 1) \
        .contiguous() \
        .view(batch_size * ray_count * ray_sample_count, channels * source_viewpoints)


def create_direction_vectors(ray_directions, world_to_camera_reference, ray_sample_count):
    """
    Normalize direction vectors and map to reference camera coordinates

    :param ray_directions: Ray directions from reference camera
    :type ray_directions: tensor[batch_size, ray_count, 3]

    :param world_to_camera_reference: World coordinates to reference camera coordinates
    :type world_to_camera_reference: tensor[batch_size, 3, 4]

    :return: Direction vectors rotates to reference camera coordinates
    :rtype: tensor[batch_size, ray_count, 3]
    """
    # fetch dimensions
    batch_size, ray_count, _ = ray_directions.shape

    # normalize direction vectors
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1).unsqueeze(-1)

    #
    world_to_camera_rotation = world_to_camera_reference[:, :3, :3]

    # TODO copy this for each point
    ray_directions = ray_directions @ world_to_camera_rotation.transpose(dim0=1, dim1=2)

    return ray_directions.unsqueeze(dim=2) \
        .expand(-1, -1, ray_sample_count, -1) \
        .contiguous() \
        .view(batch_size * ray_count * ray_sample_count, 3)


def parse_nerf(prediction_colours, prediction_density):
    """

    :param prediction_colours: RGB outputs from MLP
    :type prediction_colours: tensor[batch_size, ray_count, ray_sample_count, 3]

    :param prediction_density: Density outputs from MLP
    :type prediction_density:  tensor[batch_size, ray_count, ray_sample_count]

    :return:
    """
    batch_size, ray_count, _ = prediction_density.shape
    device, dtype = prediction_density.device, prediction_density.dtype

    alpha = 1. - torch.exp(-prediction_density)
    # TODO parameterize 1e-10 here?
    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones(batch_size, ray_count, 1, device=device, dtype=dtype),
                1. - alpha + 1e-10
            ],
            dim=2
        ),
        dim=2
    )

    weights = alpha * transmittance[..., :-1]
    prediction_rgb = torch.sum(
        weights.unsqueeze(dim=3) * prediction_colours,
        dim=2
    )

    return prediction_rgb
