"""
TODO: file docstring
"""
import torch
import gc

from kornia.utils import create_meshgrid
import torch.nn.functional as functional


def create_source_depth_values(
    depth_bounds,
    image_height,
    image_width,
    depth_resolution,
    source_translation
):
    """
    For each batch index `b` create a linear space from `depth_bounds[b][0]` to `depth_bounds[b][1]`
    with length `depth_resolution`, then repeat this grid image_height * image_width times to obtain
    a point for every pixel and depth for the reference image.

    Finally, translate the reference grid to the source grid using the translation component of the
    homography from reference to source images.

    :param tensor[batch_size, 2]    depth_bounds:       Min and max depth for scene and viewpoint, with shape (batch_size, 2)
    :param int                      depth_resolution:   Number of depth values
    :param int                      image_height:       Image height
    :param int                      image_width:        Image width
    :param tensor[batch_size, 3, 1] source_translation: Translation from reference to source image

    :return tensor[batch_size, 3, depth_resolution * image_height * image_width]: Depth increments for batch
    """
    # grid_space.shape: (depth_resolution, 1)
    grid_space = torch.linspace(0.0, 1.0, steps=depth_resolution, dtype=depth_bounds.dtype, device=depth_bounds.device) \
        .unsqueeze(0) \
        .transpose(0, 1)

    # depth_grid.shape: (depth_resolution, batch_size)
    depth_grid = depth_bounds[:, 0].mul(1.0 - grid_space) + depth_bounds[:, 1].mul(grid_space)

    # depth_grid.shape: (batch_size, depth_resolution)
    depth_grid = depth_grid.transpose(0, 1)

    # repeat depth grid values for each point in image reference
    # depth_grid.shape: (batch_size, 1, depth_resolution * image_height * image_width)
    batch_size, _ = depth_bounds.shape
    depth_grid = depth_grid.unsqueeze(-1) \
        .unsqueeze(-1) \
        .repeat(1, 1, image_height, image_width) \
        .view(batch_size, 1, depth_resolution * image_height * image_width)

    # return.shape: (batch_size, 3, depth_resolution * image_height * image_width)
    return source_translation / depth_grid


def create_source_plane_values(image_height, image_width, padding, depth_resolution, batch_size, source_rotation):
    """
    For each item in the batch, create an (x, y, z) value at the source view warped from a linear grid
    at the reference view. Repeat each value for every depth interval for combination with depth offsets.

    :param int                      image_height:     Image height (padded)
    :param int                      image_width:      Image width (padded)
    :param int                      depth_resolution: Depth resolution
    :param int                      batch_size:       Batch size
    :param tensor[batch_size, 3, 3] source_rotation:  Rotation from reference view to source view

    :return tensor[batch_size, 3, depth_resolution * image_height * image_width]: Image offsets for batch
    """
    # create 2D meshgrid for sampling values
    # reference_grid.shape: (1, image_height, image_width, 2)
    reference_grid = create_meshgrid(image_height, image_width, normalized_coordinates=False, dtype=source_rotation.dtype, device=source_rotation.device)

    # subtract padding from pixel offsets
    reference_grid -= padding

    # reshape grid
    # reference_grid.shape: (1, 2, image_height * image_width)
    reference_grid = reference_grid.permute(0, 3, 1, 2)
    reference_grid = reference_grid.reshape(1, 2, image_height * image_width)

    # repeat grid for each element in batch
    # reference_grid.shape: (batch_size, 2, image_height * image_width)
    reference_grid = reference_grid.expand(batch_size, -1, -1)

    # add depth offset of 1
    # reference_grid.shape: (batch_size, 3, image_height * image_width)
    reference_grid = torch.cat((reference_grid, torch.ones_like(reference_grid[:, :1])), 1)

    # repeat grid for each depth
    # reference_grid.shape: (batch_size, 3, depth_resolution * image_height * image_width)
    reference_grid = reference_grid.repeat(1, 1, depth_resolution)

    # rotate vectors from reference view to source view
    return source_rotation @ reference_grid


def normalize_grid(grid, image_height, image_width):
    """
    Normalize a sampling grid to have depth in [0, 1] and x and y dimensions in [-1, 1]

    :param tensor[batch_size, 3, depth_resolution * image_height * image_width] grid: Source grid
    :param int image_height: Image height
    :param int image_width: Image width

    :return tensor[batch_size, depth_resolution, image_width, image_height, 2]:
        Normalized grid with values in range [0, 1] × [-1, 1] × [-1, 1]
    """
    # fetch batch size from input grid shape
    batch_size, _, _ = grid.shape

    # normalize (x, y, z) to (x / z, y / z, 1)
    # grid.shape: (batch_size, 3, depth_resolution * image_height * image_width)
    grid = grid[:, :2] / grid[:, 2:]

    # scale width and height to [-1, 1]
    grid[:, 0] = grid[:, 0] / ((image_width - 1) / 2) - 1
    grid[:, 1] = grid[:, 1] / ((image_height - 1) / 2) - 1

    # permute grid for sampling against source features
    # return.shape: (batch_size, depth_resolution * image_width * image_height, 2)
    return grid.permute(0, 2, 1)


def create_volume_grid(
    source_features,
    image_warp_matrices,
    depth_bounds,
    padding,
    depth_resolution
):
    """
    Create a volume grid for interpolating image features with `torch.nn.functional.grid_sample`.

    :param tensor[batch_size, channels, width, height] source_features: Source features
    :param tensor[batch_size, 3, 4] image_warp_matrices: Transformations to reference view frustrum to source view
    :param tensor[batch_size, 2] depth_bounds: Near and far bounds for reference view
    :param int padding: Pad the grid at the reference view before transformation
    :param int depth_resolution: Depth intervals between near and far bounds
    :return: tensor[batch_size, depth_resolution, width_padded, height_padded, 2] Grid for interpolation
    """
    # fetch tensor dimensions
    batch_size, channels, height, width = source_features.shape

    # pad images for interpolation
    # TODO vary this parameter?
    height_padded, width_padded = height + 2 * padding, width + 2 * padding

    # split out reference to source mappings into rotation and translation components
    # source_rotation.shape: (batch_size, 3, 3)
    source_rotation = image_warp_matrices[:, :, :3]
    # source_rotation.shape: (batch_size, 3, 1)
    source_translation = image_warp_matrices[:, :, 3:]

    # repeat depth grid values for each point in image reference
    # source_grid.shape: (batch_size, depth_resolution, width_padded, height_padded, 2)
    return normalize_grid(
        create_source_depth_values(depth_bounds, height_padded, width_padded, depth_resolution, source_translation) \
            + create_source_plane_values(
                height_padded,
                width_padded,
                padding,
                depth_resolution,
                batch_size,
                source_rotation
            ),
        height,
        width
    ) \
        .view(batch_size, depth_resolution, width_padded, height_padded, 2)


def interpolate_at_grid(source_features, source_grid) :
    """
    Interpolate a set of image features using at a grid.

    :param tensor[batch_size, channels, width, height] source_features: Source features.
    :param tensor[batch_size, depth_resolution, grid_width, grid_height, 2] source_grid: Grid for interpolation.
    :return tensor[batch_size, channels, depth_resolution, grid_height, grid_width]: Interpolated features at grid coordinates.
    """
    batch_size, channels, _, _ = source_features.shape
    _, depth_resolution, width_padded, height_padded, _ = source_grid.shape

    # return.shape: (batch_size, channels, depth_resolution, height_padded, width_padded)
    return functional.grid_sample(
        source_features,
        source_grid.view(batch_size, depth_resolution, width_padded * height_padded, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ) \
        .view(batch_size, channels, depth_resolution, height_padded, width_padded)


def build_volume_features(
    image_features,
    source_images,
    image_warp_matrices,
    depth_bounds,
    padding=12,
    depth_resolution=128
):
    # TODO pydocs for dimensions
    # fetch dimensions from input data
    batch_size, viewpoints, feature_channels, height, width = image_features.shape
    _, _, image_channels, _, _ = source_images.shape
    source_viewpoints = viewpoints - 1
    padded_height, padded_width = height + 2 * padding, width + 2 * padding

    # features are down sampled by 4 in each direction, interpolate the images on the new grid
    # source_images.shape: (batch_size, source_viewpoints, image_channels, height, width)
    source_images = functional.interpolate(
        source_images.reshape(batch_size * source_viewpoints, *source_images.shape[2:]),
        (height, width),
        mode='bilinear',
        align_corners=False
    ) \
        .view(batch_size, source_viewpoints, image_channels, height, width)

    # we want to iterate over source images, so split up inputs and
    # reference_feature.shape: (batch_size, feature_channels, height, width)
    # source_features.shape: (batch_size, source_viewpoints, feature_channels, height, width)
    source_features = image_features[:, 1:]
    reference_feature = functional.pad(image_features[:, 0], (padding, padding, padding, padding), "constant", 0)

    # initialize volume features:
    #  0-31  : cost volume for source features
    #  32-40 : down sampled source image data
    # volume_features.shape (batch_size, feature_channels + source_viewpoints * image_channels, depth_resolution, height, width)
    volume_feature_channels = feature_channels + source_viewpoints * image_channels
    volume_features = torch.empty(
        (batch_size, volume_feature_channels, depth_resolution, padded_height, padded_width),
        device=image_features.device,
        dtype=image_features.dtype
    )

    # running total of variance cost metric
    # reference_volume.shape: (batch_size, channels, depth_resolution, height, width)
    reference_volume = reference_feature.unsqueeze(2) \
        .repeat(1, 1, depth_resolution, 1, 1)
    volume_sum = reference_volume
    volume_square_sum = reference_volume ** 2

    # TODO get rid of this
    del reference_volume, reference_feature

    # count of source images containing each voxel
    # used to compute variance metric
    grid_masks = torch.ones(
        (batch_size, depth_resolution, padded_height, padded_width),
        device=volume_sum.device
    )

    for viewpoint_index in range(source_viewpoints):
        source_image = source_images[:, viewpoint_index]
        source_feature = source_features[:, viewpoint_index]
        image_warp_matrix = image_warp_matrices[:, viewpoint_index]

        # source_grid.shape: (batch_size, depth_resolution, width_padded, height_padded, 2)
        source_grid = create_volume_grid(
            source_feature,
            image_warp_matrix,
            depth_bounds,
            padding,
            depth_resolution
        )

        # add image data to volume features
        volume_features[:, feature_channels + viewpoint_index * 3 : feature_channels + (viewpoint_index + 1) * 3] = interpolate_at_grid(source_image, source_grid)

        # add image features to running sum for volume cost metric
        source_volume = interpolate_at_grid(source_feature, source_grid)
        volume_sum = volume_sum + source_volume
        volume_square_sum = volume_square_sum + source_volume ** 2

        # update grid masks
        source_grid = source_grid.view(batch_size, depth_resolution, padded_height, padded_width, 2)
        grid_masks += ((source_grid > -1.0) * (source_grid < 1.0)) \
            .prod(dim=-1) \
            .type(volume_sum.dtype)

        # TODO get rid of this
        del source_volume, source_feature, image_warp_matrix, source_grid

    # add variance metric to volume features
    count = (1.0 / grid_masks).unsqueeze(dim=1)
    volume_features[:, :32] = volume_square_sum * count - (volume_sum * count) ** 2

    return volume_features


# def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
#     """
#     src_feat: (B, C, H, W)
#     proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
#     depth_values: (B, D, H, W)
#     out: (B, C, D, H, W)
#     """
#
#     if src_grid==None:
#         B, C, H, W = src_feat.shape
#         device = src_feat.device
#
#         if pad>0:
#             H_pad, W_pad = H + pad*2, W + pad*2
#         else:
#             H_pad, W_pad = H, W
#
#         depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
#         D = depth_values.shape[1]
#
#         R = proj_mat[:, :, :3]  # (B, 3, 3)
#         T = proj_mat[:, :, 3:]  # (B, 3, 1)
#         # create grid from the ref frame
#         ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
#         if pad>0:
#             ref_grid -= pad
#
#         ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
#         ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
#         ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
#         ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
#         ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
#         src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
#         del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory
#
#
#
#         src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
#         del src_grid_d
#         src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
#         src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
#         src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
#         src_grid = src_grid.view(B, D, W_pad, H_pad, 2)
#
#     B, D, W_pad, H_pad = src_grid.shape[:4]
#     warped_src_feat = functional.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
#                                     mode='bilinear', padding_mode='zeros',
#                                     align_corners=True)  # (B, C, D, H*W)
#     warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
#     # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
#     return warped_src_feat, src_grid
#
# batch_size = 2
# height = 3
# width = 3
# channels = 3
# padding = 1
# depth_resolution = 3
# depth_bounds = torch.tensor([
#     [1.125, 6.175],
#     [2.25, 5.75]
# ])
# image_warp_matrices = torch.rand((batch_size, 3, 4))
# source_features = torch.rand((batch_size, channels, height, width))
#
# my_g = create_volume_grid(source_features, image_warp_matrices, depth_bounds, padding=padding,
#                                 depth_resolution=depth_resolution)
# my_f = interpolate_at_grid(source_features, my_g)
#
# t_vals = torch.linspace(0., 1., steps=depth_resolution, dtype=torch.float32)  # (B, D)
# near, far = depth_bounds[1]  # assume batch size==1
# dv = near * (1. - t_vals) + far * t_vals
# dv = dv.unsqueeze(0)
#
# n_f, n_g = homo_warp(source_features[1:], image_warp_matrices[1:], dv, pad=padding)
#
# print('feat ', my_f[1:], n_f, torch.allclose(my_f[1:], n_f))
# print('grid ', my_g[1:], n_g, torch.allclose(my_g[1:], n_g))
