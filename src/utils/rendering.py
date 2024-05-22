import torch

import torch.nn.functional as functional
import torch.nn.functional as F

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
    return functional.grid_sample(volume_encoding, all_point_samples_ndc * 2 - 1, align_corners=True, mode='bilinear') \
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
    channels += 1
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

        in_mask = ((point_samples_pixel >-1.0)*(point_samples_pixel < 1.0))
        in_mask = (in_mask[...,0]*in_mask[...,1]).float()
        #colours[:, (source_index + 1) * channels, :, :] = in_mask.unsqueeze(1)
        #print('nz',torch.count_nonzero(in_mask), in_mask.shape, torch.prod(torch.tensor(list(in_mask.shape))))

        colours[:, source_index * channels:(source_index + 1) * channels, :, :] = torch.cat([functional.grid_sample(
            mvs_images[:, source_index],
            point_samples_pixel,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        ), in_mask.unsqueeze(1)], dim=1)


    return colours.permute(0, 2, 3, 1) \
        .contiguous() \
        .view(batch_size * ray_count * ray_sample_count, channels * source_viewpoints)


def create_direction_vectors(ray_directions, world_to_camera_target, ray_sample_count):
    """
    Normalize direction vectors and map to reference camera coordinates

    :param ray_directions: Ray directions from reference camera
    :type ray_directions: tensor[batch_size, ray_count, 3]

    :param world_to_camera_target: World coordinates to reference camera coordinates
    :type world_to_camera_target: tensor[batch_size, 3, 4]

    :return: Direction vectors rotates to reference camera coordinates
    :rtype: tensor[batch_size, ray_count, 3]
    """
    # fetch dimensions
    batch_size, ray_count, _ = ray_directions.shape

    # normalize direction vectors
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1).unsqueeze(-1)

    #
    world_to_camera_rotation = world_to_camera_target[:, :3, :3]

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


#
# Validate against legacy code
# ...
#

# def gen_dir_feature(w2c_ref, rays_dir):
#     """
#     Inputs:
#         c2ws: [1,v,4,4]
#         rays_pts: [N_rays, N_samples, 3]
#         rays_dir: [N_rays, 3]
#
#     Returns:
#
#     """
#     dirs = rays_dir @ w2c_ref[:3,:3].t() # [N_rays, 3]
#     return dirs
#
# def rendering_angle(pose_ref, rays_dir):
#     # rays angle
#     cos_angle = torch.norm(rays_dir, dim=-1)
#
#
#     # using direction
#     if pose_ref is not None:
#         angle = gen_dir_feature(pose_ref['w2cs'][0], rays_dir/cos_angle.unsqueeze(-1))  # view dir feature
#     else:
#         angle = rays_dir/cos_angle.unsqueeze(-1)
#
#     return angle
#
# batch_idx = 0
# batch_size = 2
# ray_count = 128
# ray_sample_count = 64
# all_ray_directions = torch.rand((batch_size, 1, ray_count, 3), dtype=torch.float32) * 2 - 1
# world_to_camera = torch.rand((2, batch_size, 3, 4), dtype=torch.float32)
#
#
# angle_theirs = rendering_angle({ 'w2cs': world_to_camera[:, batch_idx] }, all_ray_directions[batch_idx, 0])
# angle_mine = create_direction_vectors(all_ray_directions[:, 0], world_to_camera[0], ray_sample_count)
# angle_mine = angle_mine.view(batch_size, ray_count, ray_sample_count, 3)
# assert torch.allclose(angle_theirs, angle_mine[batch_idx, :, 12]), 'angles match'



# def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
#     '''
#         point_samples [N_rays N_sample 3]
#     '''
#
#     N_rays, N_samples = point_samples.shape[:2]
#     point_samples = point_samples.reshape(-1, 3)
#
#     # wrap to ref view
#     if w2c_ref is not None:
#         R = w2c_ref[:3, :3]  # (3, 3)
#         T = w2c_ref[:3, 3:]  # (3, 1)
#         point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)
#
#     if intrinsic_ref is not None:
#         # using projection
#         point_samples_pixel =  point_samples @ intrinsic_ref.t()
#         point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
#         if not lindisp:
#             point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
#         else:
#             point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
#     else:
#         # using bounding box
#         near, far = near.view(1,3), far.view(1,3)
#         point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
#     del point_samples
#
#     if pad>0:
#         W_feat, H_feat = (inv_scale+1)/4.0
#         point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
#         point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)
#
#     point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
#     return point_samples_pixel
#
# def build_color_volume(point_samples, pose_ref, imgs, img_feat=None, downscale=1.0, with_mask=False):
#     '''
#     point_world: [N_ray N_sample 3]
#     imgs: [N V 3 H W]
#     '''
#
#     device = imgs.device
#     N, V, C, H, W = imgs.shape
#     inv_scale = torch.tensor([W - 1, H - 1]).to(device)
#     C += with_mask
#     C += 0 if img_feat is None else img_feat.shape[2]
#     colors = torch.empty((*point_samples.shape[:2], V*C), device=imgs.device, dtype=torch.float)
#     for i,idx in enumerate(range(V)):
#
#         w2c_ref, intrinsic_ref = pose_ref['w2cs'][idx], pose_ref['intrinsics'][idx].clone()  # assume camera 0 is reference
#         point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale)[None]
#         grid = point_samples_pixel[...,:2]*2.0-1.0
#
#         # img = F.interpolate(imgs[:, idx], scale_factor=downscale, align_corners=True, mode='bilinear',recompute_scale_factor=True) if downscale != 1.0 else imgs[:, idx]
#         data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')
#         if img_feat is not None:
#             data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)
#
#         if with_mask:
#             in_mask = ((grid >-1.0)*(grid < 1.0))
#             in_mask = (in_mask[...,0]*in_mask[...,1]).float()
#             data = torch.cat((data,in_mask.unsqueeze(1)), dim=1)
#
#         colors[...,i*C:i*C+C] = data[0].permute(1, 2, 0)
#         del grid, point_samples_pixel, data
#
#     return colors
#
# def index_point_feature(volume_feature, ray_coordinate_ref, chunk=-1):
#         ''''
#         Args:
#             volume_color_feature: [B, G, D, h, w]
#             volume_density_feature: [B C D H W]
#             ray_dir_world:[3 ray_samples N_samples]
#             ray_coordinate_ref:  [3 N_rays N_samples]
#             ray_dir_ref:  [3 N_rays]
#             depth_candidates: [N_rays, N_samples]
#         Returns:
#             [N_rays, N_samples]
#         '''
#
#         device = volume_feature.device
#         H, W = ray_coordinate_ref.shape[-3:-1]
#
#
#         if chunk != -1:
#             features = torch.zeros((volume_feature.shape[1],H,W), device=volume_feature.device, dtype=torch.float, requires_grad=volume_feature.requires_grad)
#             grid = ray_coordinate_ref.view(1, 1, 1, H * W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
#             for i in range(0, H*W, chunk):
#                 features[:,i:i + chunk] = F.grid_sample(volume_feature, grid[:,:,:,i:i + chunk], align_corners=True, mode='bilinear')[0]
#             features = features.permute(1,2,0)
#         else:
#             grid = ray_coordinate_ref.view(-1, 1, H,  W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
#             features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')[:,:,0].permute(2,3,0,1).squeeze()#, padding_mode="border"
#         return features
#
#
#
# feat_h = 24
# feat_w = 36
# img_h = 4 * feat_h
# img_w = 4 * feat_w
# pad = 12
# depth_resolution = 128
# rays_pts = torch.rand((batch_size, 1, ray_count, ray_sample_count, 3), dtype=torch.float32) * 2 - 1
# rays_ndc = torch.rand((batch_size, 1, ray_count, ray_sample_count, 3), dtype=torch.float32)
# imgs = torch.rand((batch_size, 3, 3, img_h, img_w), dtype=torch.float32)
# volume_feature = torch.rand((batch_size, 8, depth_resolution, feat_h + 2 * pad, feat_w + 2 * pad), dtype=torch.float32)
#
# world_to_cameras = torch.rand((batch_size, 3, 3, 4), dtype=torch.float32)
# intrinsics = torch.rand((batch_size, 3, 3, 3), dtype=torch.float32)
# depth_bounds = torch.tensor([
#     [
#         [2, 6],
#         [2, 6],
#         [2, 6],
#     ],
#     [
#         [2, 6],
#         [2, 6],
#         [2, 6],
#     ]
# ], dtype=torch.float32)
# pose_ref = {
#     'w2cs': world_to_cameras[batch_idx],
#     'intrinsics': intrinsics[batch_idx],
# }
#
# ray_feats = index_point_feature(volume_feature[batch_idx:batch_idx+1], rays_ndc[batch_idx, 0])
#
# # volume_features.shape: (batch_size, ray_count, ray_sample_count, 8)
# volume_features_mine = interpolate_volume_encoding(
#     volume_feature,
#     rays_ndc
# )
# volume_features_mine = volume_features_mine.view(batch_size, ray_count, ray_sample_count, 8)
# assert torch.allclose(ray_feats, volume_features_mine[batch_idx]), 'volume features match'
#
#
# colour_volume_theirs = build_color_volume(rays_pts[batch_idx, 0], pose_ref, imgs[batch_idx:batch_idx+1], None, with_mask=False, downscale=1.0)
#
# source_image_colours_mine = interpolate_pixel_colours(
#     rays_pts[:, 0],
#     imgs,
#     world_to_cameras,
#     intrinsics,
#     depth_bounds,
# )
# source_image_colours_mine = source_image_colours_mine.view(batch_size, ray_count, ray_sample_count, 9)
# assert torch.allclose(colour_volume_theirs[...,:9], source_image_colours_mine[batch_idx]), 'colour volumes match'
#
#
#
# def gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False):
#     N_rays, N_samples = rays_pts.shape[:2]
#     if img_feat is not None:
#         feat_dim += img_feat.shape[1]*img_feat.shape[2]
#
#     if not use_color_volume:
#         input_feat = torch.empty((N_rays, N_samples, feat_dim), device=imgs.device, dtype=torch.float)
#         ray_feats = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
#         input_feat[..., :8] = ray_feats
#         input_feat[..., 8:] = build_color_volume(rays_pts, pose_ref, imgs, img_feat, with_mask=False, downscale=img_downscale)
#     else:
#         input_feat = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
#     return input_feat
#
# # TODO: check points feats
# # TODO cat the mask as well as data
#
#
#
# # check output parsing
# def raw2alpha(sigma):
#
#     alpha_softmax = F.softmax(sigma, 1)
#
#     alpha = 1. - torch.exp(-sigma)
#
#     T = torch.cumprod(
#         torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1),
#         -1
#     )[:, :-1]
#     weights = alpha * T  # [N_rays, N_samples]
#     return alpha, weights, alpha_softmax
#
# def raw2outputs(raw, white_bkgd=False):
#     """Transforms model's predictions to semantically meaningful values.
#     Args:
#         raw: [num_rays, num_samples along ray, 4]. Prediction from model.
#         z_vals: [num_rays, num_samples along ray]. Integration time.
#         rays_d: [num_rays, 3]. Direction of each ray.
#     Returns:
#         rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
#         disp_map: [num_rays]. Disparity map. Inverse of depth map.
#         acc_map: [num_rays]. Sum of weights along each ray.
#         weights: [num_rays, num_samples]. Weights assigned to each sampled color.
#         depth_map: [num_rays]. Estimated distance to object.
#     """
#
#     rgb = raw[..., :3] # [N_rays, N_samples, 3]
#
#     alpha, weights, alpha_softmax = raw2alpha(raw[..., 3])  # [N_rays, N_samples]
#     rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
#
#     acc_map = torch.sum(weights, -1)
#
#     if white_bkgd:
#         rgb_map = rgb_map + (1. - acc_map[..., None])
#     return rgb_map
#
#
# network_output = torch.rand((batch_size, ray_count, ray_sample_count, 4), dtype=torch.float32)
# # raw: (num_rays, num_samples along ray, 4)
# their_rgb_map = raw2outputs(network_output[batch_idx])
#
# # [batch_size, ray_count, ray_sample_count, 3]
# #
# #     :type prediction_density:  tensor[batch_size, ray_count, ray_sample_count]
# my_rgb_map = parse_nerf(network_output[...,:3], network_output[..., 3])
#
# assert torch.allclose(their_rgb_map, my_rgb_map[batch_idx]), 'rgb maps match'