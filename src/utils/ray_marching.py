import torch


def ray_offsets_sampled(
    height,
    width,
    ray_count,
    batch_size=1,
    dtype=None,
    device=None,
    generator=None
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
    :param torch.Generator generator: Random number generator for testing

    :return tensor[batch_size, ray_count]: Sampled pixel offsets
    """
    return torch.randint(0, width, (batch_size, ray_count), dtype=dtype, device=device, generator=generator), \
        torch.randint(0, height, (batch_size, ray_count), dtype=dtype, device=device, generator=generator)


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
        torch.linspace(0, width - 1, width, dtype=dtype, device=device),
        indexing='ij'
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


def ray_offsets_row(
    width,
    height,
    row_index,
    row_batch_size,
    batch_size=1,
    dtype=None,
    device=None
):
    """
    Compute a row pixel offsets for a batch of images.
    Assumes image dimensions are the same across a batch.

    :param int width: Image width
    :param int row_index: Index of row to return
    :param int row_batch_size: Rows to process in parallel
    :param int batch_size: Batch size
    :param torch.dtype dtype: Data type for return tensor
    :param torch.device device: Device for return tensor

    :return tensor[batch_size, row_batch_size * width]: Pixel offsets for each (x, y) in each image in batch
    """
    y_offsets, x_offsets = torch.meshgrid(
        torch.linspace(row_index, row_index + (row_batch_size - 1), row_batch_size, dtype=dtype, device=device),
        torch.linspace(0, width - 1, width, dtype=dtype, device=device),
        indexing='ij'
    )
    x_offsets = x_offsets.reshape(width * row_batch_size) \
        .unsqueeze(0) \
        .repeat((batch_size, 1))
    y_offsets = y_offsets.reshape(width * row_batch_size) \
        .unsqueeze(0) \
        .repeat((batch_size, 1))

    return x_offsets, y_offsets


def create_ray_offsets_row(*args, **kwargs):
    return lambda : ray_offsets_row(*args, **kwargs)


def create_rays(ray_offset_function, intrinsics, cameras_to_world):
    """
    Create rays in world co-ordinates for a camera with given intrinsic and extrinsic parameters.


    :param ray_offset_function: Function returning pixel offsets for rays
    :type ray_offset_function: Callable

    :param intrinsics: Camera intrinsic parameters
    :type intrinsics: tensor[batch_size, 3, 3]

    :param cameras_to_world: Mapping from camera frame to world frame
    :type cameras_to_world: tensor[batch_size, 3, 4]

    :return: Ray origins, directions and pixel co-ordinates
    :rtype: tuple[tensor[batch_size, 3], tensor[batch_size, ray_count, 3], tensor[batch_size, 2, ray_count]]
    """
    # move rays to same device as camera
    device = intrinsics.device
    dtype = cameras_to_world.dtype

    # x_offsets.shape [batch_size, ray_count]
    # y_offsets.shape [batch_size, ray_count]
    x_offsets, y_offsets = ray_offset_function()

    # add a dimension at end of intrinsics because PyTorch broadcast
    # semantics would prepend instead
    # this allows batch-wise operations on pixel offsets
    intrinsics = intrinsics.unsqueeze(-1)

    # scale offsets with intrinsic parameters and append ones for z values
    # camera_directions.shape [batch_size, ray_count, 3]
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

    # ray_directions.shape (batch_size, ray_count, 3)
    # pixel_coordinates.shape (batch_size, 2, ray_count)
    ray_directions = camera_directions @ rotations.transpose(-2, -1)
    pixel_coordinates = torch.stack((y_offsets, x_offsets), dim=1)

    # ray origins are translation of cameras to world
    rays_origin = translations.clone()

    return rays_origin, ray_directions, pixel_coordinates


def generate_depth_samples(depth_bounds, ray_count, ray_sample_count, generator=None):
    """
    Generate depth sample points for a batch of rays.


    :param depth_bounds: Depth bounds for batch
    :type depth_bounds: tensor[batch_size, 2]

    :param ray_count: Number of rays
    :type ray_count: int

    :param ray_sample_count: Number of samples per ray
    :type ray_sample_count: int

    :param generator: Random number generator, used for testing
    :type generator: torch.Generator

    :return: Samples for each ray for batch
    :rtype: tensor[batch_size, ray_count, ray_sample_count]
    """
    batch_size, _ = depth_bounds.shape
    dtype, device = depth_bounds.dtype, depth_bounds.device

    # generate linear space from near to far
    # depth_min.shape (batch_size)
    # depth_max.shape (batch_size)
    depth_min, depth_max = depth_bounds[:, 0], depth_bounds[:, 1]

    # sample_grid.shape (ray_sample_count, batch_size)
    sample_grid = torch.linspace(0, 1, steps=ray_sample_count, dtype=dtype, device=device) \
        .unsqueeze(dim=1) \
        .expand((-1, batch_size))

    # depth_samples.shape (batch_size, ray_count, ray_sample_count)
    depth_samples = (depth_min * (1. - sample_grid) + depth_max * sample_grid) \
        .transpose(0, 1) \
        .unsqueeze(1) \
        .expand(-1, ray_count, -1)

    # get intervals between samples
    sample_midpoints = .5 * (depth_samples[..., 1:] + depth_samples[..., :-1])
    sample_upper = torch.cat([sample_midpoints, depth_samples[..., -1:]], -1)
    sample_lower = torch.cat([depth_samples[..., :1], sample_midpoints], -1)
    sample_noise = torch.rand(depth_samples.shape, generator=generator, device=device, dtype=dtype)

    # return.shape: (batch_size, ray_count, ray_sample_count)
    return sample_lower + (sample_upper - sample_lower) * sample_noise


def get_nd_coordinates(
    point_samples,
    world_to_camera_target,
    intrinsic_target,
    depth_bounds_target,
    image_size,
    padding,
):
    """
    Map 3D co-ordinates to normalized device co-ordinates.


    :param point_samples: 3D co-ordinates of samples for rays
    :type point_samples: tensor[batch_size, ray_count, ray_sample_count, 3]

    :param world_to_camera_target: World to camera matrix for target view
    :type world_to_camera_target: tensor[batch_size, 3, 4]

    :param intrinsic_target: Intrinsic parameters for target view
    :type intrinsic_target: tensor[batch_size, 3, 3]

    :param depth_bounds_target: Depth bounds for target view
    :type depth_bounds_target: tensor[batch_size, 2]

    :param image_size: Image dimensions minus one, [width - 1, height - 1]
    :type image_size: tensor[2]

    :param padding: Padding used in feature warps
    :type padding: int

    :return: 3D normalized device co-ordinates of samples for rays
    :rtype: tensor[batch_size, ray_count, ray_sample_count, 3]
    """
    # point_samples.shape (batch_size, ray_count, ray_sample_count, 3)
    batch_size, ray_count, ray_sample_count, _ = point_samples.shape
    depth_min_target, depth_max_target = \
        depth_bounds_target[:, 0].view(batch_size, 1), \
            depth_bounds_target[:, 1].view(batch_size, 1)

    # reshape to be a list of points
    point_samples = point_samples.view(batch_size, -1, 3)

    # map from world to target view
    target_rotation = world_to_camera_target[:, :3, :3]
    target_translation = world_to_camera_target[:, :3, 3:]
    point_samples = torch.matmul(point_samples, target_rotation.transpose(dim0=1, dim1=2)) + target_translation.reshape(batch_size, 1, 3)

    # map to pixel co-ordinates,
    point_samples_pixel = point_samples @ intrinsic_target.transpose(dim0=1, dim1=2)

    # TODO: what if z values are less than 1 here?
    point_samples_pixel[..., :2] = point_samples_pixel[..., :2] / point_samples_pixel[..., -1:] / image_size.view(1, 1, 2)
    point_samples_pixel[..., 2] = (point_samples_pixel[..., 2] - depth_min_target) / (depth_max_target - depth_min_target)

    if padding > 0:
        # TODO: change hard coded downscaling factor from feature net
        width_feature, height_feature = (image_size + 1) / 4.0
        point_samples_pixel[..., 1] = point_samples_pixel[..., 1] * height_feature / (height_feature + padding * 2) + padding / (height_feature + padding * 2)
        point_samples_pixel[..., 0] = point_samples_pixel[..., 0] * width_feature / (width_feature + padding * 2) + padding / (width_feature + padding * 2)

    point_samples_pixel = point_samples_pixel.view(batch_size, ray_count, ray_sample_count, 3)
    return point_samples_pixel


def march_rays(
    mvs_images,
    intrinsic_params,
    cameras_to_world,
    world_to_cameras,
    ray_offset_function,
    ray_count,
    ray_sample_count,
    depth_bounds,
    padding=12
):
    # fetch tensor dimensions from images
    batch_size, viewpoints, channels, height, width = mvs_images.shape
    dtype, device = mvs_images.dtype, mvs_images.device

    # get target camera data
    target_viewpoint_index = viewpoints - 1
    world_to_camera_target = world_to_cameras[:, target_viewpoint_index]
    intrinsic_target = intrinsic_params[:, target_viewpoint_index]
    depth_bounds_target = depth_bounds[:, target_viewpoint_index]
    mvs_image_target = mvs_images[:, target_viewpoint_index]

    # allocate for just the target viewpoint
    # may wish to use sources in future
    all_ray_directions = torch.empty((batch_size, 1, ray_count, 3), dtype=dtype, device=device)
    all_ray_origins = torch.empty((batch_size, 1, ray_count, 3), dtype=dtype, device=device)
    all_depth_samples = torch.empty((batch_size, 1, ray_count, ray_sample_count), dtype=dtype, device=device)
    all_point_samples = torch.empty((batch_size, 1, ray_count, ray_sample_count, 3), dtype=dtype, device=device)
    all_point_samples_ndc = torch.empty((batch_size, 1, ray_count, ray_sample_count, 3), dtype=dtype, device=device)
    all_image_colours = torch.empty((batch_size, 1, ray_count, 3), dtype=dtype, device=device)
    for viewpoint in range(target_viewpoint_index, target_viewpoint_index + 1):
        intrinsics = intrinsic_params[:, viewpoint]
        camera_to_world = cameras_to_world[:, viewpoint]

        # TODO generate all rays directions and origins for batches and viewpoints
        # then pass into this function, which will do the depth sampling

        # rays_origin.shape: (batch_size, 3)
        # ray_directions.shape: (batch_size, ray_count, 3)
        # pixel_coordinates.shape: (batch_size, 2, ray_count)
        rays_origin, ray_directions, pixel_coordinates = create_rays(ray_offset_function, intrinsics, camera_to_world)

        # fetch colours of target image at these pixel offsets
        # TODO pixel coordinates aren't used elsewhere, can return a different shape to save this op
        # TODO what about when sampling everything, can be cheap?
        pixel_coordinates_int = pixel_coordinates.long()
        for batch in range(batch_size):
            x_coordinates = pixel_coordinates_int[batch, 0]
            y_coordinates = pixel_coordinates_int[batch, 1]
            target_image_colours = mvs_image_target[batch, :, x_coordinates, y_coordinates]

            all_image_colours[batch, 0, :, :] = target_image_colours.t()

        # add ray origins to tensor
        rays_origin = rays_origin.unsqueeze(dim=1)

        # rays_origin.shape: (batch_size, ray_count, 3)
        rays_origin = rays_origin.expand(-1, ray_count, -1)

        # add ray directions to tensor
        all_ray_directions[:, 0] = ray_directions

        # generate depth samples
        # depth_samples.shape: (batch_size, ray_count, ray_sample_count)
        # TODO make the samples argument to this function
        depth_samples = generate_depth_samples(depth_bounds[:, viewpoint], ray_count, ray_sample_count)

        # point_samples.shape (batch_size, ray_count, ray_sample_count, 3)
        point_samples = rays_origin.unsqueeze(dim=2) + depth_samples.unsqueeze(-1) * ray_directions.unsqueeze(dim=2)


        # TODO move this outside function
        image_size = torch.tensor([width - 1, height - 1], device=mvs_images.device, dtype=mvs_images.dtype)
        points_ndc = get_nd_coordinates(
            point_samples,
            world_to_camera_target,
            intrinsic_target,
            depth_bounds_target,
            image_size,
            padding
        )

        all_ray_origins[:, 0] = rays_origin
        all_depth_samples[:, 0] = depth_samples
        all_point_samples[:, 0] = point_samples
        all_point_samples_ndc[:, 0] = points_ndc

    return all_ray_directions, \
        all_ray_origins, \
        all_depth_samples, \
        all_point_samples, \
        all_point_samples_ndc, \
        all_image_colours


def get_rays_mvs(H, W, intrinsic, c2w, N=1024, isRandom=False, is_precrop_iters=False, chunk=-1, idx=-1):

    device = c2w.device
    if isRandom:
        if is_precrop_iters and torch.rand((1,)) > 0.3:
            xs, ys = torch.randint(W//6, W-W//6, (N,), generator=generator).float().to(device), torch.randint(H//6, H-H//6, (N,), generator=generator).float().to(device)
        else:
            xs, ys = randx.float().to(device), randy.float().to(device)
    else:
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        if chunk>0:
            ys, xs = ys[idx*chunk:(idx+1)*chunk], xs[idx*chunk:(idx+1)*chunk]
        ys, xs = ys.to(device), xs.to(device)

    dirs = torch.stack([(xs-intrinsic[0,2])/intrinsic[0,0], (ys-intrinsic[1,2])/intrinsic[1,1], torch.ones_like(xs)], -1) # use 1 instead of -1


    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].clone()
    pixel_coordinates = torch.stack((ys,xs)) # row col
    return rays_o, rays_d, pixel_coordinates


def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
    '''
        point_samples [N_rays N_sample 3]
    '''

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    # wrap to ref view
    if w2c_ref is not None:
        R = w2c_ref[:3, :3]  # (3, 3)
        T = w2c_ref[:3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)

    if intrinsic_ref is not None:
        # using projection
        point_samples_pixel =  point_samples @ intrinsic_ref.t()
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(1,3), far.view(1,3)
        point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
    del point_samples

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
        point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

    point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
    return point_samples_pixel


def build_rays(imgs, depths, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=0, is_precrop_iters=False, ref_idx=0, importanceSampling=False, with_depth=False, is_volume=False):
    '''

    Args:
        imgs: [N V C H W]
        depths: [N V H W]
        poses: w2c c2w intrinsic [N V 4 4] [B V levels 3 3)]
        init_depth_min: [B D H W]
        depth_interval:
        N_rays: int
        N_samples: same as D int
        level: int 0 == smalest
        near_fars: [B D 2]

    Returns:
        [3 N_rays N_samples]
    '''

    device = imgs.device

    N, V, C, H, W = imgs.shape
    w2c_ref, intrinsic_ref = pose_ref['w2cs'][ref_idx], pose_ref['intrinsics'][ref_idx]  # assume camera 0 is reference
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    ray_coordinate_ref = []
    near_ref, far_ref = pose_ref['near_fars'][ref_idx, 0], pose_ref['near_fars'][ref_idx, 1]
    ray_coordinate_world, ray_dir_world, colors, depth_candidates = [],[],[],[]
    rays_os, rays_ds, cos_angles, rays_depths = [],[],[],[]

    for i in range(0,1):
        intrinsic = intrinsics[i]  #!!!!!! assuming batch size equal to 1
        c2w, w2c = c2ws[i].clone(), w2cs[i].clone()

        rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic, c2w, N_rays, is_precrop_iters=is_precrop_iters)   # [N_rays 3]


        # direction
        ray_dir_world.append(rays_d)    # toward camera [N_rays 3]

        # position
        rays_o = rays_o.reshape(1, 3)
        rays_o = rays_o.expand(N_rays, -1)
        rays_os.append(rays_o)

        # colors
        pixel_coordinates_int = pixel_coordinates.long()
        color = imgs[0, i, :, pixel_coordinates_int[0], pixel_coordinates_int[1]] # [3 N_rays]
        colors.append(color)

        if depths.shape[2] != 1:
            rays_depth = depths[0,i,pixel_coordinates_int[0], pixel_coordinates_int[1]]
            rays_depths.append(rays_depth)

        # travel along the rays
        if with_depth:
            depth_candidate = near_fars[pixel_coordinates_int[0], pixel_coordinates_int[1]].reshape(-1,1) #  [ray_samples N_samples]
        else:
            if importanceSampling:
                near, far = rays_depth - 0.1, rays_depth + 0.1
                near, far = near.view(N_rays, 1), far.view(N_rays, 1)
            else:
                near, far = near_fars[0, i, 0], near_fars[0, i, 1]

            t_vals = torch.linspace(0., 1., steps=N_samples).view(1,N_samples).to(device)
            depth_candidate = near * (1. - t_vals) + far * (t_vals)
            depth_candidate = depth_candidate.expand([N_rays, N_samples])

            # get intervals between samples
            mids = .5 * (depth_candidate[..., 1:] + depth_candidate[..., :-1])
            upper = torch.cat([mids, depth_candidate[..., -1:]], -1)
            lower = torch.cat([depth_candidate[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(depth_candidate.shape, device=device)
            depth_candidate = lower + (upper - lower) * t_rand

        point_samples = rays_o.unsqueeze(1) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(1)   #  [ray_samples N_samples 3 ]
        depth_candidates.append(depth_candidate) #  [ray_samples N_rays]

        # position
        ray_coordinate_world.append(point_samples)  # [ray_samples N_samples 3] xyz in [0,1]
        points_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=near_ref, far=far_ref, pad=pad)

        ray_coordinate_ref.append(points_ndc)

    ndc_parameters = {'w2c_ref':w2c_ref, 'intrinsic_ref':intrinsic_ref, 'inv_scale':inv_scale, 'near':near_ref, 'far':far_ref}
    colors = torch.cat(colors, dim=1).permute(1,0)
    rays_depths = torch.cat(rays_depths) if len(rays_depths)>0 else None
    depth_candidates = torch.cat(depth_candidates, dim=0)
    ray_dir_world = torch.cat(ray_dir_world, dim=0)
    ray_coordinate_world = torch.cat(ray_coordinate_world, dim=0)
    rays_os = torch.cat(rays_os, dim=0).permute(1,0)
    ray_coordinate_ref = torch.cat(ray_coordinate_ref, dim=0)

    return ray_coordinate_world, ray_dir_world, colors, ray_coordinate_ref, depth_candidates, rays_os, rays_depths, ndc_parameters


# cameras_to_world_source = torch.rand((2, 3, 4))
# intrinsic_source = torch.rand((2, 3, 3))
# world_to_camera_reference = torch.rand((2, 3, 4))
# intrinsic_reference = torch.rand((2, 3, 3))
#
# # depth samples
# batch_size = 2
# ray_count = 6
# ray_sample_count = 3
# height = 2
# width = 3
#
#
# mvs_images = torch.rand((batch_size, 4, 3, height, width))
# #intrinsic_params = torch.eye(3).unsqueeze(0).repeat(4, 1, 1).unsqueeze(0).repeat(2, 1, 1, 1)
# intrinsic_params = torch.rand((batch_size, 4, 3, 3))
# cameras_to_world = torch.rand((batch_size, 4, 3, 4))
# world_to_cameras = torch.rand((batch_size, 4, 3, 4))
# depths = torch.rand((1, 1, 1))
# r = torch.rand((4, 2))
# depth_bounds = torch.stack(
#     [
#         r, r
#     ], dim=0
# ) * 2 + 2
#
#
# ray_offset_function = create_ray_offsets_deterministic(
#     height,
#     width,
#     ray_count,
#     batch_size=2,
#     dtype=torch.float32
# )
# all_ray_directions, all_ray_origins, all_depth_samples, all_point_samples, all_point_samples_ndc = march_rays(
#     mvs_images,
#     intrinsic_params,
#     cameras_to_world,
#     world_to_cameras,
#     ray_offset_function,
#     ray_count,
#     ray_sample_count,
#     depth_bounds,
#     padding=2
# )
#
# pose_ref = {'w2cs': world_to_cameras[0], 'intrinsics': intrinsic_params[0],
#             'c2ws': cameras_to_world[0],'near_fars':depth_bounds[0]}
#
# rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters = build_rays(
#     mvs_images[:1],
#     depths,
#     pose_ref,
#     world_to_cameras[0],
#     cameras_to_world[0],
#     intrinsic_params[0],
#     depth_bounds,
#     ray_count,
#     ray_sample_count,
#     pad=2
# )
#
# assert torch.allclose(all_ray_directions[0], rays_dir), 'directions close'
# assert torch.allclose(all_ray_origins[0].permute(0, 2, 1), rays_o), 'origins close'
# assert torch.allclose(all_point_samples[0], rays_pts), 'pts close'
# assert torch.allclose(all_point_samples_ndc[0], rays_NDC), 'pts ndc close'
#
#
#
# pose_ref = {'w2cs': world_to_cameras[1], 'intrinsics': intrinsic_params[1],
#             'c2ws': cameras_to_world[1],'near_fars':depth_bounds[1]}
#
# rays_pts, rays_dir, _, rays_NDC, _, rays_o, _, _ = build_rays(
#     mvs_images[1:],
#     depths,
#     pose_ref,
#     world_to_cameras[1],
#     cameras_to_world[1],
#     intrinsic_params[1],
#     depth_bounds,
#     ray_count,
#     ray_sample_count,
#     pad=2
# )
#
# assert torch.allclose(all_ray_directions[1], rays_dir), 'directions close'
# assert torch.allclose(all_ray_origins[1].permute(0, 2, 1), rays_o), 'origins close'
# assert torch.allclose(all_point_samples[1], rays_pts), 'pts close'
# assert torch.allclose(all_point_samples_ndc[1], rays_NDC), 'pts ndc close'
