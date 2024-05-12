import torch
import pytest

from utils.ray_marching import create_rays, create_ray_offsets_deterministic, create_ray_offsets_sampled


def test_ray_offsets_sampled_single():
    """Test `ray_offsets_sampled` for batch size 1"""
    height = 2
    width = 3
    ray_count = 3

    ray_offset_function = create_ray_offsets_sampled(height, width, ray_count)
    x_offsets, y_offsets = ray_offset_function()

    assert x_offsets.shape == (1, ray_count), 'x offsets have required shape'
    assert y_offsets.shape == (1, ray_count), 'y offsets have required shape'

    assert x_offsets.min() >= 0 and x_offsets.max() < width, 'x offsets sampled in required range'
    assert y_offsets.min() >= 0 and y_offsets.max() < height, 'y offsets sampled in required range'


def test_ray_offsets_sampled_batch():
    """Test `ray_offsets_sampled` for larger batch size"""
    height = 2
    width = 3
    ray_count = 3
    batch_size = 3

    ray_offset_function = create_ray_offsets_sampled(height, width, ray_count, batch_size=batch_size)
    x_offsets, y_offsets = ray_offset_function()

    assert x_offsets.shape == (batch_size, ray_count), 'x offsets have required shape'
    assert y_offsets.shape == (batch_size, ray_count), 'y offsets have required shape'

    # TODO account for floating point errors in assertions
    assert x_offsets.min() >= 0 and x_offsets.max() < width, 'x offsets sampled in required range'
    assert y_offsets.min() >= 0 and y_offsets.max() < height, 'y offsets sampled in required range'


def test_ray_offsets_deterministic_single():
    """Test `ray_offsets_deterministic` for batch size 1"""
    height = 2
    width = 3
    ray_count = 2

    ray_offset_function = create_ray_offsets_deterministic(height, width, ray_count)
    x_offsets, y_offsets = ray_offset_function()

    assert x_offsets.shape == (1, width * height), 'x offsets have required shape'
    assert y_offsets.shape == (1, width * height), 'y offsets have required shape'

    expected_x_offsets = torch.tensor([[0.0, 1.0, 2.0]]).repeat((1, height))
    expected_y_offsets = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(x_offsets, expected_x_offsets), 'x offsets generated in required range'
    assert torch.allclose(y_offsets, expected_y_offsets), 'y offsets generated in required range'


def test_ray_offsets_deterministic_batch():
    """Test `ray_offsets_deterministic` for larger batch size"""
    height = 2
    width = 3
    ray_count = 2
    batch_size = 2

    ray_offset_function = create_ray_offsets_deterministic(height, width, ray_count, batch_size=batch_size)
    x_offsets, y_offsets = ray_offset_function()

    assert x_offsets.shape == (batch_size, width * height), 'x offsets have required shape'
    assert y_offsets.shape == (batch_size, width * height), 'y offsets have required shape'

    expected_x_offsets = torch.tensor([[0.0, 1.0, 2.0]]).repeat((1, height))
    expected_y_offsets = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(x_offsets[0], expected_x_offsets), 'x offsets generated in required range for first batch'
    assert torch.allclose(y_offsets[0], expected_y_offsets), 'y offsets generated in required range for first batch'
    assert torch.allclose(x_offsets[1], expected_x_offsets), 'x offsets generated in required range for second batch'
    assert torch.allclose(y_offsets[1], expected_y_offsets), 'y offsets generated in required range for second batch'


def test_create_rays_intrinsic_trivial():
    """Test `create_rays` for batch size 1 and trivial intrinsic and extrinsic parameters"""
    height = 2
    width = 3
    ray_count = 2

    # function to generate ray pixel coordinates
    ray_offset_function = create_ray_offsets_deterministic(height, width, ray_count, dtype=torch.float32)

    # trivial matrices
    intrinsic_params = torch.tensor([
        # identity matrix
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ], dtype=torch.float32)
    camera_to_world_params = torch.eye(4, dtype=torch.float32)[:3, :4] \
        .unsqueeze(0)

    rays_origin, ray_directions, pixel_coordinates = create_rays(
        ray_offset_function,
        intrinsic_params,
        camera_to_world_params,
    )

    assert torch.allclose(rays_origin, torch.zeros((1, 3)))
    assert torch.allclose(ray_directions, torch.tensor([
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
    ], dtype=torch.float32))
    assert torch.allclose(pixel_coordinates, torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2]
    ], dtype=torch.float32))


def test_create_rays_intrinsic_batch():
    """Test `create_rays` for batch size 12and non-trivial intrinsic  parameters"""
    height = 2
    width = 3
    ray_count = 2
    batch_size = 2

    # function to generate ray pixel coordinates
    ray_offset_function = create_ray_offsets_deterministic(height, width, ray_count, batch_size=batch_size, dtype=torch.float32)

    # non-trivial intrinsic matrices
    intrinsic_params = torch.tensor([
        # first item in batch has no offset, scales by 2
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ],
        # second item in batch has offset 1, scales by 1
        [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]
    ], dtype=torch.float32)

    # trivial matrices
    camera_to_world_params = torch.eye(4, dtype=torch.float32)[:3, :4] \
        .unsqueeze(0) \
        .repeat(batch_size, 1, 1)

    rays_origin, ray_directions, pixel_coordinates = create_rays(
        ray_offset_function,
        intrinsic_params,
        camera_to_world_params,
    )

    # verify origins
    assert torch.allclose(rays_origin, torch.zeros((batch_size, 3)))

    # verify ray directions
    assert torch.allclose(ray_directions[0], torch.tensor([
        [0,  0,  1],
        [.5, 0,  1],
        [1,  0,  1],
        [0,  .5, 1],
        [.5, .5, 1],
        [1,  .5, 1],
    ], dtype=torch.float32))
    assert torch.allclose(ray_directions[1], torch.tensor([
        [-1, -1, 1],
        [0,  -1, 1],
        [1,  -1, 1],
        [-1, 0,  1],
        [0,  0,  1],
        [1,  0,  1],
    ], dtype=torch.float32))

    # verify pixel coordinates
    expected_pixel_coordinates = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2]
    ], dtype=torch.float32) \
        .repeat((batch_size, 1, 1))
    assert torch.allclose(pixel_coordinates, expected_pixel_coordinates)


def test_create_rays_extrinsic_batch():
    """Test `create_rays` for batch size 2 and non-trivial extrinsic  parameters"""
    height = 2
    width = 3
    ray_count = 2
    batch_size = 2

    # function to generate ray pixel coordinates
    ray_offset_function = create_ray_offsets_deterministic(height, width, ray_count, batch_size=batch_size, dtype=torch.float32)

    # trivial matrices
    intrinsic_params = torch.eye(3, dtype=torch.float32) \
        .unsqueeze(0) \
        .repeat((batch_size, 1, 1))

    # non-trivial extrinsics
    camera_to_world_params = torch.tensor([
        # first item in batch flips about x-axis, has origin (0, 0, 0)
        [
            [-1, 0, 0, 0],
            [0,  1, 0, 0],
            [0,  0, 1, 0],
        ],
        # second item does not rotate, has origin (0, 1, 0)
        [
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    ], dtype=torch.float32)

    rays_origin, ray_directions, pixel_coordinates = create_rays(
        ray_offset_function,
        intrinsic_params,
        camera_to_world_params,
    )

    assert torch.allclose(rays_origin[0], torch.tensor([0, 0, 0], dtype=torch.float32))
    assert torch.allclose(rays_origin[1], torch.tensor([0, 1, 0], dtype=torch.float32))

    expected_ray_directions_first = torch.tensor([
        [0,  0, 1],
        [-1, 0, 1],
        [-2, 0, 1],
        [0,  1, 1],
        [-1, 1, 1],
        [-2, 1, 1],
    ], dtype=torch.float32)
    expected_ray_directions_second = torch.tensor([
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
    ], dtype=torch.float32)
    assert torch.allclose(ray_directions[0], expected_ray_directions_first)
    assert torch.allclose(ray_directions[1], expected_ray_directions_second)

    expected_pixel_coordinates = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2]
    ], dtype=torch.float32) \
        .repeat((batch_size, 1, 1))
    assert torch.allclose(pixel_coordinates, expected_pixel_coordinates)

#TODO data type and deveice tests