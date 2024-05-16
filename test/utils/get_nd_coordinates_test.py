import torch
import pytest

from utils.ray_marching import get_nd_coordinates


def test_nd_coordinates_2x2_no_padding_or_transform():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are already normalized"""
    padding = 0
    point_samples = torch.tensor([[
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0, 1]], dtype=torch.float32)
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    assert torch.allclose(normalized_coordinates, point_samples), 'normalized co-ordinates expected to match'


def test_nd_coordinates_2x2_no_padding_or_transform_depth_range():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are normalized in (x, y) by depth is not"""
    padding = 0
    point_samples = torch.tensor([[
        [
            [1, 0, 2],
            [0, 1, 2]
        ],
        [
            [0, 1, .5],
            [1, 1, .5]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0.5, 2]], dtype=torch.float32)

    # image size is one pixel less
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    expected_ndc_samples = torch.tensor([[
        [
            [0.5, 0,   1],
            [0,   0.5, 1]
        ],
        [
            [0, 2, 0],
            [2, 2, 0]
        ],
    ]], dtype=torch.float32)
    assert torch.allclose(normalized_coordinates, expected_ndc_samples), 'normalized co-ordinates expected to match'


def test_nd_coordinates_2x2_no_padding_intrinsic_transform():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are already normalized, with intrinsic rotation"""
    padding = 0
    point_samples = torch.tensor([[
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [-1, 0,  0],
        [0,  -1, 0],
        [0,  0,  1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0, 1]], dtype=torch.float32)
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    expected_ndc_samples = torch.tensor([[
        [
            [-1, 0,  1],
            [0,  -1, 1]
        ],
        [
            [0,  -1,  1],
            [-1, -1,  1]
        ],
    ]], dtype=torch.float32)
    assert torch.allclose(normalized_coordinates, expected_ndc_samples), 'normalized co-ordinates expected to match'


def test_nd_coordinates_2x2_no_padding_extrinsic_transform_xflip():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are already normalized, with extrinsic x rotation"""
    padding = 0
    point_samples = torch.tensor([[
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [-1, 0,  0, 0],
        [0,  1,  0, 0],
        [0,  0,  1, 0],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0, 1]], dtype=torch.float32)
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    expected_ndc_samples = torch.tensor([[
        [
            [-1, 0,  1],
            [0,  1,  1]
        ],
        [
            [0,  1,  1],
            [-1, 1,  1]
        ],
    ]], dtype=torch.float32)
    assert torch.allclose(normalized_coordinates, expected_ndc_samples), 'normalized co-ordinates expected to match'


def test_nd_coordinates_2x2_no_padding_extrinsic_transform_translate():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are already normalized, with extrinsic translation"""
    padding = 0
    point_samples = torch.tensor([[
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [1,  0,  0,  1],
        [0,  1,  0, -1],
        [0,  0,  1,  1],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0, 2]], dtype=torch.float32)
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    expected_ndc_samples = torch.tensor([[
        [
            [1,  -0.5, 1],
            [0.5, 0,   1]
        ],
        [
            [0.5, 0, 1],
            [1,   0, 1]
        ],
    ]], dtype=torch.float32)
    assert torch.allclose(normalized_coordinates, expected_ndc_samples), 'normalized co-ordinates expected to match'

def test_nd_coordinates_2x2_padding_or_transform():
    """Test `get_nd_coordinates` for batch size 1 in the case coordinates are already normalized"""
    padding = 2
    point_samples = torch.tensor([[
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1]
        ],
    ]], dtype=torch.float32)
    world_to_camera_reference = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]], dtype=torch.float32)
    intrinsic_reference = torch.tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.float32)
    depth_bounds_reference = torch.tensor([[0, 1]], dtype=torch.float32)
    image_size = torch.tensor([1, 1], dtype=torch.float32)

    normalized_coordinates = get_nd_coordinates(
        point_samples,
        world_to_camera_reference,
        intrinsic_reference,
        depth_bounds_reference,
        image_size,
        padding,
    )

    expected_ndc_samples = torch.tensor([[
        [
            [5/9.0, 4/9.0, 1],
            [4/9.0, 5/9.0, 1]
        ],
        [
            [4/9.0, 5/9.0, 1],
            [5/9.0, 5/9.0, 1]
        ],
    ]], dtype=torch.float32)
    assert torch.allclose(normalized_coordinates, expected_ndc_samples), 'normalized co-ordinates expected to match'
